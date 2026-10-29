"""Paired uniform3/uniform6 Qwen inference; keeps every raw output for review."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core import PROMPT, atomic_json, normalize
from app.sampling import sample_frames

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
INSTRUCTION = PROMPT + " Keep each JSON text field under 40 Korean characters."
MODES = {"uniform3": 3, "uniform6": 6}


def memory_rss():
    """Linux process RSS; /proc/self works with virtualized process IDs too."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def available_max(values):
    return max((v for v in values if v is not None), default=None)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def messages_for(frames):
    # No clip identifier, path, annotations or camera metadata enters this message.
    content = []
    for timestamp, image in frames:
        content.extend([{"type": "text", "text": f"Frame at {timestamp:.2f}s"},
                        {"type": "image", "image": image}])
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


def select_clips(clips, ids):
    if not ids:
        return clips
    wanted = ids.split(",")
    by_id = {c["id"]: c for c in clips}
    if len(set(wanted)) != len(wanted) or any(x not in by_id for x in wanted):
        raise ValueError("Unknown or duplicate clip ID")
    return [by_id[x] for x in wanted]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ids", help="Optional comma-separated diagnostic subset; default all clips")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("Invalid run ID")
    if not 1 <= args.batch_size <= 4 or not 1 <= args.threads <= 16:
        parser.error("Use batch size 1-4 and threads 1-16")
    manifest_path = ROOT / "samples/manifest.json"
    clips = select_clips(json.loads(manifest_path.read_text())["clips"], args.ids)
    folder = ROOT / "experiments" / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preparing", "model": MODEL, "revision": REVISION,
        "clip_ids": [c["id"] for c in clips], "modes": MODES,
        "batch_size": args.batch_size, "threads": args.threads,
        "dtype": "bfloat16", "device": "cpu", "attention": "sdpa",
        "max_image_size": [576, 324], "max_new_tokens": 260,
        "do_sample": False, "padding_side": "left", "seed": 0,
        "prompt": INSTRUCTION,
        "manifest_sha256": sha256(manifest_path),
        "code_sha256": {p: sha256(ROOT / p) for p in
                        ("scripts/compare_vlm.py", "app/sampling.py", "app/core.py")},
        "python": platform.python_version(), "completed_inferences": 0,
        "evaluation_note": "Outputs and category changes are not accuracy. Human review required.",
    }
    atomic_json(folder / "run.json", config)
    run_started = time.perf_counter()
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        config["packages"] = {n: importlib.metadata.version(n) for n in
                              ("torch", "transformers", "av", "Pillow")}
        torch.manual_seed(0)
        torch.set_num_threads(args.threads)
        started = time.perf_counter()
        model_path = snapshot_download(MODEL, revision=REVISION,
                                       allow_patterns=["*.json", "*.txt", "*.safetensors"])
        config["download_or_cache_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        engine = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").eval()
        config["model_load_seconds"] = time.perf_counter() - started
        config["status"] = "running"
        atomic_json(folder / "run.json", config)
        records = {mode: [] for mode in MODES}
        print("MODEL LOADED", flush=True)
        for offset in range(0, len(clips), args.batch_size):
            batch = clips[offset:offset + args.batch_size]
            # Alternate mode order between groups to avoid always favouring one warm state.
            order = list(MODES) if (offset // args.batch_size) % 2 == 0 else list(reversed(MODES))
            for mode in order:
                prepared, messages = [], []
                started = time.perf_counter()
                for clip in batch:
                    frames = sample_frames(ROOT / clip["file"], 0, clip["duration"], MODES[mode])
                    for _, image in frames:
                        image.thumbnail((576, 324))
                    messages.append(messages_for(frames))
                    prepared.append(frames)
                extraction_seconds = time.perf_counter() - started
                # RSS samples include model and processor; not incremental per-clip memory.
                rss = [memory_rss()]
                started = time.perf_counter()
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", padding=True)
                rss.append(memory_rss())
                with torch.inference_mode():
                    outputs = engine.generate(**inputs, max_new_tokens=260, do_sample=False)
                rss.append(memory_rss())
                raw_outputs = processor.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                seconds = time.perf_counter() - started
                batch_id = f"{mode}-{offset:03d}"
                for clip, frames, raw in zip(batch, prepared, raw_outputs):
                    evidence = []
                    evidence_dir = folder / mode / clip["id"]
                    evidence_dir.mkdir(parents=True)
                    for index, (timestamp, image) in enumerate(frames):
                        path = evidence_dir / f"frame-{index}.png"
                        image.save(path)
                        evidence.append({"time": timestamp, "file": str(path.relative_to(folder)),
                                         "sha256": sha256(path), "size": list(image.size)})
                        image.close()
                    result = normalize(raw) | {
                        "clip_id": clip["id"], "mode": mode, "review": "unreviewed",
                        "input_video_sha256": sha256(ROOT / clip["file"]),
                        "frame_times": [t for t, _ in frames], "evidence_frames": evidence,
                        "batch_id": batch_id, "batch_size": len(batch),
                        "batch_extraction_seconds": extraction_seconds,
                        "batch_inference_seconds": seconds,
                        "amortized_inference_seconds": seconds / len(batch),
                        "rss_sample_max_bytes": available_max(rss),
                        "memory_note": "Max of three process RSS samples, not true peak or per-clip memory.",
                    }
                    atomic_json(folder / mode / (clip["id"] + ".json"), result)
                    records[mode].append(result)
                    config["completed_inferences"] += 1
                    print(mode, clip["id"], result["category"], result["summary_ko"], flush=True)
                atomic_json(folder / "run.json", config)
                print("BATCH", batch_id, "seconds", round(seconds, 2), flush=True)
                del inputs, outputs, messages, prepared
        totals = {}
        for mode, values in records.items():
            batches = {r["batch_id"]: r for r in values}
            totals[mode] = {
                "clips": len(values), "valid_json": sum(r["parse_ok"] for r in values),
                "batch_inference_seconds": sum(r["batch_inference_seconds"] for r in batches.values()),
                "rss_sample_max_bytes": available_max(r["rss_sample_max_bytes"] for r in values),
            }
        pairs = []
        first = {r["clip_id"]: r for r in records["uniform3"]}
        for six in records["uniform6"]:
            three = first[six["clip_id"]]
            pairs.append({"clip_id": six["clip_id"],
                          "uniform3": three["summary_ko"], "uniform6": six["summary_ko"],
                          "category_changed": three["category"] != six["category"],
                          "human_verdict": "unreviewed"})
        atomic_json(folder / "summary.json", {"modes": totals, "pairs": pairs,
                    "note": "JSON validity and changed descriptions do not establish correctness."})
        config["status"] = "completed"
    except BaseException as error:
        config["status"] = "failed"
        config["error_type"] = type(error).__name__
        raise
    finally:
        config["total_wall_seconds"] = time.perf_counter() - run_started
        atomic_json(folder / "run.json", config)


if __name__ == "__main__":
    main()
