"""Compare timestamp coverage, without VLM inference or label-guided selection."""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.sampling import SAMPLER_VERSION, motion_probes, sample_frames

MODES = (("uniform3", "uniform", 3), ("uniform6", "uniform", 6), ("hybrid6", "hybrid", 6))
CATEGORIES = {"entry_exit", "object_handling", "vehicle_loading"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("run-id must be 1-64 letters, digits, underscores or hyphens")
    manifest_path = ROOT / "samples/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = ROOT / "experiments" / args.run_id
    # Exclusive directory creation prevents accidental replacement of prior experiments.
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": args.run_id,
        "sampler_version": SAMPLER_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sampler_sha256": hashlib.sha256((ROOT / "app/sampling.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_hz": 2,
        "modes": [name for name, _, _ in MODES],
        "status": "running",
        "vlm_inference_performed": False,
    }
    write_json(output / "run.json", config)
    totals = {name: {"clips": 0, "frames": 0, "intervals": 0, "with_sample": 0}
              for name, _, _ in MODES}
    try:
        for index, clip in enumerate(manifest["clips"]):
            path = ROOT / clip["file"]
            end = float(clip["duration"])
            started = time.perf_counter()
            probes = motion_probes(path, 0, end, probe_hz=2)
            probe_seconds = time.perf_counter() - started
            record = {
                "clip_id": clip["id"],
                "input_video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "probe_seconds": probe_seconds,
                "probes": probes,
                "modes": {},
            }
            # Only path, time bounds and motion probes enter the frame selector.
            for name, strategy, budget in MODES:
                started = time.perf_counter()
                frames = sample_frames(path, 0, end, budget, strategy, probes=probes)
                times = [t for t, _ in frames]
                record["modes"][name] = {
                    "requested_frames": budget,
                    "actual_frames": len(frames),
                    "frame_times": times,
                    "selection_decode_seconds": time.perf_counter() - started,
                }
                for _, image in frames:
                    image.close()
            # Consult partial annotations only AFTER all three selections are complete.
            for name, _, _ in MODES:
                result = record["modes"][name]
                coverage = []
                for reference in clip.get("ground_truth", []):
                    if reference["category"] not in CATEGORIES:
                        continue
                    coverage.append({
                        "activity": reference["activity"],
                        "start": reference["start"],
                        "end": reference["end"],
                        "with_sample": any(
                            reference["start"] <= t <= reference["end"]
                            for t in result["frame_times"]
                        ),
                    })
                result["reference_interval_coverage"] = coverage
                totals[name]["clips"] += 1
                totals[name]["frames"] += result["actual_frames"]
                totals[name]["intervals"] += len(coverage)
                totals[name]["with_sample"] += sum(r["with_sample"] for r in coverage)
            write_json(output / f"clip-{index:03d}.json", record)
            print(f"{index+1}/{len(manifest['clips'])}: {clip['id']}", flush=True)
        summary = {
            "run_id": args.run_id,
            "modes": totals,
            "note": (
                "Timestamp coverage of partial clip-level reference intervals, NOT recognition "
                "accuracy or recall. Overlapping clips can duplicate source events. Missing "
                "annotations do not prove absence. Timings are single sequential measurements "
                "affected by caching; hybrid requires probe_seconds in addition to selection."
            ),
        }
        write_json(output / "summary.json", summary)
        config["status"] = "completed"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        config["status"] = "failed"
        config["error"] = str(error)
        raise
    finally:
        write_json(output / "run.json", config)


if __name__ == "__main__":
    main()
