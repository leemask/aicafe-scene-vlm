import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core import ROOT, Provider, extract_frames, video_info, windows, atomic_json

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LOCK = threading.Lock()
STORE_LOCK = threading.Lock()
app = FastAPI(title="AICAFE Scene Lab", docs_url=None, redoc_url=None)

@app.middleware("http")
async def same_origin(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "다른 사이트에서의 요청은 허용하지 않습니다."}, status_code=403)
    return await call_next(request)

def catalog():
    p = ROOT / "samples/manifest.json"
    sample = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"clips": []}
    entries = list(sample["clips"])
    for p in DATA.glob("*.json"):
        entries.append(json.loads(p.read_text(encoding="utf-8")))
    return entries

def resolve_clip(clip_id):
    for clip in catalog():
        if clip["id"] == clip_id:
            return clip, ROOT / clip["file"]
    raise HTTPException(404, "영상을 찾을 수 없습니다.")

@app.get("/api/status")
def status():
    return {"openai_configured": bool(os.getenv("OPENAI_API_KEY")), "busy": LOCK.locked(),
            "default_provider": "ollama", "note": "로컬 실행 도구. 모델 연결 전에는 AI 판정을 생성하지 않습니다."}

@app.get("/api/clips")
def clips():
    return {"clips": catalog()}

@app.get("/api/video/{clip_id}")
def video(clip_id: str):
    clip, path = resolve_clip(clip_id)
    return FileResponse(path, media_type="video/mp4")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, "MP4/MOV/AVI/MKV/WebM 영상을 선택하세요.")
    clip_id = "upload-" + uuid.uuid4().hex[:12]
    path = DATA / (clip_id + ext)
    size = 0
    try:
        with path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 250 * 1024 * 1024:
                    raise HTTPException(413, "최대 업로드 크기는 250MB입니다.")
                f.write(chunk)
        info = video_info(path)
        if info["duration"] > 600:
            raise HTTPException(400, "1차 버전은 10분 이하 영상만 지원합니다.")
        # Standard browser-playable format; no shell interpolation of filenames.
        from .core import ffmpeg
        import subprocess
        out = DATA / (clip_id + ".normalized.mp4")
        proc = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-an", "-vf", "scale='min(1280,iw)':-2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-movflags", "+faststart", str(out)], capture_output=True, timeout=300)
        if proc.returncode:
            out.unlink(missing_ok=True)
            raise ValueError("영상 변환 실패")
        path.unlink()
        record = {"id": clip_id, "title": (file.filename or "사용자 영상")[:160], "file": str(out.relative_to(ROOT)),
            **info, "source": "user_upload", "ground_truth": [], "annotation_status": "미검토"}
        atomic_json(DATA / (clip_id + ".json"), record)
        return record
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "영상 정보를 읽거나 변환하지 못했습니다.")

class AnalyzeRequest(BaseModel):
    clip_id: str
    provider: str = "ollama"
    model: str = Field(default="", max_length=150)
    window_seconds: int = Field(default=10, ge=4, le=30)
    max_windows: int = Field(default=12, ge=1, le=30)
    allow_cloud: bool = False

def result_path(job_id):
    if not job_id.replace("-", "").isalnum() or len(job_id) > 80:
        raise HTTPException(400, "잘못된 작업 ID")
    return RESULTS / (job_id + ".json")

def run_job(job, path, cfg):
    try:
        provider = Provider(cfg.provider, cfg.model)
        for i, (start, end) in enumerate(windows(job["duration"], cfg.window_seconds, cfg.max_windows)):
            frames = extract_frames(path, start, end)
            record = provider.analyze(frames) | {"id": str(i), "start": start, "end": end}
            folder = RESULTS / job["id"]
            folder.mkdir(exist_ok=True)
            image = folder / f"{i}.jpg"
            frames[len(frames)//2][1].save(image, quality=85)
            record["thumbnail"] = f"/results/{job['id']}/{i}.jpg"
            with STORE_LOCK:
                # Preserve review edits made while the job is running.
                existing = json.loads(result_path(job["id"]).read_text(encoding="utf-8"))
                for old in existing.get("segments", []):
                    for seg in job["segments"]:
                        if old["id"] == seg["id"]:
                            seg["review"], seg["review_note"] = old.get("review", "unreviewed"), old.get("review_note", "")
                job["segments"].append(record)
                job["progress"] = len(job["segments"])
                atomic_json(result_path(job["id"]), job)
        job["status"] = "completed"
    except Exception as exc:
        # Never include keys, request headers or raw remote response bodies in a UI error.
        job["status"] = "failed"
        job["error"] = "모델 분석 실패 (" + type(exc).__name__ + "). 모델 설치·실행 상태와 서버 환경변수를 확인하세요."
    finally:
        with STORE_LOCK:
            existing = json.loads(result_path(job["id"]).read_text(encoding="utf-8"))
            for i, segment in enumerate(job["segments"]):
                if i < len(existing.get("segments", [])):
                    segment["review"] = existing["segments"][i].get("review", "unreviewed")
                    segment["review_note"] = existing["segments"][i].get("review_note", "")
            atomic_json(result_path(job["id"]), job)
        LOCK.release()

@app.post("/api/analyze")
def analyze(cfg: AnalyzeRequest):
    if cfg.provider not in {"ollama", "openai", "smolvlm", "qwen"}:
        raise HTTPException(400, "지원하지 않는 모델 연결 방식")
    if cfg.provider == "openai" and not cfg.allow_cloud:
        raise HTTPException(400, "OpenAI로 분석 프레임을 전송하는 옵션을 선택하세요.")
    if cfg.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "서버에 OPENAI_API_KEY를 설정하세요.")
    clip, path = resolve_clip(cfg.clip_id)
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, "다른 분석이 진행 중입니다.")
    job = {"id": uuid.uuid4().hex, "clip_id": clip["id"], "status": "running", "duration": clip["duration"],
        "created_at": datetime.now(timezone.utc).isoformat(), "provider": cfg.provider, "model": cfg.model,
        "segments": [], "progress": 0, "total": len(windows(clip["duration"], cfg.window_seconds, cfg.max_windows)),
        "sampled_until": windows(clip["duration"], cfg.window_seconds, cfg.max_windows)[-1][1],
        "scope": "샘플 프레임 기반 구간 설명. 정확한 사건 시작·종료를 검출한 결과가 아닙니다."}
    atomic_json(result_path(job["id"]), job)
    threading.Thread(target=run_job, args=(job, path, cfg), daemon=True).start()
    return job

@app.get("/api/jobs")
def jobs():
    out = []
    for p in RESULTS.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        if "clip_id" in data:
            out.append(data)
    return {"jobs": sorted(out, key=lambda x: x.get("created_at", ""), reverse=True)}

@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    path = result_path(job_id)
    if not path.exists():
        raise HTTPException(404, "분석 결과 없음")
    return json.loads(path.read_text(encoding="utf-8"))

class Review(BaseModel):
    verdict: str
    note: str = Field(default="", max_length=2000)

@app.post("/api/jobs/{job_id}/segments/{segment_id}/review")
def review(job_id: str, segment_id: str, req: Review):
    if req.verdict not in {"correct", "incorrect", "missed", "uncertain", "unreviewed"}:
        raise HTTPException(400, "잘못된 검토 상태")
    with STORE_LOCK:
        data = job(job_id)
        for s in data["segments"]:
            if s["id"] == segment_id:
                s["review"], s["review_note"] = req.verdict, req.note
                atomic_json(result_path(job_id), data)
                return {"saved": True}
    raise HTTPException(404, "구간 없음")

@app.get("/api/export/{job_id}")
def export(job_id: str):
    job(job_id)
    return FileResponse(result_path(job_id), filename=f"aicafe-{job_id}.json", media_type="application/json")

@app.get("/")
def home():
    return RedirectResponse("/web/index.html")

app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")
app.mount("/results", StaticFiles(directory=RESULTS), name="results")

# A server restart cannot resume a thread from a previous process.
for p in RESULTS.glob("*.json"):
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("status") == "running":
        data.update(status="interrupted", error="서버가 재시작되어 분석이 중단되었습니다. 다시 실행하세요.")
        atomic_json(p, data)
