"""FastAPI web UI for the playing-card deck checker.

The browser owns the camera (getUserMedia) and streams JPEG frames up over a
WebSocket; this server never touches camera hardware directly, which is what
lets it run on a host that isn't the user's own laptop. Everything else is
unchanged from the local version: a state machine (WAITING -> RECORDING ->
PROCESSING -> DONE) drives detection via detector.py, with the same WebSocket
used to broadcast state updates back down to the client, plus a handful of
POST endpoints (stop / reset / settings).

Run with: uvicorn src.web.server:app
"""

import asyncio
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import detector  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


class State(str, Enum):
    WAITING = "waiting"
    RECORDING = "recording"
    PROCESSING = "processing"
    DONE = "done"


class Session:
    """Mutable state for the single capture session, guarded by a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = State.WAITING
        self.frames: list = []
        self.box: tuple | None = None
        self.stop_requested = False
        self.results: dict | None = None
        self.best_frame_jpegs: dict = {}
        self.session_id = uuid.uuid4().hex[:8]

    def reset(self):
        with self.lock:
            self.state = State.WAITING
            self.frames = []
            self.stop_requested = False
            self.results = None
            self.best_frame_jpegs = {}


class Settings:
    """Mutable detection thresholds, adjustable at runtime from the UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.confidence = detector.CONFIDENCE_THRESHOLD
        self.min_consensus = detector.MIN_CONSENSUS_FRAMES
        self.presence_confidence = detector.PRESENCE_CONFIDENCE

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "confidence": self.confidence,
                "min_consensus": self.min_consensus,
                "presence_confidence": self.presence_confidence,
            }

    def update(self, confidence: float, min_consensus: int, presence_confidence: float) -> None:
        with self.lock:
            self.confidence = min(max(confidence, 0.01), 0.99)
            self.min_consensus = max(int(min_consensus), 1)
            self.presence_confidence = min(max(presence_confidence, 0.01), 0.99)


session = Session()
settings = Settings()
model = None  # loaded during startup
event_loop: asyncio.AbstractEventLoop | None = None
active_websockets: set[WebSocket] = set()
websockets_lock = threading.Lock()


def broadcast(message: dict) -> None:
    """Thread-safe broadcast of a JSON message to all connected WebSocket clients."""
    if event_loop is None:
        return
    with websockets_lock:
        targets = list(active_websockets)
    for ws in targets:
        asyncio.run_coroutine_threadsafe(_safe_send(ws, message), event_loop)


async def _safe_send(ws: WebSocket, message: dict) -> None:
    try:
        await ws.send_json(message)
    except Exception:
        with websockets_lock:
            active_websockets.discard(ws)


def encode_jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


presence_check_busy = False


async def handle_incoming_frame(data: bytes) -> None:
    """Decode a JPEG frame sent by the browser and feed it into the state machine."""
    global presence_check_busy

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return

    height, width = frame.shape[:2]
    box = detector.compute_guide_box(width, height)

    with session.lock:
        state = session.state
        session.box = box

    if state == State.WAITING:
        # A full model inference runs per frame here. On a slow CPU that's much
        # slower than the browser's send rate, so skip frames that arrive while
        # a check is still running rather than letting an ever-growing backlog
        # of stale frames pile up.
        if presence_check_busy:
            return
        presence_check_busy = True
        try:
            presence_confidence = settings.as_dict()["presence_confidence"]
            loop = asyncio.get_running_loop()
            max_conf = await loop.run_in_executor(
                None, detector.presence_confidence_in_box, model, frame, box
            )
        finally:
            presence_check_busy = False

        print(
            f"[presence] frame={width}x{height} max_conf={max_conf:.3f} "
            f"threshold={presence_confidence:.3f} -> {'DETECTED' if max_conf > presence_confidence else 'none'}",
            flush=True,
        )
        if max_conf > presence_confidence:
            with session.lock:
                if session.state == State.WAITING:
                    session.state = State.RECORDING
            broadcast({"type": "recording", "frame_count": 0})

    elif state == State.RECORDING:
        with session.lock:
            session.frames.append(frame)
            frame_count = len(session.frames)
            should_stop = session.stop_requested
        broadcast({"type": "recording", "frame_count": frame_count})
        if should_stop:
            start_processing()


def start_processing() -> None:
    with session.lock:
        if session.state != State.RECORDING:
            return
        session.state = State.PROCESSING
        frames = list(session.frames)

    broadcast({"type": "processing", "current": 0, "total": len(frames)})
    threading.Thread(target=run_processing, args=(frames,), daemon=True).start()


def render_card_preview_jpeg(frame, detection: "detector.Detection") -> bytes:
    """Crop tightly around a detection's box (with a little padding) and label it."""
    x1, y1, x2, y2 = detection.box
    height, width = frame.shape[:2]
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 200, 0), 2)
    cv2.putText(
        crop,
        f"{detection.card} {detection.confidence:.2f}",
        (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 0),
        2,
    )
    return encode_jpeg(crop)


def run_processing(frames: list) -> None:
    start_time = time.monotonic()

    def on_frame(frame_index, total, frame, detections):
        broadcast({"type": "processing", "current": frame_index, "total": total})

    current_settings = settings.as_dict()
    result = detector.run_consensus(
        model, frames, current_settings["confidence"], current_settings["min_consensus"], on_frame=on_frame
    )
    elapsed = time.monotonic() - start_time

    seen = result.seen
    missing = detector.FULL_DECK - seen
    reliability = (
        sum(result.accepted_confidences) / len(result.accepted_confidences) * 100
        if result.accepted_confidences
        else 0.0
    )
    latency_ms = (elapsed / len(frames) * 1000) if frames else 0.0
    results = {
        "seen": sorted(seen, key=lambda c: (detector.SUITS.index(c[-1]), detector.RANKS.index(c[:-1]))),
        "missing": sorted(missing, key=lambda c: (detector.SUITS.index(c[-1]), detector.RANKS.index(c[:-1]))),
        "metrics": {
            "reliability": round(reliability, 1),
            "frame_count": len(frames),
            "latency_ms": round(latency_ms, 1),
        },
    }

    best_frame_jpegs = {
        card: render_card_preview_jpeg(frames[frame_idx], result.best_detection[card])
        for card, frame_idx in result.best_frame_index.items()
    }

    with session.lock:
        session.state = State.DONE
        session.results = results
        session.best_frame_jpegs = best_frame_jpegs

    broadcast({"type": "done", **results})


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, event_loop
    event_loop = asyncio.get_running_loop()
    print(f"Loading playing-card model ({detector.MODEL_REPO})...")
    model = detector.load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/session")
async def get_session():
    return {"session_id": session.session_id}


@app.post("/api/stop")
async def stop_recording():
    with session.lock:
        if session.state == State.RECORDING:
            session.stop_requested = True
    return {"ok": True}


@app.post("/api/reset")
async def reset_session():
    session.reset()
    broadcast({"type": "waiting"})
    return {"ok": True}


@app.get("/api/card/{card}/frame")
async def card_best_frame(card: str):
    with session.lock:
        jpeg = session.best_frame_jpegs.get(card.upper())
    if jpeg is None:
        raise HTTPException(status_code=404, detail=f"No captured frame for {card}")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/settings")
async def get_settings():
    return settings.as_dict()


@app.post("/api/settings")
async def update_settings(payload: dict):
    settings.update(
        confidence=float(payload.get("confidence", settings.confidence)),
        min_consensus=int(payload.get("min_consensus", settings.min_consensus)),
        presence_confidence=float(payload.get("presence_confidence", settings.presence_confidence)),
    )
    updated = settings.as_dict()
    broadcast({"type": "settings", **updated})
    return updated


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    with websockets_lock:
        active_websockets.add(ws)
    try:
        with session.lock:
            state = session.state
            frame_count = len(session.frames)
            results = session.results

        if state == State.DONE and results:
            await ws.send_json({"type": "done", **results})
        elif state == State.RECORDING:
            await ws.send_json({"type": "recording", "frame_count": frame_count})
        else:
            await ws.send_json({"type": "waiting"})

        while True:
            data = await ws.receive_bytes()
            await handle_incoming_frame(data)
    except WebSocketDisconnect:
        pass
    finally:
        with websockets_lock:
            active_websockets.discard(ws)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
