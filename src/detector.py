"""Shared playing-card detection core used by both the CLI and the web UI.

Pure detection/consensus logic lives here, with no display code (no OpenCV
windows, no print statements) so it can be reused by any front end.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

MODEL_REPO = "sroot/yolo11s-playing-cards-detector"
MODEL_FILE = "best.pt"

CONFIDENCE_THRESHOLD = 0.35
MAYBE_THRESHOLD = 0.18  # below CONFIDENCE_THRESHOLD: borderline reads land in a "maybe" bucket
PRESENCE_CONFIDENCE = 0.25
MIN_CONSENSUS_FRAMES = 1
BATCH_SIZE = 8
CARD_ASPECT_RATIO = 5 / 7  # width:height of a physical playing card
BOX_HEIGHT_RATIO = 0.75  # guide box height as a fraction of frame height
BOX_PADDING_RATIO = 0.2  # extra margin cropped around the box on each side

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]
FULL_DECK = {f"{rank}{suit}" for rank in RANKS for suit in SUITS}


def load_model() -> YOLO:
    weights_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    return YOLO(weights_path)


def normalize_label(raw_label: str) -> Optional[str]:
    """Normalize a raw model label to e.g. 'AS', '10D', 'KH'. Returns None if unrecognized."""
    label = raw_label.strip().upper().replace("_", "").replace("-", "")
    return label if label in FULL_DECK else None


def compute_guide_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """A centered card-shaped box telling the user where to hold the card."""
    box_height = int(frame_height * BOX_HEIGHT_RATIO)
    box_width = int(box_height * CARD_ASPECT_RATIO)
    x1 = (frame_width - box_width) // 2
    y1 = (frame_height - box_height) // 2
    return x1, y1, x1 + box_width, y1 + box_height


def crop_with_padding(frame, box: tuple[int, int, int, int]):
    """Crop frame to the guide box plus a padding margin, clamped to frame bounds."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * BOX_PADDING_RATIO)
    pad_y = int((y2 - y1) * BOX_PADDING_RATIO)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(width, x2 + pad_x)
    cy2 = min(height, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame
    return frame[cy1:cy2, cx1:cx2]


@dataclass
class Detection:
    card: Optional[str]
    raw_label: str
    confidence: float
    box: tuple[int, int, int, int]
    confident: bool


def extract_detections(boxes, names, confidence_threshold: float) -> list[Detection]:
    """Pure extraction of detections from a YOLO result's boxes; no drawing."""
    detections = []
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        raw_label = names[cls_id]
        card = normalize_label(raw_label)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confident = conf > confidence_threshold and card is not None
        detections.append(Detection(card, raw_label, conf, (x1, y1, x2, y2), confident))
    return detections


# Presence detection just needs to answer "is a card roughly here?" to trigger
# recording — it doesn't need to read the card. Running it at a small input size
# makes that first-card check several times faster, so recording kicks in almost
# immediately instead of lagging behind the moment you hold a card up. The full
# per-card reading later still runs at the model's native resolution.
PRESENCE_IMGSZ = 320


def presence_confidence_in_box(model: YOLO, frame, box: tuple[int, int, int, int]) -> float:
    """Highest detection confidence for anything card-shaped in the guide box (0.0 if nothing)."""
    crop = crop_with_padding(frame, box)
    results = model.predict(crop, verbose=False, imgsz=PRESENCE_IMGSZ, conf=0.05)[0]
    if not len(results.boxes):
        return 0.0
    return float(results.boxes.conf.max())


def check_presence(model: YOLO, frame, box: tuple[int, int, int, int], presence_confidence: float) -> bool:
    """Cheap check: is anything card-shaped sitting in the guide box right now?"""
    return presence_confidence_in_box(model, frame, box) > presence_confidence


@dataclass
class ConsensusResult:
    seen: set  # cleared the high (confidence) threshold in enough frames
    maybe: set  # cleared the lower (maybe) threshold but not the high one
    detection_counts: dict
    accepted_confidences: list  # confidences backing the `seen` set (for the accuracy metric)
    best_frame_index: dict  # card -> index into the `frames` list passed to run_consensus
    best_detection: dict  # card -> the Detection with the highest confidence seen for it


OnFrameCallback = Callable[[int, int, "object", list[Detection]], None]


def run_consensus(
    model: YOLO,
    frames: list,
    confidence_threshold: float,
    min_consensus_frames: int,
    maybe_threshold: Optional[float] = None,
    batch_size: int = BATCH_SIZE,
    on_frame: Optional[OnFrameCallback] = None,
    augment: bool = False,
) -> ConsensusResult:
    """Run detection over every frame and apply multi-frame consensus voting.

    A card counts as "seen" if it clears confidence_threshold in at least
    min_consensus_frames separate frames, which filters out one-off
    misclassifications from motion blur while still catching cards that were
    only clear for a moment.

    If `maybe_threshold` (< confidence_threshold) is given, cards that clear
    only that lower bar in enough frames land in a separate `maybe` bucket —
    surfacing borderline reads the user can double-check by eye rather than
    silently dropping them into "missing" or over-trusting them as "seen".

    `augment=True` enables test-time augmentation: each frame is run through
    the model at multiple scales/flips and the results are merged. It's 2-3x
    slower but recovers cards that a single-shot pass misses on soft or oddly
    angled frames — a recall boost from the same weights, no bigger model
    needed. Only worth it here because processing is offline (post-capture).

    `on_frame(frame_index, total, frame, detections)`, if given, is called
    after each frame is processed so callers can render progress/annotations.
    """
    low_threshold = maybe_threshold if maybe_threshold is not None else confidence_threshold
    confident_counts: dict = defaultdict(int)
    maybe_counts: dict = defaultdict(int)
    confidence_by_card: dict = defaultdict(list)
    best_frame_index: dict = {}
    best_detection: dict = {}
    total = len(frames)

    # Let the model surface low-confidence candidates and do the real filtering
    # ourselves — otherwise ultralytics' default internal conf (0.25) would
    # silently drop anything below it, capping how low the thresholds can go.
    predict_conf = min(low_threshold, 0.10)

    for start in range(0, total, batch_size):
        batch = frames[start : start + batch_size]
        batch_results = model.predict(batch, verbose=False, augment=augment, conf=predict_conf)

        for offset, result in enumerate(batch_results):
            frame_index = start + offset
            # Extract everything above the low bar; bucket per-card by its best
            # confidence in this frame.
            detections = extract_detections(result.boxes, result.names, low_threshold)
            cards_in_frame = {d.card for d in detections if d.card is not None and d.confidence > low_threshold}
            for card in cards_in_frame:
                best_for_card = max(
                    (d for d in detections if d.card == card), key=lambda d: d.confidence
                )
                maybe_counts[card] += 1
                if best_for_card.confidence > confidence_threshold:
                    confident_counts[card] += 1
                    confidence_by_card[card].append(best_for_card.confidence)
                if card not in best_detection or best_for_card.confidence > best_detection[card].confidence:
                    best_detection[card] = best_for_card
                    best_frame_index[card] = frame_index

            if on_frame is not None:
                on_frame(frame_index + 1, total, batch[offset], detections)

    seen = {card for card, count in confident_counts.items() if count >= min_consensus_frames}
    maybe = {
        card for card, count in maybe_counts.items() if count >= min_consensus_frames
    } - seen
    kept = seen | maybe
    accepted_confidences = [conf for card in seen for conf in confidence_by_card[card]]
    return ConsensusResult(
        seen=seen,
        maybe=maybe,
        detection_counts=dict(maybe_counts),
        accepted_confidences=accepted_confidences,
        best_frame_index={card: idx for card, idx in best_frame_index.items() if card in kept},
        best_detection={card: det for card, det in best_detection.items() if card in kept},
    )
