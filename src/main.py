"""Real-time playing card deck checker.

Two-phase flow:
  1. Capture: a guide box is overlaid on the feed. A cheap presence check runs
     on just that cropped region each frame until a card shows up in it, at
     which point recording starts automatically. From then on frames are
     buffered with no inference so there's no lag while you flip. Press Q to
     stop capturing (you can flip through the deck more than once before
     stopping — more passes give blurry/occluded cards more chances to be
     seen clearly).
  2. Process: every buffered frame is cropped to the guide box (dropping
     background clutter and making the card occupy more of what the detector
     sees) and run through the detector. A card only counts as "seen" if it
     clears the confidence threshold in at least MIN_CONSENSUS_FRAMES separate
     frames, which filters out one-off misclassifications from motion blur
     while still catching cards that were only clear for a moment.

Prints a summary of seen vs. missing cards when processing finishes.
"""

import argparse
import sys
import time
from collections import defaultdict

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

MODEL_REPO = "sroot/yolo11s-playing-cards-detector"
MODEL_FILE = "best.pt"

CONFIDENCE_THRESHOLD = 0.35
PRESENCE_CONFIDENCE = 0.25
MIN_CONSENSUS_FRAMES = 1
BATCH_SIZE = 8
PLAYBACK_DELAY_MS = 15
CAMERA_INDEX = 0
WINDOW_NAME = "Card Deck Checker"
CARD_ASPECT_RATIO = 5 / 7  # width:height of a physical playing card
BOX_HEIGHT_RATIO = 0.75  # guide box height as a fraction of frame height
BOX_PADDING_RATIO = 0.2  # extra margin cropped around the box on each side

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]
FULL_DECK = {f"{rank}{suit}" for rank in RANKS for suit in SUITS}


def load_model() -> YOLO:
    print(f"Loading playing-card model ({MODEL_REPO})...")
    weights_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    return YOLO(weights_path)


def normalize_label(raw_label: str) -> str | None:
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
    return frame[cy1:cy2, cx1:cx2]


def draw_detections(frame, boxes, names, confidence_threshold: float) -> set[str]:
    """Draw bounding boxes for all detections; return the set of confidently-seen cards."""
    confident_cards = set()
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        raw_label = names[cls_id]
        card = normalize_label(raw_label)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confident = conf > confidence_threshold and card is not None
        color = (0, 200, 0) if confident else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{card or raw_label} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if confident:
            confident_cards.add(card)
    return confident_cards


def capture_frames(cap, model: YOLO, presence_confidence: float) -> tuple[list, tuple[int, int, int, int]]:
    """Buffer raw frames once a card appears in the guide box; no inference once recording starts."""
    frames: list = []
    recording = False
    box: tuple[int, int, int, int] | None = None

    print("Place a card in the box to start recording.")
    print("Once recording starts, flip through the deck (multiple passes are fine).")
    print("Press Q when done to start processing.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: failed to read frame from camera", file=sys.stderr)
            break

        if box is None:
            height, width = frame.shape[:2]
            box = compute_guide_box(width, height)

        if not recording:
            crop = crop_with_padding(frame, box)
            results = model.predict(crop, verbose=False)[0]
            if len(results.boxes) and float(results.boxes.conf.max()) > presence_confidence:
                recording = True
                print("Card detected — recording started.")

        display = frame.copy()
        x1, y1, x2, y2 = box
        if recording:
            frames.append(frame)
            box_color = (0, 200, 0)
            status = f"RECORDING - frames: {len(frames)} (press Q to stop)"
        else:
            box_color = (0, 200, 255)
            status = "Place a card in the box to start"

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return frames, box or (0, 0, 0, 0)


def process_frames(
    model: YOLO,
    frames: list,
    confidence_threshold: float,
    min_consensus_frames: int,
    playback_delay_ms: int,
) -> set[str]:
    """Run detection over every buffered frame and apply multi-frame consensus voting."""
    detection_counts: dict[str, int] = defaultdict(int)
    total = len(frames)

    for start in range(0, total, BATCH_SIZE):
        batch = frames[start : start + BATCH_SIZE]
        batch_results = model.predict(batch, verbose=False)

        for offset, result in enumerate(batch_results):
            annotated = batch[offset].copy()
            confident_cards = draw_detections(annotated, result.boxes, result.names, confidence_threshold)
            for card in confident_cards:
                detection_counts[card] += 1

            frame_index = start + offset + 1
            pct = frame_index / total * 100
            cv2.putText(
                annotated,
                f"Processing {frame_index}/{total} ({pct:.0f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(playback_delay_ms)
            print(f"Processed {frame_index}/{total} frames", end="\r")

    print()
    return {card for card, count in detection_counts.items() if count >= min_consensus_frames}


def print_summary(seen: set[str]) -> None:
    missing = FULL_DECK - seen
    seen_sorted = sorted(seen, key=lambda c: (SUITS.index(c[-1]), RANKS.index(c[:-1])))
    missing_sorted = sorted(missing, key=lambda c: (SUITS.index(c[-1]), RANKS.index(c[:-1])))
    print()
    print(f"✅ Seen ({len(seen_sorted)}): {', '.join(seen_sorted)}")
    print(f"❌ Missing ({len(missing_sorted)}): {', '.join(missing_sorted)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time playing card deck checker")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index (default 0)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Per-frame confidence threshold for a detection to count (default {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=MIN_CONSENSUS_FRAMES,
        help=f"Minimum number of confident frames needed for a card to count as seen (default {MIN_CONSENSUS_FRAMES})",
    )
    parser.add_argument(
        "--playback-delay",
        type=int,
        default=PLAYBACK_DELAY_MS,
        help=f"Milliseconds to pause on each frame during processing playback (default {PLAYBACK_DELAY_MS})",
    )
    parser.add_argument(
        "--presence-confidence",
        type=float,
        default=PRESENCE_CONFIDENCE,
        help=f"Confidence needed in the guide box to auto-start recording (default {PRESENCE_CONFIDENCE})",
    )
    args = parser.parse_args()

    model = load_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera index {args.camera}", file=sys.stderr)
        sys.exit(1)

    # Give the camera a moment to warm up; the first read(s) right after
    # opening can fail even though the device is valid.
    for _ in range(30):
        if cap.read()[0]:
            break
        time.sleep(0.1)

    try:
        frames, _box = capture_frames(cap, model, args.presence_confidence)
    finally:
        cap.release()

    if not frames:
        print("No frames captured.", file=sys.stderr)
        cv2.destroyAllWindows()
        sys.exit(1)

    print(f"Captured {len(frames)} frames. Processing...")
    seen = process_frames(model, frames, args.confidence, args.min_consensus, args.playback_delay)

    cv2.destroyAllWindows()
    print_summary(seen)


if __name__ == "__main__":
    main()
