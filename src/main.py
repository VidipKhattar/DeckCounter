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

import cv2

import detector

PLAYBACK_DELAY_MS = 15
CAMERA_INDEX = 0
WINDOW_NAME = "Card Deck Checker"


def draw_detections_cv2(frame, detections: list[detector.Detection]) -> None:
    """Draw bounding boxes for all detections onto an OpenCV frame in place."""
    for d in detections:
        color = (0, 200, 0) if d.confident else (0, 0, 200)
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{d.card or d.raw_label} {d.confidence:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def capture_frames(cap, model, presence_confidence: float) -> tuple[list, tuple[int, int, int, int]]:
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
            box = detector.compute_guide_box(width, height)

        if not recording and detector.check_presence(model, frame, box, presence_confidence):
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
    model,
    frames: list,
    confidence_threshold: float,
    min_consensus_frames: int,
    playback_delay_ms: int,
    augment: bool = False,
) -> set:
    """Run detection over every buffered frame, showing live progress, and return the seen set."""
    total = len(frames)

    def on_frame(frame_index, frame_total, frame, detections):
        annotated = frame.copy()
        draw_detections_cv2(annotated, detections)
        pct = frame_index / frame_total * 100
        cv2.putText(
            annotated,
            f"Processing {frame_index}/{frame_total} ({pct:.0f}%)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, annotated)
        cv2.waitKey(playback_delay_ms)
        print(f"Processed {frame_index}/{frame_total} frames", end="\r")

    result = detector.run_consensus(
        model, frames, confidence_threshold, min_consensus_frames, on_frame=on_frame, augment=augment
    )
    if total:
        print()
    return result.seen


def print_summary(seen: set) -> None:
    missing = detector.FULL_DECK - seen
    seen_sorted = sorted(seen, key=lambda c: (detector.SUITS.index(c[-1]), detector.RANKS.index(c[:-1])))
    missing_sorted = sorted(
        missing, key=lambda c: (detector.SUITS.index(c[-1]), detector.RANKS.index(c[:-1]))
    )
    print()
    print(f"✅ Seen ({len(seen_sorted)}): {', '.join(seen_sorted)}")
    print(f"❌ Missing ({len(missing_sorted)}): {', '.join(missing_sorted)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time playing card deck checker")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index (default 0)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=detector.CONFIDENCE_THRESHOLD,
        help=f"Per-frame confidence threshold for a detection to count (default {detector.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=detector.MIN_CONSENSUS_FRAMES,
        help=f"Minimum number of confident frames needed for a card to count as seen (default {detector.MIN_CONSENSUS_FRAMES})",
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
        default=detector.PRESENCE_CONFIDENCE,
        help=f"Confidence needed in the guide box to auto-start recording (default {detector.PRESENCE_CONFIDENCE})",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation during processing (slower, better recall)",
    )
    args = parser.parse_args()

    print(f"Loading playing-card model ({detector.MODEL_REPO})...")
    model = detector.load_model()

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
    seen = process_frames(
        model, frames, args.confidence, args.min_consensus, args.playback_delay, augment=args.augment
    )

    cv2.destroyAllWindows()
    print_summary(seen)


if __name__ == "__main__":
    main()
