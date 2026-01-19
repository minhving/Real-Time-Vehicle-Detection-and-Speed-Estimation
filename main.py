import argparse
import math
import os
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the script."""
    parser = argparse.ArgumentParser(
        description="Real-time vehicle detection, tracking, and speed estimation using YOLO."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO model weights file.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="video-tracking_EzI8SjOU.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_results.txt",
        help="Path to output results text file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width for the video stream.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height for the video stream.",
    )
    return parser.parse_args()


def setup_environment() -> None:
    """Configure environment variables required by dependencies."""
    # Fix OpenMP duplicate lib issue
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_model(model_path: str) -> YOLO:
    """Load the YOLO model from the given path."""
    return YOLO(model_path)


def open_video(video_path: str, width: int, height: int) -> cv2.VideoCapture:
    """Open a video file and configure its resolution."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def calculate_distance(x1, y1, x2, y2, x1p, y1p, x2p, y2p) -> float:
    """
    Compute Euclidean distance between the centers of two bounding boxes.

    Boxes are defined by their (x1, y1, x2, y2) coordinates.
    """
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x1p + x2p) / 2
    cy2 = (y1p + y2p) / 2
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


def track_and_estimate_speed(
    model: YOLO,
    cap: cv2.VideoCapture,
    output_path: str,
) -> None:
    """
    Run the main loop: detect, track, estimate speed, annotate, and log results.
    """
    prev_time = time.time()
    prev_boxes = []
    frame_index = 0

    with open(output_path, "w") as output_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # YOLO tracking
            results = model.track(frame, persist=True)
            current_time = time.time()
            delta_time = current_time - prev_time
            prev_time = current_time

            # tensor: [x1, y1, x2, y2, track_id, conf, class_id]
            current_boxes = results[0].boxes.data

            for box in current_boxes:
                x1, y1, x2, y2, track_id, conf, cls = box.tolist()
                speed_kmh = 0.0

                # Match with previous frame to calculate speed
                for prev_box in prev_boxes:
                    if track_id == prev_box[4]:  # match ID
                        dist_pixels = calculate_distance(
                            x1,
                            y1,
                            x2,
                            y2,
                            prev_box[0],
                            prev_box[1],
                            prev_box[2],
                            prev_box[3],
                        )
                        # pixel/sec to km/h: px/s * 3.6 * (meters_per_pixel, assumed 0.1)
                        speed_kmh = (dist_pixels / max(delta_time, 1e-6)) * 0.36
                        break

                # Draw on frame
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                label = f"ID:{int(track_id)} | {round(speed_kmh, 2)} km/h"
                cv2.putText(
                    frame,
                    label,
                    (x1i, y1i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                # Write to file
                output_file.write(
                    f"Frame {frame_index}, "
                    f"ID:{int(track_id)}, "
                    f"Class:{int(cls)}, "
                    f"Speed:{round(speed_kmh, 2)} km/h\n"
                )

            # Show frame
            frame_resized = cv2.resize(frame, (1000, 800))
            cv2.imshow("YOLO Tracking + Speed", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            prev_boxes = current_boxes.clone()  # clone to avoid in-place ops
            frame_index += 1


def main() -> None:
    args = parse_args()

    setup_environment()

    model = load_model(args.model)
    cap = open_video(args.video, args.width, args.height)

    try:
        track_and_estimate_speed(model, cap, args.output)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
