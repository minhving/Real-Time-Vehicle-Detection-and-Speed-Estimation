import os
import math
import time
import cv2
from ultralytics import YOLO

# Fix OpenMP duplicate lib issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO model (make sure yolo11n.pt exists in your folder)
model = YOLO("yolo11n.pt")

# Open the video
video_path = "video_tracking.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Time tracking for speed
prev_time = time.time()

# Distance conversion
meters_per_pixel = 0.01  # adjust based on known object size/distance

# Previous frame data for tracking speed
prev_boxes = []

# Frame index
i = 0

# Function to compute distance between box centers
def calculate_distance(x1, y1, x2, y2, x1p, y1p, x2p, y2p):
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x1p + x2p) / 2
    cy2 = (y1p + y2p) / 2
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

# Output file
output_file = open("annotated_results.txt", "w")

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 tracking
    results = model.track(frame, persist=True)
    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time

    current_boxes = results[0].boxes.data  # tensor: [x1, y1, x2, y2, track_id, conf, class_id]
    object_info = []

    for box in current_boxes:
        x1, y1, x2, y2, track_id, conf, cls = box.tolist()
        speed_kmh = 0

        # Match with previous frame to calculate speed
        for prev_box in prev_boxes:
            if track_id == prev_box[4]:  # match ID
                dist_pixels = calculate_distance(
                    x1, y1, x2, y2, prev_box[0], prev_box[1], prev_box[2], prev_box[3]
                )
                speed_kmh = (dist_pixels / delta_time) * 0.36
                break

        # Add info to list
        object_info.append({
            "id": int(track_id),
            "class_id": int(cls),
            "speed_kmh": round(speed_kmh, 2),
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 2)
        })

        # Draw on frame
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label = f"ID:{int(track_id)} | {round(speed_kmh, 2)} km/h"
        cv2.putText(frame, label, (x1i, y1i - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Write to file
        output_file.write(f"Frame {i}, ID:{int(track_id)}, Class:{int(cls)}, Speed:{round(speed_kmh,2)} km/h\n")
    frame_resized = cv2.resize(frame, (1000, 800))
    # Show frame
    cv2.imshow("YOLO Tracking + Speed", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_boxes = current_boxes.clone()  # clone to avoid inplace ops
    i += 1

# Cleanup
output_file.close()
cap.release()
cv2.destroyAllWindows()
