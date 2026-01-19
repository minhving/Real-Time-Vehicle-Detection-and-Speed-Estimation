## Real-Time Vehicle Detection and Speed Estimation

This project performs **real-time vehicle detection, tracking, and speed estimation** using a YOLO model from the `ultralytics` library and OpenCV.

It reads frames from a video file, tracks objects between consecutive frames, estimates their speed in km/h, overlays this information on the video, and logs the results to a text file.

---

### 1. Project Structure

- `main.py` — main entrypoint; runs detection, tracking, and speed estimation.
- `yolo11n.pt` — YOLO model weights file (you must provide this; not included in repo).
- `video-tracking_EzI8SjOU.mp4` — example input video.
- `annotated_results.txt` / `Results.txt` — text output files with per-frame, per-object results.
- `requirements.txt` — Python dependencies.

---

### 2. Prerequisites

- **Python** 3.9+ (recommended)
- A working C/C++ toolchain may be required by some dependencies (e.g., OpenCV) depending on your platform.

---

### 3. Installation

1. Create and activate a virtual environment (optional but recommended):

```bash
cd Real-Time-Vehicle-Detection-and-Speed-Estimation

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the YOLO weights file is present in this folder:

- Expected filename by default: `yolo11n.pt`

If you want to use a different weights file or path, you can override it at runtime with the `--model` flag.

---

### 4. Usage

Basic command (using the provided defaults in `main.py`):

```bash
python main.py
```

This will:

- Load the YOLO model from `yolo11n.pt`
- Open `video-tracking_EzI8SjOU.mp4`
- Display a window with bounding boxes and estimated speeds
- Write results to `annotated_results.txt`

#### 4.1. Command-Line Options

`main.py` supports several flags:

```bash
python main.py \
  --model yolo11n.pt \
  --video video-tracking_EzI8SjOU.mp4 \
  --output annotated_results.txt \
  --width 640 \
  --height 480
```

- `--model` — path to YOLO weights file.
- `--video` — path to the input video file.
- `--output` — where to save the text log of results.
- `--width` / `--height` — capture resolution for the video stream.

Press **`q`** in the video window to stop processing early.

---

### 5. Output Format

The script writes human-readable lines to the output text file, for example:

```text
Frame 0, ID:3, Class:2, Speed:45.67 km/h
Frame 0, ID:5, Class:2, Speed:38.12 km/h
Frame 1, ID:3, Class:2, Speed:47.02 km/h
...
```

- **Frame** — frame index in the processed video.
- **ID** — tracking ID assigned by the tracker.
- **Class** — numeric class ID from the YOLO model.
- **Speed** — estimated speed in km/h based on movement between frames.

---

### 6. Notes and Tips

- Speed estimation is approximate and depends on:
  - Frame rate of the video.
  - Perspective of the scene.
  - Assumptions inside the pixel-to-speed conversion (currently a simplified heuristic).
- For more accurate real-world speeds, you would need:
  - Calibration of camera and scene.
  - Known distances in the real world to convert pixels to meters.

---

### 7. Troubleshooting

- **`OSError: [WinError ...]` / OpenMP warnings**  
  The script sets `KMP_DUPLICATE_LIB_OK=TRUE` internally to help avoid some OpenMP duplicate library issues. If you still see errors, ensure that your Python environment and `ultralytics` installation are clean (try a fresh virtual environment).

- **Model file not found**  
  Check that `yolo11n.pt` exists in this directory or provide the correct path via:

  ```bash
  python main.py --model /path/to/your/model.pt
  ```

- **Video not opening**  
  Confirm the video path:

  ```bash
  python main.py --video /absolute/path/to/your/video.mp4
  ```

---

### 8. License

Add your preferred license information here (e.g., MIT, Apache 2.0, proprietary).

