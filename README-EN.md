# YOLO Area Intrusion Detection System
A desktop detection system based on PyQt5 and Ultralytics, supporting YOLOv5u / v8 / v9 / v10 / v11 / v12 and custom trained weights. It supports four input sources: images, videos, cameras, and RTSP/HLS video streams. Built-in functions include area intrusion monitoring, category filtering, real-time statistics, and Excel report export.

## Directory Structure
```
yolov8_detect/
├── main.py                 # Program entry, UI logic and event handling
├── ui/
│   ├── __init__.py
│   └── detect_ui.py        # Interface definition (decoupled from business logic)
├── utils/
│   ├── __init__.py
│   ├── detector.py         # QThread inference worker thread
│   ├── line_draw.py        # Polygon drawing and intrusion judgment
│   └── excel_report.py     # Excel report generation
├── weights/                # Store .pt model weights
├── ruqin/                  # JSON configuration for intrusion areas
│   └── ruqin_example.json
├── output/
│   ├── img_output/         # Image detection results
│   └── video_output/       # Video detection results
├── requirements.txt
└── README.md
```

## Installation
**Environment**: Python 3.8+, it is recommended to use an independent conda / venv environment.
```bash
pip install -r requirements.txt
```

If you encounter errors such as `Failed to initialize NumPy: _ARRAY_API not found`, it is caused by incompatibility between NumPy 2.x and the current PyTorch. Force downgrade NumPy:
```bash
pip install "numpy<2" --force-reinstall
```

Fix urllib3 version warnings from requests:
```bash
pip install "urllib3<2" "charset_normalizer<3.4"
```

GPU support requires matching CUDA and GPU version PyTorch. Refer to https://pytorch.org/get-started/locally/.

## Run
```bash
python main.py
```
After the program starts, the detected device will be displayed in the window title bar, such as `[GPU: NVIDIA GeForce RTX 4060]` or `[CPU]`.

## Interface Layout
Three-panel layout:
- **Left Function Panel**: Model selection & initialization → Detection source selection → Start/Pause/Stop → Result download → Area intrusion control
- **Center Detection Display**: Fixed 720×540 display area. Click with mouse to draw polygons, right-click to clear all points.
- **Right Information Panel**: Real-time statistics cards → Intrusion category checkbox list → Detailed detection log text

## Usage Process
### 1. Load Model
The drop-down menu includes built-in common weights: `yolov5nu/su/mu`, `yolov8n/s/m/l`, `yolov9t/s/c`, `yolov10n/s/m`, `yolo11n/s/m`, `yolo12n/s`, `best`. Local files in the `weights/` directory take priority; otherwise Ultralytics will download weights automatically. Click **Custom Model(.pt)** to load any third-party or self-trained weight files.

Select inference device: **Auto / GPU / CPU**. In auto mode, GPU is used when `torch.cuda.is_available()` is True. FP16 is enabled automatically when compute capability ≥ 7.0.

Click **Initialize Model**. Loading time and warmup time will be printed in the console. The first CUDA inference may take 5~30 seconds, which is normal. A pop-up will show weight file, category count and device information after successful loading.

### 2. Select Detection Source
Three buttons correspond to different sources. Clicking only previews the stream and does not start detection:
- **Select Image**: Open local jpg/png/jpeg files for static detection
- **Select Video**: Open local mp4/avi/mov files and display the first frame preview
- **Select Camera / Video Stream**: Read source from drop-down list (local camera index 0 or RTSP/HLS URL), preview one frame then release resources

### 3. Optional: Category Filtering
The right "Intrusion Categories" list fills all model categories automatically after model loading, all checked by default. Unchecked categories will not be boxed, counted or recorded in any mode. Three shortcut buttons: **Select All / Deselect All / Invert Selection**.

**Import YAML**: Supports standard YOLO `data.yaml` format, recognizes both dictionary and list `names` formats:
```yaml
# Dictionary format
names:
  0: person
  1: bicycle

# List format
names: ['helmet', 'no_helmet', 'worker']
```
Importing will overwrite the current category list. Ensure category IDs match model output indexes to avoid filtering offset errors.

### 4. Optional: Configure Intrusion Area
Check the **Area Intrusion** checkbox to enable intrusion detection mode. Set polygons in one of two ways:
- **Mouse Drawing**: Click **Draw Area** → Click at least 3 points on the screen to form a closed polygon. Points are saved to `ruqin/ruqin.json` automatically. Right-click to clear all points. Click the button again to **Stop Drawing** and exit drawing mode.
- **Upload JSON**: Click **Upload Area(JSON)** to import pre-configured coordinate files. The top-left corner of the display frame is the coordinate origin.

```json
{
  "x1": 100, "y1": 100,
  "x2": 500, "y2": 100,
  "x3": 500, "y3": 400,
  "x4": 100, "y4": 400
}
```

Uncheck Area Intrusion to run normal detection and skip this step.

### 5. Start Detection
Click the red **▶ Start Detection** button. The system runs accordingly:

|Scenario|Operation|
|-|-|
|Area Intrusion unchecked|Normal detection: draw bounding boxes for all selected categories in full frame|
|Area Intrusion checked + Polygon configured|Intrusion detection: only count and box targets entering the polygon|
|Area Intrusion checked + No polygon set|Prompt to set area or uncheck option, detection will not start|

Image detection runs one-time inference. Video and camera detection run independent background threads to keep the main UI responsive.

### 6. Video & Camera Control
- **Pause / Resume**: Pause frame reading, freeze display and stop statistics updating
- **Stop Detection**: Terminate worker thread, release camera, clear polygons and screen content

### 7. Export Results
Click green **Download Detection Results** → Select save folder. Two files will be exported:
- Media file: Image (jpg/png) or video (mp4)
- Excel report: `<filename>_report.xlsx`

## Real-time Statistics Panel
|Field|Color|Description|Available When|
|-|-|-|-|
|Current Intrusion Count (selected categories)|Red|Number of selected targets inside polygon in current frame|Area Intrusion Mode|
|Non-intrusion Detection Count (selected categories)|Green|Total selected targets detected in current frame|Normal Detection Mode|
|Total Processed Frames|Blue|Cumulative frames processed by worker thread|Always available|

The two counters are mutually exclusive: only one value updates while the other resets to zero.

## Excel Report
Contains 3 worksheets:
- **Summary**: Source type, source path, model file, total categories, filtered categories, inference device, FP16 status, area intrusion status, polygon vertex count, start & end time, processing duration, total frames, intrusion events, total detected targets, output path
- **DetectionLog**: Per-target records including frame ID, timestamp, category ID, category name, confidence, bounding box coordinates X1/Y1/X2/Y2, intrusion status
- **ClassSummary**: Occurrence times and in-area times grouped by category

The log supports up to 20,000 entries. No new records will be added beyond the limit to avoid memory overload during long video processing.

## System Architecture
- **Decoupled UI and Logic**: `ui/detect_ui.py` only defines `Ui_MainWindow` layout and styles. All business logic is encapsulated in the `UI_Logic_Window` class in `main.py`.
- **Multithreaded Inference**: `DetectionWorker(QThread)` in `utils/detector.py` handles video decoding, YOLO inference and video saving in an independent thread. Processed frames and statistics are sent to the main thread via `pyqtSignal`. Shared variables are protected by `QMutex`, so the UI will never freeze.
- **Stable FP16 Operation**: FP16 is enabled only when GPU compute capability ≥ 7.0. The system automatically falls back to FP32 if FP16 errors occur during warmup or inference.
- **Coordinate Scaling**: Frames are resized to 720×540 for display. Global scaling parameters map canvas coordinates back to original resolution, ensuring accurate polygon positioning across different image/video resolutions.

## FAQ
1. Stuck at "Warmup Starting" on launch
Initial CUDA warmup takes 5~30 seconds normally. Wait patiently. Check console stack traces if it lasts over 1 minute.

2. `Failed to initialize NumPy: _ARRAY_API not found`
Downgrade NumPy: `pip install "numpy<2" --force-reinstall`

3. Failed to open camera / RTSP stream
Check camera occupancy by other programs and accessibility of stream URLs. RTSP has lower latency but stricter network requirements than HLS.

4. Polygon position offset
Do not manually resize the display window. `MyLabel` is fixed at 720×540 via `setFixedSize(720, 540)`.

5. "No results available to export" displayed
Detection must be run at least once to generate exportable result files.

## Dependencies
```
ultralytics>=8.0.0     # YOLO inference framework
PyQt5>=5.15.0          # Graphical user interface
opencv-python>=4.5.0   # Video I/O and image processing
numpy<2                # Compatible with existing PyTorch versions
torch>=1.8.0           # Deep learning backend
openpyxl>=3.0.0        # Excel report generation
pyyaml>=5.4.0          # YAML category configuration import
```

## License & Acknowledgements
This project is for academic learning purposes. YOLO weights follow official Ultralytics license agreements. The GUI is built on Qt and PyQt5 framework.
