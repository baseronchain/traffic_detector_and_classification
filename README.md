````markdown
# YOLOv8 Vehicle Counter (Tkinter GUI)

Simple desktop app to **detect + track + count vehicles** from a video using **Ultralytics YOLOv8** and a **Tkinter GUI**.  
Vehicles are counted when their tracked centroid enters/crosses a horizontal **counting zone** (line + offset band).

## Features
- YOLOv8 **tracking** (`model.track(persist=True)`)
- **Counting zone** (line position adjustable in GUI)
- GUI controls: **Select Video**, **Stop**, **Reset**, **Line slider**, **Confidence slider**
- Auto device: **CUDA if available**, else CPU

## Files
- `main.py` — start GUI app
- `device_manager.py` — device + load `best.pt`
- `detector_engine.py` — tracking + counting logic
- `gui_interface.py` — Tkinter UI
- `testvideo.py` — quick OpenCV video open test

## Requirements
Python 3.8+  
```bash
pip install ultralytics torch opencv-python pillow
````

## Model

Put your weights file in the project root:

```
best.pt
```

**Important:** Class IDs counted by default (edit in `main.py` if your model is custom):

```python
{2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
```

## Run

```bash
python main.py
```

## Notes

* Frames are resized to **640×480**
* Counting happens once per track ID (prevents double counting)
* If GPU is not detected, check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

```
```
