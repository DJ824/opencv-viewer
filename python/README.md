# OpenCV Operation Viewer - PyQt

This is a Python/PyQt6 version of the Qt/OpenCV desktop app. It supports media loading, operation browsing, operation chaining, per-step saved parameters, and per-stage output previews.

## Setup

Create a virtual environment:

```powershell
cd C:\path\to\opencv-viewer\python
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run:

```powershell
python opencv_viewer_pyqt.py
```

## Linux

```bash
cd /path/to/opencv-viewer/python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python opencv_viewer_pyqt.py
```

## Notes

- Use `opencv-python` for normal desktop use.
- Use `opencv-contrib-python` instead of `opencv-python` only if you later add operations from OpenCV contrib modules.
- On Windows, install Python 3.10 or newer from python.org or the Microsoft Store.
- If PowerShell blocks venv activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
