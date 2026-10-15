# Windows installation

1. Install Python 3.11 and confirm `python --version`.
2. From the repository root, create and activate a virtual environment.
3. Install `requirements.txt`; it installs one non-headless `opencv-contrib-python` wheel, NumPy, and pytest.
4. Run `python scripts/download_models.py`.
5. Run `python main.py --models verify` and `python main.py --help`.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
python main.py --help
```

Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
python main.py --help
```

Do not install `opencv-python` alongside `opencv-contrib-python`, and do not use headless wheels for webcam windows. If OpenCV import fails, install the Microsoft Visual C++ Redistributable. Windows N/KN may need the Media Feature Pack.

The ONNX models are downloaded from official OpenCV Zoo endpoints and remain local after download. See [model provenance](../models/README.md).
