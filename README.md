# Dental Cavity Detection

A Flask web application backed by a TensorFlow ResNet50 model for classifying dental x-ray images into four cavity severity levels. The repo also contains the training script and supporting assets/plots displayed in the UI.

## Prerequisites

- Python 3.11 (tested) with `pip`
- (Optional) A virtual environment to keep dependencies isolated
- Training images arranged under `Train/` and `test/` with class folders (`mild cavity`, `moderate cavity`, `no cavity`, `sevear cavity`)

## Setup

```powershell
# (optional) python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the app

```powershell
$env:FLASK_DEBUG="true"   # optional
python app.py
```

The UI is served at `http://127.0.0.1:5000/`. Register, log in, and upload an image to receive predictions plus preprocessing visualizations.

## Training the model

```powershell
python RESNET_50_TRAIN.py
```

Outputs:

- Updated `ResNet50_model.h5`
- `class_names.pkl` consumed by the web app
- Updated plots under `static/` (`Accu_plt.png`, `loss_plt.png`, `f1_graph.jpg`, `confusion_matrix.jpg`)

## Troubleshooting

- **Missing model/class names**: Train the model first so both `ResNet50_model.h5` and `class_names.pkl` exist in the project root.
- **Large TensorFlow download**: Installation can take several minutes; use a stable connection.
- **File upload failures**: Ensure the file input uses supported formats (PNG/JPG/JPEG) and that the `static/images` folder is writable.
