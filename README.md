# Zero-Shot BOM Pattern Detector

A zero-shot pattern detection system designed for engineering/BOM (Bill of Materials) technical drawings. This application locates target patterns (symbols, icons, components) on drawings without requiring model retraining or fine-tuning, leveraging a MobileNetV3-Small backbone for fast CPU inference.


## Features
- **Zero-Shot Detection**: Locate new patterns instantly without fine-tuning.
- **Fast CPU Inference**: Optimized using a MobileNetV3-Small feature extractor.
- **Interactive UI**: Powered by Streamlit with real-time parameter tuning.
- **Visual Heatmaps**: Combined cosine similarity heatmaps overlaid on the drawings.
- **Post-Processing (Soft-NMS)**: Eliminates redundant/overlapping bounding boxes.
- **Data Export**: Download detected coordinates and scores as a CSV.

## Deployment on Streamlit:
https://homeassessment-bhhgnb7ajfeuu6haz4r5lp.streamlit.app/

## How to run this project:

### 1. Prerequisites
Ensure you have Python installed (version **3.9 to 3.11** is recommended).

### 2. Clone the Repository
Clone this repository to your local machine:
```bash
git clone git@github.com:trungnguyendang/Home_Assessment.git
cd Home_Assessment
```


### 3. Install Dependencies
Install all the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Launch the project in local
Start the local development server:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser (typically at `http://localhost:8501`).

### 5. Run the Notebooks (Optional)
If you want to run the pipeline step-by-step or view details on the MobileNetV3 architecture:
1. Install Jupyter:
   ```bash
   pip install notebook
   ```
2. Open the notebook:
   ```bash
   jupyter notebook mobilenet_v3.ipynb
   ```

---

## Hyperparameter Guide

In the Streamlit sidebar, you can configure the following settings:

* **Enable Rotation Search**: Toggle this off if your target pattern always appears at the same rotation (0°). Leaving this disabled speeds up CPU processing significantly.
* **Similarity Threshold (Default: `0.80`)**: The minimum cosine similarity score required for a window to be considered a match.
* **IoU Threshold (Soft-NMS) (Default: `0.30`)**: Controls how overlapping candidate boxes are suppressed. If two boxes overlap by more than this threshold, the one with lower confidence will have its score decayed.
* **Stride Factor (Default: `0.50`)**: The step size of the sliding window relative to its size. A smaller stride (e.g., `0.25`) yields a more accurate and higher-resolution heatmap but takes longer to compute.
