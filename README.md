# Zero-Shot BOM Pattern Detector

A zero-shot pattern detection system designed for engineering/BOM (Bill of Materials) technical drawings. This application locates target patterns (symbols, icons, components) on drawings without requiring model retraining or fine-tuning, leveraging a MobileNetV3-Small backbone for fast CPU inference.


## Features
- **Zero-Shot Detection**: Locate new patterns instantly without fine-tuning.
- **Fast CPU Inference**: Optimized using a MobileNetV3-Small feature extractor.
- **Interactive UI**: Powered by Streamlit with real-time parameter tuning.
- **Visual Heatmaps**: Combined cosine similarity heatmaps overlaid on the drawings.
- **Post-Processing (Soft-NMS)**: Eliminates redundant/overlapping bounding boxes.
- **Data Export**: Download detected coordinates and scores as a CSV.

