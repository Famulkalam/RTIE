# Model Card: Radiator Thermal Intelligence Engine (RTIE)

## Model Details
- **Developer**: FAMUL KALAM
- **Model Date**: February 2026
- **Model Version**: v1.0.0 (Epoch 15 Checkpoint)
- **Model Type**: Multi-Task EfficientNet-B0 + Physics Feature Fusion
- **License**: MIT

## Intended Use
- **Primary Use Case**: Automated fault detection in radiator systems using thermal images.
- **Intended Users**: Heating engineers, facility managers, and automated maintenance systems.
- **Input**: Thermal images (Grayscale or Ironbow palette) of radiators.
- **Output**: 
  - Fault Class (Efficient, Imbalance, Blockage, Scaling, Air Trapped)
  - Efficiency Score (0-100)
  - Confidence Score (0-1)

## Factors
- **Environment**: Performance may degrade in extreme ambient temperatures or poor lighting conditions (though thermal imaging mitigates lighting issues).
- **Sensor Quality**: Tested robustly against Gaussian noise (up to 15σ), ensuring compatibility with lower-resolution thermal cameras.
- **Deployment**: Optimized for edge deployment (ONNX) with <10ms latency on CPU.

## Metrics
The model is evaluated on a hold-out test set of 750 synthetic images.
- **Accuracy**: 95.07%
- **Recall (Blockage)**: 100.0% (Safety Critical)
- **Expected Calibration Error (ECE)**: 0.1044

## Training Data
- **Source**: Synthetic data generated via `dataset_generator.py`.
- **Size**: 5,000 images (1,000 per class).
- **Augmentations**: Gaussian noise, random blur, intensity shift, rotation.

## Evaluation Data
- **Source**: Independent split of synthetic data (15% split).
- **Size**: 750 images.
- **stress Tests**: Evaluated against rotation, blur, noise, and simulated camera drift.

## Ethical Considerations
- **Automation Bias**: Users should not rely solely on the AI for safety-critical decisions without human oversight, although the "Safety Gate" logic is designed to flag low-confidence predictions.
- **Data Privacy**: The model processes thermal images which generally preserve anonymity compared to optical cameras.

## Caveats & Recommendations
- **Sim2Real Gap**: The model is trained on physics-based synthetic data. Fine-tuning on real-world data is recommended for production deployment.
- **Calibration**: While improved, ECE is 0.10. For strict safety bounds, manual review of all "Medium" confidence predictions is advised.
