# Radiator Thermal Intelligence Engine (RTIE)

A production-grade AI system for radiator fault detection and efficiency scoring using thermal imaging. Built with **EfficientNet-B0**, **Physics-Based Feature Fusion**, and **Uncertainty Estimation**.

**Status**: ✅ Production Ready
**Test Accuracy**: 95.07%
**Inference Latency**: ~10ms (ONNX on CPU)

---

## 🏗️ System Architecture

The model fuses deep learning visual features with explicit physics-based thermal features (gradients, entropy, cold spots) to achieve robust fault detection.

```mermaid
graph TD
    subgraph Input
        IMG[Thermal Image (224x224)]
        PHYS[Physics Features (10-dim)]
    end

    subgraph "RTIE Model (EfficientNet-B0)"
        BB[EfficientNet Backbone] -->|1280-dim| FUSE[Feature Fusion Layer]
        PHYS --> FUSE
        FUSE -->|1290-dim| FC1[FC Layers + MC Dropout]
        FC1 --> FC2[FC Layers + MC Dropout]
        
        FC2 --> HEAD_CLS[Classification Head]
        FC2 --> HEAD_EFF[Efficiency Head]
        FC2 --> HEAD_CONF[Confidence Head]
    end

    subgraph "Safety & Business Logic"
        HEAD_CONF -->|Sigmoid| CAL[Temperature Scaling]
        HEAD_CLS -->|Softmax| PRED[Fault Prediction]
        HEAD_EFF -->|Sigmoid| EFF[Efficiency Score]
        
        PRED & CAL --> GATE{Safety Gate}
        GATE -->|Low Conf/High Uncertainty| REV[Manual Review]
        GATE -->|High Conf| APP[Auto-Approve]
        
        PRED --> BIZ[Business Metrics Calc]
        BIZ --> COST[Est. Cost Impact]
        BIZ --> CO2[CO2 Reduction]
    end

    Input --> BB
```

---

## 📊 Performance Results

The model was retrained for **15 epochs** with aggressive Gaussian noise augmentation to ensure field robustness.

### 1. Classification Accuracy
- **Test Set Accuracy**: **95.07%** (750 held-out images)
- **Blockage Detection**: **100% Recall** (Critical safety requirement met)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| **efficient** | 1.0000 | 1.0000 | 1.0000 |
| **imbalance** | 1.0000 | 0.9200 | 0.9583 |
| **blockage** | 0.9868 | 1.0000 | 0.9934 |
| **scaling** | 0.9252 | 0.9067 | 0.9158 |
| **air_trapped** | 0.8528 | 0.9267 | 0.8882 |

### 2. Calibration & Reliability
- **ECE (Expected Calibration Error)**: **0.1044** (Reduced from 0.31 via Temperature Scaling)
- **Reliability**: The model's confidence score accurately reflects its probability of correctness.

### 3. Robustness Verification
We extensively tested the model against synthetic perturbations to simulate real-world conditions.

| Perturbation | Robustness | Accuracy | Notes |
|---|---|---|---|
| **Gaussian Noise (15σ)** | **High** | **94.8%** | Extremely robust to grainy thermal sensors. |
| **Temp Shift (±5°C)** | High | >94.0% | Calibration remains stable under thermal drift. |
| **Blur (7x7)** | High | 94.0% | Feature extraction survives low-res focus issues. |
| **Rotation (±20°)** | High | 95.6% | Robust to installer camera angles. |

![Robustness Chart](report/robustness_chart.png)

### 4. Field Simulation
Simulated deployment scenarios passed verification:
- ✅ **Aspect Ratio**: Robust to squashed/stretched radiator shapes (>94%).
- ✅ **Camera Distance**: Robust to 0.8x-1.2x zoom/crop (>93%).
- ✅ **Sensor Drift**: Robust to ±3°C offset (>93%).

---

## 🚀 Installation & Usage

### 1. Install Dependencies
```bash
git clone https://github.com/Famulkalam/RTIE.git
cd RTIE
pip install -r requirements.txt
```

### 2. Generate Data & Train
```bash
# Generate 5,000 synthetic thermal images
python3 dataset_generator.py

# Train the model (will auto-save best_model.pth)
python3 train.py
```

### 3. Run API
Start the FastAPI server for real-time inference:
```bash
uvicorn app:app --port 8000
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@data/synthetic/blockage/0001.png"
```

**Example Response:**
```json
{
  "status": "auto_approved",
  "fault_label": "blockage",
  "confidence": 0.88,
  "estimated_annual_cost_gbp": 108.86,
  "co2_reduction_potential_kg": 90.59
}
```

---

## 📂 Project Structure
- `model.py`: PyTorch model (EfficientNet-B0 + Physics Fusion).
- `train.py`: Multi-task training loop.
- `evaluate.py`: Calibration (ECE) and metric calculation.
- `robustness_test.py`: Perturbation testing suite.
- `field_simulation.py`: Deployment scenario simulation.
- `app.py`: Production API with business logic.
- `report/`: Generated charts, heatmaps, and metrics.
