import sys
import os

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
import numpy as np
from PIL import Image
import config
from model import RTIEModel
from feature_engineering import extract_features
from business_metrics import compute_business_impact

# ─── Model Loading ───
model = None
device = torch.device("cpu") # Default to CPU for Spaces (unless GPU upgraded)

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = RTIEModel(pretrained=False)
        # Check for model path (handle both local repo and potential HF space stricture)
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.pth")
        
        if not os.path.exists(ckpt_path):
             # Fallback: maybe in current dir if deployed flattened
             ckpt_path = "best_model.pth"
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            # Handle possible key mismatch if saved with 'model_state_dict' wrapper vs direct
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Model initialized with random weights.")

load_model()

# ─── Inference Logic ───
def predict(image):
    if image is None:
        return "Please upload an image.", None, None

    # Preprocess
    img_pil = Image.fromarray(image).convert("L")
    img_rgb = Image.merge("RGB", [img_pil, img_pil, img_pil])
    
    # Resize/Transform
    img_tensor = np.array(img_rgb.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    img_tensor = img_tensor.transpose(2, 0, 1) # HWC -> CHW
    img_tensor = img_tensor / 255.0
    # Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).to(device)

    # Physics Features
    # Resize to simpler grid for feature extraction
    img_small = img_pil.resize((config.GRID_COLS, config.GRID_ROWS), Image.BICUBIC)
    matrix = np.array(img_small, dtype=np.float32)
    # Approx mapping 0-255 -> 30-80 C
    matrix = matrix / 255.0 * 50.0 + 30.0
    feats = extract_features(matrix)
    feats_tensor = torch.FloatTensor(feats).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        # Using predict_with_uncertainty from model.py
        result = model.predict_with_uncertainty(img_tensor, feats_tensor, T=config.MC_FORWARD_PASSES)
    
    # Parse Results
    class_idx = result["predicted_class"][0].item()
    fault_label = config.CLASS_NAMES[class_idx]
    confidence = result["confidence"][0].item()
    uncertainty = result["uncertainty_std"][0].item()
    efficiency = result["efficiency_score"][0].item()
    status = result["status"][0]
    
    probs = result["class_probs"][0].cpu().numpy()
    class_probs = {config.CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    
    # Business Metrics
    impact = compute_business_impact(fault_label, efficiency)
    
    # ─── Output Formatting ───
    
    # 1. Main Result Text
    status_emoji = "✅" if status == "auto_approved" else "⚠️"
    output_md = f"""
    ### Diagnosis: **{fault_label.upper()}**
    
    **Status**: {status_emoji} {status.replace('_', ' ').title()}
    
    - **Confidence**: {confidence:.1%}
    - **Uncertainty**: {uncertainty:.4f} (Low is better)
    - **Efficiency Score**: {efficiency:.1f}/100
    """
    
    # 2. Business Impact
    impact_md = f"""
    ### Business Impact
    
    - **Energy Loss**: {impact['energy_loss_pct']}%
    - **Est. Annual Cost**: £{impact['estimated_annual_cost_gbp']:.2f}
    - **CO₂ Reduction Potential**: {impact['co2_reduction_potential_kg']:.1f} kg
    """
    
    return output_md, class_probs, impact_md

# ─── Gradio Interface ───
with gr.Blocks(title="RTIE - Radiator Fault Detection") as demo:
    gr.Markdown("# 🌡️ RTIE: Radiator Thermal Intelligence Engine")
    gr.Markdown("Upload a thermal image of a radiator to detect faults, estimate efficiency, and calculate potential savings.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Thermal Image Input")
            btn = gr.Button("Analyze Radiator", variant="primary")
            
        with gr.Column():
            # Outputs
            result_markdown = gr.Markdown(label="Diagnosis")
            probs_plot = gr.Label(num_top_classes=5, label="Class Probabilities")
            impact_markdown = gr.Markdown(label="Business Impact")
            
    btn.click(fn=predict, inputs=input_image, outputs=[result_markdown, probs_plot, impact_markdown])
    
    gr.Markdown("---")
    gr.Markdown("**Note**: This demo runs the RTIE EfficientNet-B0 model with physics-based feature fusion. It includes uncertainty estimation via MC Dropout.")

if __name__ == "__main__":
    demo.launch()
