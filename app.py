import streamlit as st
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops
import torchvision.models as tvm
from torchvision import transforms
from ast import If

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Zero-Shot BOM Pattern Detector", layout="wide")

# ==========================================
# GLOBALS & CACHED RESOURCES
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 512
PATTERN_SIZE = 224

@st.cache_resource
def get_device_constants():
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    return mean, std

_MEAN, _STD = get_device_constants()

@st.cache_resource
def get_preprocess_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

preprocess = get_preprocess_transforms()

# ==========================================
# Define ImageEncoder(MobileNetV3)
# ==========================================

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        # Load pretrained MobileNetV3 Small for fast CPU inference
        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.trunk = backbone.features  # output: [B, 576, H/32, W/32]

        # Projection head: 576 -> embed_dim (used for the global descriptor only)
        self.head = nn.Sequential(
            nn.Linear(576, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )

    def spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, 576, H/32, W/32] — the raw convolutional feature map."""
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, EMBED_DIM] L2-normalised global descriptor."""
        feats  = self.trunk(x)                # [B, 576, H_f, W_f]
        pooled = feats.mean(dim=(2, 3))       # [B, 576]  global avg pool
        return F.normalize(self.head(pooled), p=2, dim=-1)



@st.cache_resource
def load_model():
    model = ImageEncoder(embed_dim=EMBED_DIM).to(DEVICE)
    model.eval()
    
    # Pre-compute the "white background" bias to remove false positives in empty space
    white_canvas = np.full((PATTERN_SIZE, PATTERN_SIZE, 3), 255, dtype=np.uint8)
    t = torch.from_numpy(white_canvas).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(DEVICE)
    t = (t - _MEAN) / _STD
    with torch.no_grad():
        white_emb = model(t).squeeze(0)  
    return model, white_emb

model, white_embedding = load_model()

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def crop_to_content(img: np.ndarray, pad: int = 4, threshold: int = 200) -> np.ndarray:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return img[y1:y2, x1:x2]

def pad_to_multiple(img: np.ndarray, multiple: int = 32, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    fill = (pad_value,) * 3 if img.ndim == 3 else pad_value
    return cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=fill)

def resize_preserve_ar(img: np.ndarray, target: int = PATTERN_SIZE, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if img.ndim == 3:
        canvas = np.full((target, target, img.shape[2]), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((target, target), pad_value, dtype=img.dtype)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    resized = resize_preserve_ar(img_rgb, target=PATTERN_SIZE)
    return preprocess(resized).unsqueeze(0)

def rotate_image(img: np.ndarray, angle: float, bg_color=(255, 255, 255)) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=bg_color)

def crops_to_batch(bom_bgr: np.ndarray, boxes: list, size: int = PATTERN_SIZE) -> torch.Tensor:
    frames = []
    valid_indices = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = bom_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = resize_preserve_ar(crop_rgb, target=size)
        t = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
        frames.append(t)
        valid_indices.append(i)
    if not frames:
        return None, []
    batch = torch.stack(frames, dim=0).to(DEVICE, non_blocking=True)
    batch = (batch - _MEAN) / _STD
    return batch, valid_indices

# ==========================================
# CORE INFERENCE & NMS
# ==========================================
def sliding_window_detections_gpu(
    bom_bgr: np.ndarray,
    dimension_groups: dict,
    query_bank: torch.Tensor,
    scales: list = [1.0],
    stride_factor: float = 0.5,
    sim_thresh: float = 0.70,
    batch_size: int = 128
):
    H_img, W_img = bom_bgr.shape[:2]
    all_boxes, all_scores, all_heatmaps = [], [], {}
    
    white_score_thresh = 0.90  
    
    with torch.no_grad():
        for scale in scales:
            for (pat_h, pat_w), angle_indices in dimension_groups.items():
                w_s, h_s = int(pat_w * scale), int(pat_h * scale)
                if w_s <= 0 or h_s <= 0 or w_s > W_img or h_s > H_img:
                    continue
                
                step_x, step_y = max(1, int(w_s * stride_factor)), max(1, int(h_s * stride_factor))
                xs = list(range(0, W_img - w_s + 1, step_x))
                ys = list(range(0, H_img - h_s + 1, step_y))
                if not xs or not ys: continue
                
                heatmap = np.zeros((len(ys), len(xs)), dtype=np.float32)
                windows = [(x, y, x + w_s, y + h_s) for y in ys for x in xs]
                
                for i in range(0, len(windows), batch_size):
                    batch_boxes = windows[i:i+batch_size]
                    batch_tensor, valid_idx = crops_to_batch(bom_bgr, batch_boxes)
                    if batch_tensor is None:
                        continue
                    
                    embs = model(batch_tensor)  
                    
                    # 1. Reject "empty white" windows
                    white_sims = (embs @ white_embedding).cpu().numpy()
                    is_white = white_sims > white_score_thresh
                    
                    # 2. Compute similarity with the relevant query patterns
                    group_queries = query_bank[angle_indices] 
                    sims = (embs @ group_queries.T).cpu().numpy() 
                    best_sims = sims.max(axis=1) 
                    
                    for j, v_idx in enumerate(valid_idx):
                        if is_white[j]: continue
                        
                        score = best_sims[j]
                        bx1, by1, bx2, by2 = batch_boxes[v_idx]
                        
                        r = (by1 // step_y)
                        c = (bx1 // step_x)
                        if score > heatmap[r, c]:
                            heatmap[r, c] = score
                        
                        if score >= sim_thresh:
                            all_boxes.append([bx1, by1, bx2, by2])
                            all_scores.append(score)
                            
                all_heatmaps[scale] = heatmap
    return all_boxes, all_scores, all_heatmaps

def soft_nms(boxes, scores, iou_threshold=0.3, sigma=0.5):
    if len(boxes) == 0:
        return []
    
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    keep = []
    indices = np.argsort(s)[::-1]
    
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        if indices.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[indices[1:]] - inter)
        
        weight = np.ones_like(iou)
        mask = iou > iou_threshold
        weight[mask] = np.exp(-(iou[mask] * iou[mask]) / sigma)
        
        s[indices[1:]] *= weight
        
        valid = s[indices[1:]] >= 0.01
        indices = indices[1:][valid]
        indices = indices[np.argsort(s[indices])[::-1]]
        
    return keep

# ==========================================
# STREAMLIT UI
# ==========================================

st.title("Zero-Shot BOM Pattern Detector")
st.markdown("Upload a technical drawing (BOM) and any pattern you want to locate within it.")

with st.sidebar:
    st.header("1. Upload Inputs")
    bom_file = st.file_uploader("Upload BOM Drawing (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
    pattern_file = st.file_uploader("Upload Pattern Image (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
    
    st.header("2. Hyperparameters")
    enable_rotation = st.checkbox("Enable Rotation Search", value=False, help="Uncheck for much faster CPU execution if pattern is fixed.")
    sim_thresh = st.slider("Similarity Threshold", min_value=0.5, max_value=0.99, value=0.80, step=0.01)
    iou_thresh = st.slider("IoU Threshold (Soft-NMS)", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    stride_factor = st.slider("Stride Factor", min_value=0.1, max_value=1.0, value=0.50, step=0.05)
    max_detections = st.number_input("Max Detections", min_value=1, max_value=200, value=40)
    
    run_btn = st.button("Run Detection", type="primary", use_container_width=True)

# Process logic
if bom_file and pattern_file and run_btn:
    with st.spinner("Processing Images & Running Inference (this may take a minute on CPU)..."):
        
        # Load images
        bom_bytes = np.frombuffer(bom_file.read(), np.uint8)
        bom_bgr = cv2.imdecode(bom_bytes, cv2.IMREAD_COLOR)
        
        pat_bytes = np.frombuffer(pattern_file.read(), np.uint8)
        pattern_bgr = cv2.imdecode(pat_bytes, cv2.IMREAD_COLOR)
        
        # Padding BOM
        bom_bgr = pad_to_multiple(bom_bgr, multiple=32, pad_value=255)
        
        # Crop Pattern
        pattern_crop = crop_to_content(pattern_bgr, pad=4)
        pattern_rgb = cv2.cvtColor(pattern_crop, cv2.COLOR_BGR2RGB)
        
        # Setup query bank
        ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315] if enable_rotation else [0]
        
        query_embs = []
        angle_dimensions = {}
        for angle in ROTATION_ANGLES:
            if angle == 0:
                rot = pattern_rgb
            else:
                rot = rotate_image(pattern_rgb, angle)
            
            # Use tighter bounding box for the rotated image
            rot_cropped = crop_to_content(rot, pad=2)
            angle_dimensions[angle] = rot_cropped.shape[:2]
            
            t = to_tensor(rot_cropped).to(DEVICE)
            with torch.no_grad():
                query_embs.append(model(t).squeeze(0))
                
        query_bank = torch.stack(query_embs, dim=0)
        
        # Group angles by dimension
        dimension_groups = {}
        for idx, angle in enumerate(ROTATION_ANGLES):
            dims = angle_dimensions[angle]
            if dims not in dimension_groups:
                dimension_groups[dims] = []
            dimension_groups[dims].append(idx)
        
        # Run detection
        scales = [0.75, 1.0, 1.25]
        all_boxes, all_scores, all_heatmaps = sliding_window_detections_gpu(
            bom_bgr,
            dimension_groups=dimension_groups,
            query_bank=query_bank,
            scales=scales,
            stride_factor=stride_factor,
            sim_thresh=sim_thresh,
            batch_size=128
        )
        
        # NMS
        keep_indices = soft_nms(all_boxes, all_scores, iou_threshold=iou_thresh, sigma=0.5)
        final_boxes = [all_boxes[i] for i in keep_indices][:max_detections]
        final_scores = [all_scores[i] for i in keep_indices][:max_detections]
        
    st.success(f"Detection complete! Found {len(final_boxes)} patterns.")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Combined Heatmap")
        if all_heatmaps:
            max_shape = max([h.shape for h in all_heatmaps.values()], key=lambda s: s[0]*s[1])
            combined_heatmap = np.zeros(max_shape, dtype=np.float32)
            for hmap in all_heatmaps.values():
                h_resized = cv2.resize(hmap, (max_shape[1], max_shape[0]), interpolation=cv2.INTER_LINEAR)
                combined_heatmap = np.maximum(combined_heatmap, h_resized)
            
            # Resize combined heatmap to original BOM dimensions for overlay
            heatmap_disp = cv2.resize(combined_heatmap, (bom_bgr.shape[1], bom_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            fig_hm, ax_hm = plt.subplots(figsize=(8, 8))
            ax_hm.imshow(cv2.cvtColor(bom_bgr, cv2.COLOR_BGR2GRAY), cmap='gray')
            im = ax_hm.imshow(heatmap_disp, cmap='jet', alpha=0.4, vmin=0, vmax=1.0)
            
            plt.colorbar(im, ax=ax_hm, shrink=0.5)
            ax_hm.axis('off')
            st.pyplot(fig_hm)
        else:
            st.warning("No heatmaps generated.")
            
    with col2:
        st.subheader("Detected Bounding Boxes")
        bom_rgb_disp = cv2.cvtColor(bom_bgr, cv2.COLOR_BGR2RGB)
        fig_bb, ax_bb = plt.subplots(figsize=(8, 8))
        ax_bb.imshow(bom_rgb_disp)
        
        for i, (bx1, by1, bx2, by2) in enumerate(final_boxes):
            rect = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2, edgecolor='red', facecolor='none')
            ax_bb.add_patch(rect)
            ax_bb.text(bx1, by1 - 5, f"{final_scores[i]:.2f}", color='red', fontsize=10, weight='bold')
            
        ax_bb.axis('off')
        st.pyplot(fig_bb)
        
    # Results Table
    st.subheader("Results Data")
    if len(final_boxes) > 0:
        df = pd.DataFrame({
            "Confidence Score": final_scores,
            "X1": [b[0] for b in final_boxes],
            "Y1": [b[1] for b in final_boxes],
            "X2": [b[2] for b in final_boxes],
            "Y2": [b[3] for b in final_boxes],
        })
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='detection_results.csv',
            mime='text/csv',
        )
    else:
        st.info("No bounding boxes met the criteria.")
        
elif run_btn:
    st.error("Please upload BOTH a BOM Drawing and a Pattern Image before running.")
