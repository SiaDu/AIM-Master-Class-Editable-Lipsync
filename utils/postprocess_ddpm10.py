"""postprocess_ddpm10.py
Convert 10‑D diffusion output (and GT) back to full vertices and optionally add
emotion residual.

Example (inside notebook):

```python
import importlib, postprocess_ddpm10 as pp
importlib.reload(pp)

pred_v, gt_v = pp.run_postprocess(
    model_path   = "weight/best.pth",
    test_loader  = Dtest_loader,
    params_pkl   = "processed_lipsync_data.pkl",
    template_id  = "FaceTalk_170904_00128_TA",
    templates_pkl="/transfer/VOCAtrainingdata/templates.pkl",
    pose_scale   = 1.5,
    emotion      = "happy",
    emotion_scale= 1.0,
    sample_idx   = 0,
    guide_w      = 2.0,
)
print(pred_v.shape, gt_v.shape)  # (T,5023,3)
```
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_ddpm import DDPM, DenoiseTransformer

# ------------------------------------------------------------------ helper: unnorm 10‑dim

def make_pca10_params(params: dict):
    """Slice first 10 PCA components/stats from 284‑dim params."""
    return {
        "mean": params["pca_mean"][:10],
        "std" : params["pca_std"][:10],
        "comp": params["pca_components"][:10],   # (10,15069)
        "pca_internal_mean": params["pca_internal_mean"],
        "offset_mean": params["offset_mean"],
        "offset_std" : params["offset_std"],
    }


def unnormalize_pca10(norm_pca10: np.ndarray, p: dict) -> np.ndarray:
    """(T,10) -> (T,5023,3)"""
    scores  = norm_pca10 * p["std"] + p["mean"]
    flat    = scores @ p["comp"] + p["pca_internal_mean"]
    flat    = flat * p["offset_std"] + p["offset_mean"]
    return flat.reshape(-1, 5023, 3)

# ------------------------------------------------------------------ main api

def run_postprocess(
    *,
    model_path   : str,
    test_loader  : DataLoader,
    params_pkl   : str,
    template_id  : str,
    templates_pkl: Optional[str] = None,
    pose_scale   : float = 1.0,
    emotion      : Optional[Literal["happy", "sad", "angry"]] = None,
    emotion_scale: float = 1.0,
    sample_idx   : int   = 0,
    guide_w      : float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run 10‑dim DDPM once and rebuild vertices (pred & GT)."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) slice PCA params
    full_params = pickle.load(open(params_pkl, "rb"))
    p10 = make_pca10_params(full_params)

    # 2) build & load 10‑dim DDPM
    denoiser = DenoiseTransformer(pose_dim=10, audio_dim=464).to(device)
    ddpm     = DDPM(denoiser, betas=(1e-4, 0.02), n_T=1000, device=device)
    ddpm.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=False)
    ddpm.eval()

    # 3) fetch first batch and sample
    audio_cond, gt_norm = next(iter(test_loader))  # gt_norm shape (B,T,10)
    audio_cond, gt_norm = audio_cond.to(device), gt_norm.to(device)

    n_sample = audio_cond.size(0)
    with torch.no_grad():
        pred_norm = ddpm.sample(n_sample, audio_cond, (audio_cond.size(1), 10), guide_w)

    pred_np = pred_norm[sample_idx].cpu().numpy()
    gt_np   = gt_norm [sample_idx].cpu().numpy()

    pred_off = unnormalize_pca10(pred_np, p10)
    gt_off   = unnormalize_pca10(gt_np,   p10)

    # 4) add template
    if templates_pkl and templates_pkl.lower().endswith('.obj'):
        import trimesh
        mesh = trimesh.load_mesh(templates_pkl, process=False)
        base = mesh.vertices
    else:
        tmpl_path = templates_pkl or (Path(params_pkl).parent / "templates.pkl")
        templates = pickle.load(open(tmpl_path, "rb"), encoding="latin1")
        base = templates[template_id]

    pred_v = pred_off * pose_scale + base
    gt_v   = gt_off + base

    # 5) optional emotion residual
    if emotion:
        neutral = Path("Emotion/neutral.obj"); emo_obj = Path(f"Emotion/{emotion}.obj")
        if neutral.exists() and emo_obj.exists():
            import trimesh
            res = trimesh.load_mesh(emo_obj, process=False).vertices - \
                  trimesh.load_mesh(neutral, process=False).vertices
            pred_v += emotion_scale * res
        else:
            print("⚠️  emotion obj files not found, skipping residual")

    return pred_v, gt_v
