import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lipsync_transformer import LipsyncTransformer

# ------------------------------------------------------------------ helpers

def unnormalize_pca(
    norm_pca: np.ndarray,
    *,
    pca_mean: np.ndarray,
    pca_std: np.ndarray,
    components: np.ndarray,
    pca_internal_mean: np.ndarray,
    offset_mean: np.ndarray,
    offset_std : np.ndarray,
) -> np.ndarray:
    """Invert the two normalisation steps + PCA back‑projection."""
    pca_scores   = norm_pca * pca_std + pca_mean            # (T, k)
    flat_offset  = pca_scores @ components + pca_internal_mean  # (T, 15069)
    flat_offset  = flat_offset * offset_std + offset_mean
    return flat_offset.reshape(-1, 5023, 3)

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred_vertices, gt_vertices) for **one** sequence.

    `sample_idx` chooses which item in the *first* batch of `test_loader` is used.
    """
    # 1. load model
    model = LipsyncTransformer(max_seq_len=30)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # 2. run one batch
    audio, gt_norm = next(iter(test_loader))
    audio, gt_norm = audio.cpu(), gt_norm.cpu()
    with torch.no_grad():
        pred_norm = model(audio).cpu()

    pred_np = pred_norm[sample_idx].numpy()  # (T, k)
    gt_np   = gt_norm[sample_idx].numpy()

    # 3. load preprocess params
    params = pickle.load(open(params_pkl, "rb"))

    pred_off = unnormalize_pca(
        pred_np,
        pca_mean           = params["pca_mean"],
        pca_std            = params["pca_std"],
        components         = params["pca_components"],
        pca_internal_mean  = params["pca_internal_mean"],
        offset_mean        = params["offset_mean"],
        offset_std         = params["offset_std"],
    )
    gt_off = unnormalize_pca(
        gt_np,
        pca_mean           = params["pca_mean"],
        pca_std            = params["pca_std"],
        components         = params["pca_components"],
        pca_internal_mean  = params["pca_internal_mean"],
        offset_mean        = params["offset_mean"],
        offset_std         = params["offset_std"],
    )

    # 4. template
    if templates_pkl and templates_pkl.lower().endswith('.obj'):
        import trimesh
        mesh = trimesh.load_mesh(templates_pkl, process=False)
        base = mesh.vertices
    else:
        tmpl_path = templates_pkl or (Path(params_pkl).parent / "templates.pkl")
        templates = pickle.load(open(tmpl_path, "rb"), encoding="latin1")
        base = templates[template_id]
        
    pred_verts = pred_off * pose_scale + base
    gt_verts   = gt_off + base

    # 5. optional emotion residual
    if emotion:
        neutral = Path("Emotion/neutral.obj")
        emo_obj = Path(f"Emotion/{emotion}.obj")
        if neutral.exists() and emo_obj.exists():
            import trimesh
            res = trimesh.load_mesh(emo_obj, process=False).vertices - \
                  trimesh.load_mesh(neutral, process=False).vertices
            pred_verts += emotion_scale * res
        else:
            print("⚠️  emotion obj files not found, skipping residual")

    return pred_verts, gt_verts

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Run post‑process and save npy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--templates", default=None)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--out", default="pred_vertices.npy")
    parser.add_argument("--pose-scale", type=float, default=1.0)
    parser.add_argument("--emotion", choices=["happy","sad","angry"], default=None)
    parser.add_argument("--emotion-scale", type=float, default=1.0)
    args = parser.parse_args()
    # dummy loader for cli demo (expects pickled (audio,expr) tuple)
    audio, expr = pickle.load(open("dummy_batch.pkl", "rb"))
    loader = DataLoader(list(zip(audio, expr)), batch_size=len(audio))
    pred, gt = run_postprocess(model_path=args.model, test_loader=loader,
                               params_pkl=args.params, templates_pkl=args.templates,
                               template_id=args.template_id, pose_scale=args.pose_scale,
                               emotion=args.emotion, emotion_scale=args.emotion_scale)
    np.save(args.out, pred)
    print("saved", args.out)