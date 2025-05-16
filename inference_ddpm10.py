"""inference_ddpm10_refactored.py  —  Run DDPM-10 on audio features and postprocess

Refactored to load template and emotion residual once for long sequence inference.
"""
import pickle
from pathlib import Path
from typing import Optional, Literal, List

import numpy as np
import torch
from diffusion_ddpm import DDPM, DenoiseTransformer

# -----------------------------------------------------------------------------
# Helper: slice & unnormalize PCA-10 offsets (no base)
# -----------------------------------------------------------------------------
def make_pca10_params(params: dict) -> dict:
    """Slice first 10 PCA components and stats."""
    return {
        "mean": params["pca_mean"][:10],
        "std": params["pca_std"][:10],
        "comp": params["pca_components"][:10],
        "pca_internal_mean": params["pca_internal_mean"],
        "offset_mean": params["offset_mean"],
        "offset_std": params["offset_std"],
    }

def unnormalize_pca10(norm_pca10: np.ndarray, p: dict) -> np.ndarray:
    """Invert PCA10 normalization: (T,10) -> (T,5023,3)."""
    scores = norm_pca10 * p["std"] + p["mean"]                 # (T,10)
    flat = scores @ p["comp"] + p["pca_internal_mean"]       # (T,15069)
    flat = flat * p["offset_std"] + p["offset_mean"]         # (T,15069)
    return flat.reshape(-1, 5023, 3)                            # (T,5023,3)

# -----------------------------------------------------------------------------
# Helper: infer offsets from features (no base)
# -----------------------------------------------------------------------------
def infer_off_from_feats(
    feats: np.ndarray,
    ddpm_model_path: str,
    p10: dict,
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    guide_w: float,
    device: str = "cuda"
) -> np.ndarray:
    """Infer PCA10-normalized offsets for a single chunk (no base)."""
    T, W, D = feats.shape
    flat = feats.reshape(T, W * D)
    feats_norm = (flat - audio_mean[None]) / (audio_std[None] + 1e-9)
    cond = torch.from_numpy(feats_norm[None]).float().to(device)  # (1,T,WIN*D)

    # build and load model
    denoiser = DenoiseTransformer(pose_dim=10, audio_dim=W * D).to(device)
    ddpm = DDPM(denoiser, betas=(1e-4,0.02), n_T=1000, device=device)
    ddpm.load_state_dict(
        torch.load(ddpm_model_path, map_location="cpu", weights_only=True), strict=False
    )
    ddpm.eval()

    # sample
    with torch.no_grad():
        pred_norm = ddpm.sample(1, cond, (T,10), guide_w=guide_w)[0].cpu().numpy()  # (T,10)

    # unnormalize
    pred_off = unnormalize_pca10(pred_norm, p10)  # (T,5023,3)
    return pred_off

# -----------------------------------------------------------------------------
# Long-sequence inference with single template load
# -----------------------------------------------------------------------------
def infer_long_sequence(
    feats: np.ndarray,
    ddpm_model_path: str,
    params_pkl: str,
    templates_pkl: Optional[str],
    template_id: str,
    pose_scale: float = 1.0,
    emotion: Optional[Literal["happy","sad","angry"]] = None,
    emotion_scale: float = 1.0,
    guide_w: float = 2.0,
    device: str = "cuda",
    chunk_size: int = 30,
) -> np.ndarray:
    """
    Inference over long sequence by chunking. Loads template and emotion residual once.
    """
    # load params & pca10
    params = pickle.load(open(params_pkl,"rb"))
    p10 = make_pca10_params(params)
    audio_mean, audio_std = params["audio_mean"], params["audio_std"]

    # load template base
    if templates_pkl and templates_pkl.lower().endswith('.obj'):
        import trimesh
        mesh = trimesh.load_mesh(templates_pkl, process=False)
        base = mesh.vertices  # (5023,3)
    else:
        tmpl = templates_pkl or (Path(params_pkl).parent/"templates.pkl")
        templates = pickle.load(open(tmpl,"rb"), encoding='latin1')
        base = templates[template_id]

    # compute emotion delta once
    if emotion:
        import trimesh
        neutral_v = trimesh.load_mesh("Emotion/neutral.obj",process=False).vertices
        emo_v = trimesh.load_mesh(f"Emotion/{emotion}.obj",process=False).vertices
        emotion_delta = (emo_v - neutral_v) * emotion_scale
    else:
        emotion_delta = 0

    # chunk inference and stitch
    T = feats.shape[0]
    verts_chunks: List[np.ndarray] = []
    for start in range(0, T, chunk_size):
        end = min(start+chunk_size, T)
        chunk = feats[start:end]
        if chunk.shape[0] < chunk_size:
            pad = np.zeros((chunk_size-chunk.shape[0],*chunk.shape[1:]),dtype=chunk.dtype)
            chunk = np.vstack([chunk,pad])
        # infer offsets
        pred_off = infer_off_from_feats(
            feats=chunk,
            ddpm_model_path=ddpm_model_path,
            p10=p10,
            audio_mean=audio_mean,
            audio_std=audio_std,
            guide_w=guide_w,
            device=device,
        )  # (chunk_size,5023,3)
        # apply base, scale, emotion
        verts = base + pose_scale*pred_off + emotion_delta
        verts_chunks.append(verts[: end-start])

    return np.vstack(verts_chunks)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--params',required=True)
    parser.add_argument('--templates',default=None)
    parser.add_argument('--template-id',required=True)
    parser.add_argument('--out',default='pred_ddpm10.npy')
    parser.add_argument('--pose-scale',type=float,default=1.0)
    parser.add_argument('--emotion',choices=['happy','sad','angry'],default=None)
    parser.add_argument('--emotion-scale',type=float,default=1.0)
    parser.add_argument('--guide-w',type=float,default=2.0)
    parser.add_argument('--chunk-size',type=int,default=30)
    args=parser.parse_args()
    # load feats
    if args.feats.endswith('.wav'):
        from audio_preprocess import process
        feats = process(args.feats,args.feats.replace('.wav',''))
    else:
        feats = np.load(args.feats)
    # run
    verts = infer_long_sequence(
        feats=feats,
        ddpm_model_path=args.model,
        params_pkl=args.params,
        templates_pkl=args.templates,
        template_id=args.template_id,
        pose_scale=args.pose_scale,
        emotion=args.emotion,
        emotion_scale=args.emotion_scale,
        guide_w=args.guide_w,
        device='cuda',
        chunk_size=args.chunk_size,
    )
    np.save(args.out,verts)
    print('Saved',args.out,verts.shape)
