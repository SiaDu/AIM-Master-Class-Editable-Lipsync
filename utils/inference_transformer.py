"""inference_transformer.py  —  Run LipsyncTransformer on audio features and postprocess
 
Supports both short and long sequences by chunking into fixed-length segments.
"""
 
import pickle
from pathlib import Path
from typing import Optional, Literal, Tuple, List
 
import numpy as np
import torch
 
from lipsync_transformer import LipsyncTransformer
 
# -----------------------------------------------------------------------------
# Core single-chunk inference
# -----------------------------------------------------------------------------
 
def infer_from_feats(
    feats: np.ndarray,
    transformer_model_path: str,
    params_pkl: str,
    templates_pkl: Optional[str] = None,
    template_id: str = "",
    pose_scale: float = 1.0,
    emotion: Optional[Literal["happy", "sad", "angry"]] = None,
    emotion_scale: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    Infer vertices for a single segment of features (T <= max_seq_len).
    Args:
        feats: (T, WIN, D) array
    Returns:
        verts: (T, 5023, 3)
    """

    # 0) Normalize
    params = pickle.load(open(params_pkl, "rb"))
    audio_mean, audio_std = params["audio_mean"], params["audio_std"]  # (D,)

    # 1. flatten to (1, T, WIN*D)
    T, W, D = feats.shape
    feats_flat = feats.reshape(T, W * D)   # (T, 464)
    feats_norm = (feats_flat - audio_mean[None, :]) / (audio_std[None, :] + 1e-9)
    x = torch.from_numpy(feats_norm[None, :, :]).float().to(device)
 
    # 2. load model
    model = LipsyncTransformer(max_seq_len=T)
    state = torch.load(transformer_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
 
    # 3. predict normalized PCA
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()[0]  # (T, k)
 
    # 4. un-normalize and back-project
    params = pickle.load(open(params_pkl, "rb"))
    from postprocess_LT import unnormalize_pca
    pred_off = unnormalize_pca(
        pred_norm,
        pca_mean=params["pca_mean"],
        pca_std=params["pca_std"],
        components=params["pca_components"],
        pca_internal_mean=params["pca_internal_mean"],
        offset_mean=params["offset_mean"],
        offset_std=params["offset_std"],
    )
 
    # 5. add template
    if templates_pkl and templates_pkl.lower().endswith('.obj'):
        import trimesh
        mesh = trimesh.load_mesh(templates_pkl, process=False)
        base = mesh.vertices
    else:
        tmpl_path = templates_pkl or (Path(params_pkl).parent / "templates.pkl")
        templates = pickle.load(open(tmpl_path, "rb"), encoding="latin1")
        base = templates[template_id]

    verts = pred_off * pose_scale + base
 
    # 6. optional emotion
    if emotion:
        neutral = Path("Emotion/neutral.obj")
        emo_obj = Path(f"Emotion/{emotion}.obj")
        if neutral.exists() and emo_obj.exists():
            import trimesh
            res = trimesh.load_mesh(emo_obj, process=False).vertices - \
                  trimesh.load_mesh(neutral, process=False).vertices
            verts += emotion_scale * res
    return verts
 
# -----------------------------------------------------------------------------
# Long-sequence inference via chunking
# -----------------------------------------------------------------------------
 
def infer_long_sequence(
    feats: np.ndarray,
    transformer_model_path: str,
    params_pkl: str,
    templates_pkl: Optional[str] = None,
    template_id: str = "",
    pose_scale: float = 1.0,
    emotion: Optional[Literal["happy", "sad", "angry"]] = None,
    emotion_scale: float = 1.0,
    device: str = "cuda",
    chunk_size: int = 30,
) -> np.ndarray:
    """
    Split a long feature array into segments of length <= chunk_size,
    infer each with infer_from_feats, and stitch results back to full (T,5023,3).
    """
    T = feats.shape[0]
    verts_list: List[np.ndarray] = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = feats[start:end]
        # pad last segment if shorter
        if chunk.shape[0] < chunk_size:
            pad = np.zeros((chunk_size - chunk.shape[0], *chunk.shape[1:]), dtype=chunk.dtype)
            chunk = np.vstack([chunk, pad])
        verts_chunk = infer_from_feats(
            feats=chunk,
            transformer_model_path=transformer_model_path,
            params_pkl=params_pkl,
            templates_pkl=templates_pkl,
            template_id=template_id,
            pose_scale=pose_scale,
            emotion=emotion,
            emotion_scale=emotion_scale,
            device=device,
        )
        verts_list.append(verts_chunk[: end - start])
    return np.vstack(verts_list)
 
# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Infer lipsync vertices from features')
    parser.add_argument('--feats', required=True, help='.npy file or raw wav via audio_preprocess')
    parser.add_argument('--model', required=True)
    parser.add_argument('--params', required=True)
    parser.add_argument('--templates', default=None)
    parser.add_argument('--template-id', required=True)
    parser.add_argument('--out', default='pred_vertices.npy')
    parser.add_argument('--pose-scale', type=float, default=1.0)
    parser.add_argument('--emotion', choices=['happy','sad','angry'], default=None)
    parser.add_argument('--emotion-scale', type=float, default=1.0)
    parser.add_argument('--chunk-size', type=int, default=30)
    args = parser.parse_args()
 
    # load features
    if args.feats.lower().endswith('.wav'):
        from audio_preprocess import process
        feats = process(args.feats, args.feats.replace('.wav',''))
    else:
        feats = np.load(args.feats)
 
    verts = infer_long_sequence(
        feats=feats,
        transformer_model_path=args.model,
        params_pkl=args.params,
        templates_pkl=args.templates,
        template_id=args.template_id,
        pose_scale=args.pose_scale,
        emotion=args.emotion,
        emotion_scale=args.emotion_scale,
        chunk_size=args.chunk_size,
    )
    np.save(args.out, verts)
    print('Saved vertices:', verts.shape)