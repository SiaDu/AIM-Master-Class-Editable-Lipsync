from typing import Dict, List, Tuple
import numpy as np


def build_paired_sequences(
    data_verts: np.ndarray,
    audio_data: Dict[str, Dict[str, Dict]],
    seq_to_idx: Dict[str, Dict[str, Dict[int, int]]],
    templates: Dict[str, np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Return aligned audio, offset sequences and their labels.

    Parameters
    ----------
    data_verts : np.ndarray
        Full vertex array of shape (N_frames, 5023, 3).
    audio_data : dict
        Nested dict so that ``audio_data[subject_id][sentence_id]['audio']`` is an
        array shaped (T, 16, 29).
    seq_to_idx : dict
        Mapping ``seq_to_idx[subject_id][sentence_id][frame_number] -> global_frame_index``.
    templates : dict
        ``templates[subject_id]`` is a template mesh of shape (5023, 3).

    Returns
    -------
    paired_audio : list[np.ndarray]
        Each element is an audio tensor with shape (Li, 16, 29).
    paired_offset : list[np.ndarray]
        Same-length list where each element is a vertex offset tensor with shape (Li, 5023, 3).
    paired_names : list[str]
        Strings concatenating ``f"{subject_id}_{sentence_id}"``.
    """

    paired_audio: List[np.ndarray] = []
    paired_offset: List[np.ndarray] = []
    paired_names: List[str] = []

    for subject_id, sentences in seq_to_idx.items():
        if subject_id not in audio_data:
            continue
        for sentence_id, frame_map in sentences.items():
            if sentence_id not in audio_data[subject_id]:
                continue

            # --- frame alignment --------------------------------------------------
            sorted_indices = [frame_map[i] for i in sorted(frame_map)]

            expression_seq = data_verts[sorted_indices]  # (L, 5023, 3)
            audio_seq      = audio_data[subject_id][sentence_id]['audio']  # (L, 16, 29)

            # --- trim to common length -------------------------------------------
            L = min(len(expression_seq), len(audio_seq))
            expression_seq = expression_seq[:L]
            audio_seq      = audio_seq[:L]

            # --- to offset --------------------------------------------------------
            offset_seq = expression_seq - templates[subject_id][None, ...]

            # --- collect ----------------------------------------------------------
            paired_audio.append(audio_seq)
            paired_offset.append(offset_seq)
            paired_names.append(f"{subject_id}_{sentence_id}")

    return paired_audio, paired_offset, paired_names


if __name__ == "__main__":
    import pickle, pathlib, argparse

    parser = argparse.ArgumentParser(description="Build paired sequences and dump to pickle.")
    parser.add_argument("--data-verts", type=pathlib.Path, required=True)
    parser.add_argument("--audio", type=pathlib.Path, required=True)
    parser.add_argument("--seq-to-idx", type=pathlib.Path, required=True)
    parser.add_argument("--templates", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default="paired_data.pkl")
    args = parser.parse_args()

    with args.data_verts.open("rb") as f:
        data_verts = pickle.load(f)
    with args.audio.open("rb") as f:
        audio_data = pickle.load(f)
    with args.seq_to_idx.open("rb") as f:
        seq_to_idx = pickle.load(f)
    with args.templates.open("rb") as f:
        templates = pickle.load(f)

    pa, po, pn = build_paired_sequences(data_verts, audio_data, seq_to_idx, templates)
    pickle.dump((pa, po, pn), args.out.open("wb"))
    print(f"saved {len(pa)} paired sentences to {args.out}")
