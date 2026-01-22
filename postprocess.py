import numpy as np

ID2LAB = {0: "Vowel", 1: "Voiced consonant", 2: "Unvoiced consonant", 3: "Sonorant"}


def frames_to_seg_labels(frame_ids: np.ndarray, hop_samples: int, level: str = "B2"):
    """
    Возвращает labels для write_seg: [{"position": <>, "level": "...", "name": "..."}]
    """
    T = int(len(frame_ids))
    if T == 0:
        return []

    labels = []

    cur = int(frame_ids[0])
    labels.append({"position": 0, "level": level, "name": ID2LAB[cur]})

    for t in range(1, T):
        v = int(frame_ids[t])
        if v != cur:
            labels.append(
                {"position": t * hop_samples, "level": level, "name": ID2LAB[v]}
            )
            cur = v

    # финальная граница
    labels.append({"position": T * hop_samples, "level": level, "name": ID2LAB[cur]})
    return labels
