import os
import glob
import numpy as np

from segtools import read_seg


# нормализация меток - убираю цифры, знаки
def normalize_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if " " in label:
        return ""
    while label and label[-1].isdigit():
        label = label[:-1]
    label = label.replace("'", "")
    return label


#задаю классы
vowels = {"a", "o", "u", "i", "e", "y"}
sonorants = {"m", "n", "l", "r", "j"}
voiced_cons = {"b", "d", "g", "v", "z", "zh"}
voiceless_cons = {"p", "t", "k", "s", "f", "h", "sh", "sc", "ch", "c"}

phoneme_to_class = {}
for ph in vowels:
    phoneme_to_class[ph] = 0
for ph in voiced_cons:
    phoneme_to_class[ph] = 1
for ph in voiceless_cons:
    phoneme_to_class[ph] = 2
for ph in sonorants:
    phoneme_to_class[ph] = 3


def main():
    frame_ms = 10.0

    seg_files = sorted(glob.glob("corpus/**/*.seg_B2", recursive=True))

    y_list = []
    id_list = []
    sr_list = []

    for seg_path in seg_files:
        params, labels = read_seg(seg_path)
        sr = int(params["SAMPLING_FREQ"])

        hop = int(
            round(sr * frame_ms / 1000.0)
        )
        if hop <= 0:
            continue

        positions = []
        class_ids = []

        for i in range(len(labels) - 1):
            raw = labels[i]["name"]
            lab = normalize_label(raw)
            if not lab or lab not in phoneme_to_class:
                continue

            start = int(labels[i]["position"])
            end = int(labels[i + 1]["position"])
            if end <= start:
                continue

            positions.append((start, end))
            class_ids.append(phoneme_to_class[lab])

        if not positions:
            continue

        total_end = max(e for (_, e) in positions)
        T = int(np.ceil(total_end / hop))

        y = np.full(
            (T,), fill_value=-1, dtype=np.int64
        )

        for (start, end), cid in zip(positions, class_ids):
            fs = int(start // hop)
            fe = int(np.ceil(end / hop))
            fs = max(0, fs)
            fe = min(T, fe)
            if fe > fs:
                y[fs:fe] = cid

        if np.any(y == -1):
            last = -1
            for i in range(T):
                if y[i] != -1:
                    last = y[i]
                elif last != -1:
                    y[i] = last
            last = -1
            for i in range(T - 1, -1, -1):
                if y[i] != -1:
                    last = y[i]
                elif last != -1:
                    y[i] = last
            y[y == -1] = 0

        file_id = os.path.basename(seg_path).replace(".seg_B2", "")
        id_list.append(file_id)
        sr_list.append(sr)
        y_list.append(y)

    np.savez_compressed(
        "labels_frames.npz",
        ids=np.array(id_list),
        sr=np.array(sr_list),
        frame_ms=np.array([frame_ms]),
        y=np.array(y_list, dtype=object),
    )

    print("labels_frames.npz")
    print(len(id_list))


if __name__ == "__main__":
    main()
