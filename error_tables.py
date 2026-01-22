import os
import glob
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import LogMelSeqDataset, collate_pad
from model import CRNN
from segtools import read_seg

ID2LAB = {0: "Vowel", 1: "Voiced consonant", 2: "Voiceless consonant", 3: "Sonorant"}


def normalize_label(raw_label: str) -> str:
    label = (raw_label or "").strip().lower()
    if " " in label:
        return ""
    while label and label[-1].isdigit():
        label = label[:-1]
    label = label.replace("'", "")
    return label


def find_seg_path(file_id: str) -> str | None:
    paths = glob.glob(f"corpus/**/{file_id}.seg_B2", recursive=True)
    return paths[0] if paths else None


def build_phoneme_intervals(labels: list[dict]) -> tuple[list[int], list[str]]:
    positions = [int(lab["position"]) for lab in labels]
    phones_raw = [lab.get("name", "") for lab in labels]
    phones = [normalize_label(p) for p in phones_raw]
    return positions, phones


def phoneme_at_sample(pos_samples: int, positions: list[int], phones: list[str]) -> str:
    if not positions or len(positions) < 2:
        return ""

    if pos_samples <= positions[0]:
        return phones[0]

    if pos_samples >= positions[-1]:
        return phones[-2] if len(phones) >= 2 else phones[-1]

    idx = int(np.searchsorted(np.array(positions), pos_samples, side="right")) - 1
    idx = max(0, min(idx, len(phones) - 2))
    return phones[idx]


def save_bar_png(
    x_labels, y_values, title, xlabel, ylabel, out_path, rotate=45, figsize=(10, 5)
):
    plt.figure(figsize=figsize)
    plt.bar(x_labels, y_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=rotate)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NPZ = "features_logmel_mfcc.npz"
    INPUT_DIM = 53
    CKPT = "phoneme_crnn_best.pth"
    frame_ms = 10.0
    seg_suffix = ".seg_B2"

    out_dir = os.path.join("reports", "phoneme_errors")
    os.makedirs(out_dir, exist_ok=True)

    ds = LogMelSeqDataset(NPZ)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pad)

    model = CRNN(input_dim=INPUT_DIM, hidden_dim=128, output_dim=4)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device)
    model.eval()

    phone_total = Counter()
    phone_err = Counter()
    phone_confusions = defaultdict(Counter)
    phone_trueclass = defaultdict(Counter)

    with torch.no_grad():
        for X_pad, y_pad, lengths, ids in dl:
            X_pad = X_pad.to(device)
            y_pad = y_pad.to(device)

            logits = model(X_pad)
            pred = logits.argmax(dim=-1)

            for b in range(X_pad.shape[0]):
                T = int(lengths[b].item())
                file_id = ids[b]

                true_seq = y_pad[b, :T].cpu().numpy().astype(int)
                pred_seq = pred[b, :T].cpu().numpy().astype(int)

                seg_paths = glob.glob(
                    f"corpus/**/{file_id}{seg_suffix}", recursive=True
                )
                if not seg_paths:
                    continue

                seg_path = seg_paths[0]
                params, labels = read_seg(seg_path)
                sr = int(params["SAMPLING_FREQ"])
                hop_samples = int(round(sr * frame_ms / 1000.0))
                if hop_samples <= 0:
                    continue

                positions, phones = build_phoneme_intervals(labels)
                if len(positions) < 2:
                    continue

                for t in range(T):
                    pos = t * hop_samples + hop_samples // 2
                    ph = phoneme_at_sample(pos, positions, phones)
                    if not ph:
                        continue

                    tru = int(true_seq[t])
                    pr = int(pred_seq[t])

                    phone_total[ph] += 1
                    phone_trueclass[ph][ID2LAB[tru]] += 1

                    if tru != pr:
                        phone_err[ph] += 1
                        phone_confusions[ph][(ID2LAB[tru], ID2LAB[pr])] += 1

    top_n = 15
    top_by_count = phone_err.most_common(top_n)

    if top_by_count:
        phonemes = [ph for ph, _ in top_by_count]
        counts = [phone_err[ph] for ph in phonemes]
        save_bar_png(
            phonemes,
            counts,
            title="Топ ошибочных фонем",
            xlabel="Фонема",
            ylabel="Ошибки",
            out_path=os.path.join(out_dir, "top_phonemes_by_error_count.png"),
            rotate=45,
            figsize=(10, 5),
        )

    min_frames = 300
    rates = []
    for ph in phone_total:
        tot = phone_total[ph]
        if tot >= min_frames:
            rates.append((phone_err[ph] / tot, ph, phone_err[ph], tot))

    rates.sort(reverse=True, key=lambda x: x[0])
    top_by_rate = rates[:15]

    if top_by_rate:
        phonemes = [ph for _, ph, _, _ in top_by_rate]
        values = [rate for rate, _, _, _ in top_by_rate]
        save_bar_png(
            phonemes,
            values,
            title=f"Топ фонем c самыми сильным ошибками",
            xlabel="Фонема",
            ylabel="Доля ошибки",
            out_path=os.path.join(out_dir, "top_phonemes_by_error_rate.png"),
            rotate=45,
            figsize=(10, 5),
        )

    worst_k = 3
    worst_phonemes = [ph for ph, _ in phone_err.most_common(worst_k)]

    for ph in worst_phonemes:
        conf = phone_confusions[ph]
        if not conf:
            continue

        top_conf = conf.most_common(6)
        labels_conf = [f"{a}→{b}" for (a, b), _ in top_conf]
        values_conf = [c for _, c in top_conf]

        save_bar_png(
            labels_conf,
            values_conf,
            title=f"Самые частые ошибки для фонемы'{ph}'",
            xlabel="С чем путает",
            ylabel="Количество",
            out_path=os.path.join(out_dir, f"confusions_{ph}.png"),
            rotate=0,
            figsize=(9, 4),
        )

    good = []
    for ph in phone_total:
        tot = phone_total[ph]
        if tot >= min_frames:
            good.append((phone_err[ph] / tot, ph, phone_err[ph], tot))

    good.sort(key=lambda x: x[0])
    top_best = good[:15]

    if top_best:
        phonemes = [ph for _, ph, _, _ in top_best]
        values = [rate for rate, _, _, _ in top_best]
        save_bar_png(
            phonemes,
            values,
            title=f"Фонемы с меньшим количеством ошибок",
            xlabel="Фонема",
            ylabel="Доля ошибок",
            out_path=os.path.join(out_dir, "best_phonemes_by_low_error_rate.png"),
            rotate=45,
            figsize=(10, 5),
        )


if __name__ == "__main__":
    main()
