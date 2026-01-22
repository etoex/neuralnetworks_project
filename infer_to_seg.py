import os
import numpy as np
import torch

from dataset import LogMelSeqDataset
from model import CRNN
from segtools import write_seg
from postprocess import frames_to_seg_labels


def main():
    npz_path = "features_logmel_mfcc.npz"
    model_path = "phoneme_crnn_best.pth"
    out_dir = "pred_segs"
    frame_ms = 10.0
    level = "B2"

    os.makedirs(out_dir, exist_ok=True)

    lab = np.load("labels_frames.npz", allow_pickle=True)
    sr_map = {str(fid): int(sr) for fid, sr in zip(lab["ids"], lab["sr"])}

    ds = LogMelSeqDataset(npz_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(input_dim=53, hidden_dim=128, output_dim=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            x, _, file_id = ds[i]
            x = x.unsqueeze(0).to(device)

            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            sr = sr_map[file_id]
            hop = int(round(sr * frame_ms / 1000.0))

            labels = frames_to_seg_labels(pred, hop_samples=hop, level=level)

            params = {"SAMPLING_FREQ": sr, "BYTE_PER_SAMPLE": 2, "N_CHANNEL": 1}

            out_path = os.path.join(out_dir, f"{file_id}.seg_pred")
            write_seg(params, labels, out_path)

            if i % 20 == 0:
                print(f"[{i+1}/{len(ds)}] saved {out_path}")


if __name__ == "__main__":
    main()
