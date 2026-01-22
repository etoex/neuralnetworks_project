import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import LogMelSeqDataset, collate_pad
from model import CRNN

LABELS = ["Vowels", "Voiced", "Unvoiced", "Sonorants"]


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LogMelSeqDataset("features_logmel_mfcc.npz")
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pad)

    model = CRNN(input_dim=53, hidden_dim=128, output_dim=4)
    model.load_state_dict(torch.load("phoneme_crnn_best.pth", map_location=device))
    model.to(device)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for X_pad, y_pad, lengths, ids in dl:
            X_pad = X_pad.to(device)
            y_pad = y_pad.to(device)

            logits = model(X_pad)
            pred = logits.argmax(dim=-1)

            mask = y_pad != -1

            all_true.extend(y_pad[mask].cpu().numpy())
            all_pred.extend(pred[mask].cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])

    print("Классификация:")
    print(classification_report(all_true, all_pred, target_names=LABELS))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="mako",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )

    plt.xlabel("Предсказано")
    plt.ylabel("На самом деле")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate()
