import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import LogMelSeqDataset, collate_pad
from model import CRNN

import csv
import matplotlib.pyplot as plt


def train():
    ds = LogMelSeqDataset("features_logmel_mfcc.npz")

    # 80/20
    n_total = len(ds)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_dl = DataLoader(
        train_ds, batch_size=4, shuffle=True, collate_fn=collate_pad, num_workers=0
    )
    val_dl = DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=collate_pad, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = CRNN(input_dim=53, hidden_dim=128, output_dim=4).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    patience = 7
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = "phoneme_crnn_best.pth"

    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for X_pad, y_pad, lengths, ids in train_dl:
            X_pad = X_pad.to(device)
            y_pad = y_pad.to(device)

            optimizer.zero_grad()

            logits = model(X_pad)

            loss = criterion(logits.reshape(-1, 4), y_pad.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_dl))

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_pad, y_pad, lengths, ids in val_dl:
                X_pad = X_pad.to(device)
                y_pad = y_pad.to(device)

                logits = model(X_pad)

                loss = criterion(logits.reshape(-1, 4), y_pad.reshape(-1))
                val_loss_sum += loss.item()

                pred = logits.argmax(dim=-1)
                mask = y_pad != -1

                correct += (pred[mask] == y_pad[mask]).sum().item()
                total += mask.sum().item()

        val_loss = val_loss_sum / max(1, len(val_dl))
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Эпоха {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f}",
            flush=True,
        )

        history.append((epoch, train_loss, val_loss, val_acc))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        with open("training_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
            w.writerows(history)

        epochs = [h[0] for h in history]
        train_losses = [h[1] for h in history]
        val_losses = [h[2] for h in history]
        val_accs = [h[3] for h in history]

        plt.figure()
        plt.plot(epochs, train_losses, label="train_loss")
        plt.plot(epochs, val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Train/Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curves.png")
        plt.close()

        plt.figure()
        plt.plot(epochs, val_accs, label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("val_acc_curve.png")
        plt.close()

        print("training_history.csv, loss_curves.png, val_acc_curve.png", flush=True)


if __name__ == "__main__":
    train()
