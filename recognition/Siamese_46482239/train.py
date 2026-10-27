import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from modules import SiameseNetwork, CNNEmbed, ContrastiveLoss
from dataset import get_isic_pairloader

# ----------------------------- config -----------------------------
EPOCHS = 28
DIM, IN_CHANNELS, EMBED_DIMS = 16, 3, 256
DROPOUT, NORMALISE, N_STAGES = 0.23, True, 5
LEARNING_RATE = 0.0003
MARGIN = 1.0
CKPT_PATH = "siamese_best.pt"
SEED = 0


def pair_labels(labels1: torch.Tensor, labels2: torch.Tensor) -> torch.Tensor:
    """0 = same class, 1 = different class (matches ContrastiveLoss)."""
    return (labels1 != labels2).float()


def run_epoch(model, loader, criterion, device, optimiser=None):
    """Trains if an optimiser is given, otherwise evaluates. Returns loss, pair accuracy, pair AUROC."""
    training = optimiser is not None
    model.train(training)
    losses, dists, targets = [], [], []
    with torch.set_grad_enabled(training):
        for (images1, labels1), (images2, labels2) in loader:
            images1, images2 = images1.to(device), images2.to(device)
            pairs = pair_labels(labels1, labels2).to(device)

            emb1, emb2 = model(images1, images2)
            loss = criterion(emb1, emb2, pairs)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            losses.append(loss.item())
            dists.append(F.pairwise_distance(emb1, emb2).detach().cpu())
            targets.append(pairs.cpu())
    d, t = torch.cat(dists), torch.cat(targets)
    pair_acc = ((d > MARGIN / 2).float() == t).float().mean().item()
    try:
        pair_auroc = roc_auc_score(t.numpy(), d.numpy())   # distance as score for "different"
    except ValueError:
        pair_auroc = float("nan")
    return float(np.mean(losses)), pair_acc, pair_auroc


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader = get_isic_pairloader("train")
    validate_loader = get_isic_pairloader("validate")

    model = SiameseNetwork(CNNEmbed(DIM, IN_CHANNELS, EMBED_DIMS, DROPOUT, NORMALISE, N_STAGES)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ContrastiveLoss(margin=MARGIN)

    train_losses, val_losses, best_auroc = [], [], -1.0
    print("Training model...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device, optimiser)
        va_loss, va_acc, va_auroc = run_epoch(model, validate_loader, criterion, device)
        train_losses.append(tr_loss); val_losses.append(va_loss)
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} pair-AUROC {va_auroc:.3f}")

        if va_auroc > best_auroc:
            best_auroc = va_auroc
            torch.save({"model": model.state_dict(),
                        "config": dict(dim=DIM, in_channels=IN_CHANNELS, embed_dims=EMBED_DIMS,
                                       dropout=DROPOUT, normalise=NORMALISE, n_stages=N_STAGES, margin=MARGIN)}, CKPT_PATH)
            print(f"  saved checkpoint (val pair-AUROC {va_auroc:.3f})")

    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()