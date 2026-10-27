"""
Single-image melanoma classification from a trained Siamese embedding.

Pipeline: embed train/val/test images (no augmentation) -> score each image against the TRAIN embeddings
(class prototypes, and k-nearest-neighbours) -> pick a threshold on VALIDATION -> report TEST metrics.
Usage: python predict.py --ckpt siamese_best.pt
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, f1_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dataset import IMAGE_DIMS, TRANSFORMS, ISICMelanomaDataset
from modules import CNNEmbed, SiameseNetwork

TARGET_NAMES = ["benign", "malignant"]


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    encoder = CNNEmbed(cfg["dim"], cfg["in_channels"], cfg["embed_dims"], cfg["dropout"],
                       cfg["normalise"], cfg.get("n_stages", 5))
    model = SiameseNetwork(encoder).to(device)
    model.load_state_dict(ckpt["model"])   # fails if the checkpoint came from a different encoder layout
    return model.eval()


@torch.no_grad()
def embed_split(model, split, device, batch_size, workers):
    # Plain transform (resize + normalise only) for EVERY split, including train.
    dataset = ISICMelanomaDataset(split, IMAGE_DIMS, TRANSFORMS["validate"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    zs, ys = [], []
    for x, y in loader:
        zs.append(model.encoder(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys).numpy()


# ------------------------------- scores (higher = more likely malignant) -------------------------------
def prototype_score(z_ref, y_ref, z):
    y = torch.from_numpy(y_ref)
    proto_b = z_ref[y == 0].mean(0, keepdim=True)
    proto_m = z_ref[y == 1].mean(0, keepdim=True)
    return (torch.cdist(z, proto_b) - torch.cdist(z, proto_m)).squeeze(1).numpy()


def knn_score(z_ref, y_ref, z, k=15, chunk=512, weighting="uniform"):
    """
    Malignant-neighbour score in [0, 1].
      uniform  : fraction of the k nearest training images that are malignant (only k+1 distinct values -> many ties)
      distance : same, but each neighbour is weighted by 1/distance (continuous, so far fewer ties)
    """
    y = torch.from_numpy(y_ref).float()
    out = []
    for i in range(0, len(z), chunk):
        d, idx = torch.cdist(z[i:i + chunk], z_ref).topk(k, largest=False)
        if weighting == "distance":
            w = 1.0 / (d + 1e-6)
            out.append((w * y[idx]).sum(1) / w.sum(1))
        else:
            out.append(y[idx].mean(1))
    return torch.cat(out).numpy()


# ------------------------------- reporting -------------------------------
def pick_threshold(y_val, s_val, metric="youden"):
    """Choose the decision threshold on VALIDATION data only."""
    if metric == "youden":                       # maximise TPR - FPR (favours malignant recall)
        fpr, tpr, thr = roc_curve(y_val, s_val)
        return thr[np.argmax(tpr - fpr)]
    candidates = np.unique(np.quantile(s_val, np.linspace(0.01, 0.99, 197)))

    def value(t):
        pred = (s_val >= t).astype(int)
        return accuracy_score(y_val, pred) if metric == "accuracy" else f1_score(y_val, pred, average="macro")
    return max(candidates, key=value)


def selective_report(y, s, thr):
    """Reject option: keep only the most confident predictions (confidence = distance from the threshold).
    Read 'malignant kept' and balanced accuracy: plain accuracy is inflated when the kept set is mostly benign."""
    order = np.argsort(-np.abs(s - thr))
    pred = (s >= thr).astype(int)
    print("  Reject-option (coverage = fraction of cases the model decides on):")
    for cov in (1.0, 0.8, 0.6, 0.4):
        keep = order[: int(cov * len(y))]
        yk, pk = y[keep], pred[keep]
        bal = balanced_accuracy_score(yk, pk) if len(set(yk)) > 1 else float("nan")
        print(f"    coverage {cov:4.0%}: accuracy {accuracy_score(yk, pk):.3f} | balanced acc {bal:.3f} | "
              f"malignant kept {int(yk.sum())}/{int(y.sum())}")


def report(name, y_val, s_val, y_eval, s_eval, split, metric):
    thr = pick_threshold(y_val, s_val, metric)
    pred = (s_eval >= thr).astype(int)
    prev = y_eval.mean()
    print(f"\n===== {name} | evaluated on {split} | threshold {thr:.4f} (metric: {metric}, chosen on validation) =====")
    print(f"  AUROC {roc_auc_score(y_eval, s_eval):.3f} | PR-AUC {average_precision_score(y_eval, s_eval):.3f} "
          f"(chance = prevalence {prev:.3f})")
    print(f"  accuracy {accuracy_score(y_eval, pred):.3f} (always-benign baseline {1 - prev:.3f}) | "
          f"balanced acc {balanced_accuracy_score(y_eval, pred):.3f} | macro-F1 {f1_score(y_eval, pred, average='macro'):.3f}")
    print(classification_report(y_eval, pred, target_names=TARGET_NAMES, digits=3, zero_division=0))
    selective_report(y_eval, s_eval, thr)


def pair_distances(z, y, n=3000, seed=0):
    """Are malignant images actually clustered, or only separated from benign ones?"""
    rng = np.random.default_rng(seed)
    b, m = np.flatnonzero(y == 0), np.flatnonzero(y == 1)

    def mean_d(ia, ib):
        i, j = rng.choice(ia, n), rng.choice(ib, n)
        keep = i != j
        return torch.norm(z[i[keep]] - z[j[keep]], dim=1).mean().item()

    print(f"Mean embedding distance (val): benign-benign {mean_d(b, b):.3f} | "
          f"malignant-malignant {mean_d(m, m):.3f} | benign-malignant {mean_d(b, m):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="siamese_best.pt")
    p.add_argument("--eval-split", choices=["validate", "test"], default="validate",
                   help="Use 'validate' while developing; run 'test' ONCE for the final number.")
    p.add_argument("--threshold-metric", choices=["youden", "accuracy", "macro_f1"], default="youden")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--knn-weight", choices=["uniform", "distance"], default="uniform")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    print("Embedding splits ...")
    z_tr, y_tr = embed_split(model, "train", device, args.batch_size, args.workers)
    z_va, y_va = embed_split(model, "validate", device, args.batch_size, args.workers)
    if args.eval_split == "test":
        z_ev, y_ev = embed_split(model, "test", device, args.batch_size, args.workers)
    else:
        z_ev, y_ev = z_va, y_va
    for n, y in (("train", y_tr), ("validate", y_va), (args.eval_split + " (eval)", y_ev)):
        print(f"  {n:16s}: {len(y)} images, {int(y.sum())} malignant ({100 * y.mean():.2f}%)")

    pair_distances(z_va, y_va)

    methods = {"prototype": prototype_score, "kNN": lambda zr, yr, z: knn_score(zr, yr, z, k=args.k, weighting=args.knn_weight)}
    saved = {}
    for name, fn in methods.items():
        s_va, s_ev = fn(z_tr, y_tr, z_va), fn(z_tr, y_tr, z_ev)
        print(f"\n[{name}] unique score values on eval split: {len(np.unique(s_ev))} of {len(s_ev)}")
        report(name, y_va, s_va, y_ev, s_ev, args.eval_split, args.threshold_metric)
        saved[f"{name}_val"], saved[f"{name}_eval"] = s_va, s_ev
    np.savez(f"predict_scores_{args.eval_split}.npz", y_val=y_va, y_eval=y_ev, **saved)


if __name__ == "__main__":
    main()