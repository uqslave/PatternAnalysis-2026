import random
import numpy as np
import torch
from pathlib import Path

from modules import SiameseNetwork, CNNEmbed, ContrastiveLoss
from dataset import get_isic_pairloader, PairData

# ----------------------------- config -----------------------------
EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


dim = 64
in_channels = 3 # RGB
embed_dims = 128
dropout = 0.17
normalise = True
learning_rate = 5e-4

# Test ---------------------------------------------------------
#train_pair = PairData("train")
#print(train_pair[0])
#print("\nLength: ", len(train_pair))
# --------------------------------------------------------------

train_pair_loader = get_isic_pairloader("train")
validate_pair_loader = get_isic_pairloader("validate")

encoder = CNNEmbed(dim, in_channels, embed_dims, dropout, normalise)
model = SiameseNetwork(encoder)
model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = ContrastiveLoss()

print("Training model...")
for epoch in range(EPOCHS):
    model.train()
    print("Epoch: ", epoch)
    for batch_idx, (scans1, scans2) in enumerate(train_pair_loader):
        print("\tBatch: ", batch_idx)

        images1, labels1 = scans1
        images2, labels2 = scans2

        images1 = images1.to(device)
        images2 = images2.to(device)
        pairs = labels1 & labels2

        outputs = model(images1, images2)

        loss = criterion(outputs, pairs)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()