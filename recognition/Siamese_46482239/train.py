import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from modules import SiameseNetwork, CNNEmbed, ContrastiveLoss
from dataset import get_isic_pairloader, PairData

# ----------------------------- config -----------------------------
EPOCHS = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


dim = 16
in_channels = 3 # RGB
embed_dims = 256
dropout = 0.21
normalise = True
learning_rate = 0.0002

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
training_losses = []
validate_losses = []
for epoch in range(EPOCHS):
    print("Epoch: ", epoch + 1)
    batch_loss = []

    model.train()
    for batch_idx, (scans1, scans2) in enumerate(train_pair_loader):
        # print("\tBatch: ", batch_idx, end = "\t")

        images1, labels1 = scans1
        images2, labels2 = scans2

        images1, labels1 = images1.to(device), labels1.to(device)
        images2, labels2 = images2.to(device), labels2.to(device)
        pairs = (labels1 != labels2).float()

        embedding1, embedding2 = model(images1, images2)

        loss = criterion(embedding1, embedding2, pairs)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        batch_loss.append(loss.item())
        print("Loss: ", loss.item())
    training_losses.append(np.mean(batch_loss))
    batch_loss.clear()

    print("Validation model...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (scans1, scans2) in enumerate(validate_pair_loader):
            images1, labels1 = scans1
            images2, labels2 = scans2
            images1, labels1 = images1.to(device), labels1.to(device)
            images2, labels2 = images2.to(device), labels2.to(device)
            pairs = (labels1 != labels2).float()
            embedding1, embedding2 = model(images1, images2)
            loss = criterion(embedding1, embedding2, pairs)
            batch_loss.append(loss.item())
    validate_losses.append(np.mean(batch_loss))

x = np.linspace(1, EPOCHS)
plt.plot(training_losses, label="Training Loss", color="blue")
plt.plot(validate_losses, label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
