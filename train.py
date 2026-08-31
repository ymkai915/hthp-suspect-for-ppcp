import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import gc
from torch.cuda.amp import GradScaler, autocast
import time
import urllib.request
# from gnn_sp import GNN, temperature
import os


# Analyzing MGF Files
def load_mgf_with_ms2(filepath):
    data = []
    current_spectrum = {}
    ms2_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NAME="):
                if current_spectrum:
                    ms2_data = [(mz, intensity) for mz, intensity in ms2_data if intensity > 100]
                    current_spectrum["MS2"] = ms2_data
                    data.append(current_spectrum)
                current_spectrum = {"name": line[5:]}
                ms2_data = []
            elif line.startswith("SMILES="):
                current_spectrum["SMILES"] = line[7:]
            elif line.startswith("PEPMASS="):
                current_spectrum["PEPMASS"] = line[8:]
            elif line.startswith("CHARGE="):
                current_spectrum["CHARGE"] = line[7:]
            elif line and line[0].isdigit():
                mz, intensity = map(float, line.split())
                ms2_data.append((mz, intensity))

    if current_spectrum:
        ms2_data = [(mz, intensity) for mz, intensity in ms2_data if intensity > 100]
        current_spectrum["MS2"] = ms2_data
        data.append(current_spectrum)

    return data

# Convert molecular structures to graph data
def mol_to_graph(mol):
    node_features = []
    for atom in mol.GetAtoms():
        features = [0] * 12      # C/N/O/F/S/Cl/Na/Al/P/Br/K/Mg
        atomic_num = atom.GetAtomicNum()

        # Set the corresponding properties based on the atomic number of the element
        if atomic_num == 6:  # C
            features[0] = 1
        elif atomic_num == 7:  # N
            features[1] = 1
        elif atomic_num == 8:  # O
            features[2] = 1
        elif atomic_num == 9:  # F
            features[3] = 1
        elif atomic_num == 16:  # S
            features[4] = 1
        elif atomic_num == 17:  # Cl
            features[5] = 1
        elif atomic_num == 11:  # Na
            features[6] = 1
        elif atomic_num == 13:  # Al
            features[7] = 1
        elif atomic_num == 15:  # P
            features[8] = 1
        elif atomic_num == 35:  # Br
            features[9] = 1
        elif atomic_num == 19:  # K
            features[10] = 1
        elif atomic_num == 12:  # Mg
            features[11] = 1

        node_features.append(features)

    # Building Edge Features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        bond_features = [0] * 6  # Mono-bond / Di-bond / Tri-bond / Aromatic bond / Cyclic bond / Other
        if bond_type == 1.0:
            bond_features[0] = 1
        elif bond_type == 2.0:
            bond_features[1] = 1
        elif bond_type == 3.0:
            bond_features[2] = 1
        if bond.GetIsAromatic():
            bond_features[3] = 1
        if bond.IsInRing():
            bond_features[4] = 1
        edge_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_features.append(bond_features)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    return edge_index, node_features, edge_attr

# Converting MS2 Spectral Data to Vectors
def ms2_to_vector(ms2_data, mz_range=(74, 400), bins=326000, intensity_transform="sqrt"):
    bin_size = (mz_range[1] - mz_range[0]) / bins
    vector = np.zeros(bins, dtype=np.float32)
    # Select the 200 peaks with the highest intensity
    ms2_data = sorted(ms2_data, key=lambda x: x[1], reverse=True)[:200]

    for mz, intensity in ms2_data:
        if mz_range[0] <= mz < mz_range[1]:
            bin_idx = int((mz - mz_range[0]) / bin_size)
            if intensity_transform == "sqrt":
                intensity = np.sqrt(intensity)
            vector[bin_idx] += intensity

    total_intensity = vector.sum()
    if total_intensity > 0:
        vector /= total_intensity
    else:
        vector = np.zeros_like(vector)
    return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

# Improved MPNN Model
class MPNNWithAttention(nn.Module):
    def __init__(self, in_feats, h_feats=64, out_feats=326000, num_layers=4):
        super(MPNNWithAttention, self).__init__()
        self.num_layers = num_layers
        self.node_embedding = nn.Linear(in_feats, h_feats)
        self.mpnn_layers = nn.ModuleList(
            [GATConv(h_feats, h_feats, heads=4, concat=False),
             GATConv(h_feats, h_feats, heads=4, concat=False),
             GATConv(h_feats, h_feats, heads=4, concat=False)]
        )
        self.fc = nn.Linear(h_feats, out_feats)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)
        for layer in self.mpnn_layers:
            # print(h.size())
            h = F.relu(layer(h, edge_index))
        h = global_mean_pool(h, batch)
        h = self.dropout(h)
        return self.fc(h)

# Data Loading and Partitioning
data = load_mgf_with_ms2('./MASSBANK.mgf')

# Batch Processing
batch_size = 200
for i in range(0, len(data), batch_size):
    batch_data = data[i:i + batch_size]

mols = []
for item in data:
    smiles = item.get('SMILES')
    ms2_data = item.get('MS2')
    if smiles and ms2_data:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            edge_index, node_features, edge_attr = mol_to_graph(mol)
            ms2_vector = ms2_to_vector(ms2_data)
            mols.append(Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=ms2_vector
            ))

print(f"Loaded {len(mols)} molecules with MS2 data.")

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Segmentation
random.shuffle(mols)
train_size = int(0.8 * len(mols))
val_size = int(0.1 * len(mols))
test_size = len(mols) - train_size - val_size
train_mols, val_mols, test_mols = mols[:], mols[train_size:train_size + val_size], mols[train_size + val_size:]
print(train_size, val_size, test_size)

train_loader = DataLoader(train_mols, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_mols, batch_size=32, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_mols, batch_size=32, num_workers=4, pin_memory=True)
print(len(train_loader), len(val_loader), len(test_loader))

# Model Initialization
model = MPNNWithAttention(in_feats=12, h_feats=256, out_feats=326000, num_layers=3).to(device)
# model = GNN(num_layer=4, input_dim=12, emb_dim=1024, output_dim=326000, JK = "last", drop_ratio = 0, gnn_type = "gin", disable_fingerprint = False).to(device)

# Parameter Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

# Custom Loss Functions
def loss_fn_with_cosine(predicted, target, alpha=0.5):
    criterion_mes = nn.MSELoss()
    target = target.squeeze(1)
    mse_loss = criterion_mes(predicted, target)
    cosine_loss = 1 - cosine_similarity(predicted, target, dim=1).mean()
    return alpha * mse_loss + (1 - alpha) * cosine_loss, 1-cosine_loss, mse_loss

# Optimizers and Schedulers
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',  # 'min' indicates that the objective is to minimize the validation loss
                              factor=0.1,  # The percentage by which the learning rate is reduced each time, for example, from 0.001 to 0.0001
                              patience=10, # Tolerate a validation loss that does not decrease within 10 epochs
                              verbose=True, # Output information on learning rate adjustments
                              min_lr=1e-6) # Minimum Learning Rate Limit


scaler = GradScaler()

# Start the training and validation loops
num_epochs = 150
train_losses, val_losses = [], []

for epoch in range(1, num_epochs+1):
    # Training Loop
    model.train()
    train_loss = torch.zeros(1).to(device)
    for batch in train_loader:
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.y = batch.y.to(device).view(batch.num_graphs, -1)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
        loss, _, _ = loss_fn_with_cosine(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss


    train_loss /= len(train_loader)
    train_losses.append(train_loss.item())

    # Validation Loop
    model.eval()
    val_loss = torch.zeros(1).to(device)
    cos_sum = 0.0
    mse_sum = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch.x = batch.x.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_attr = batch.edge_attr.to(device)
            batch.y = batch.y.to(device)

            batch_size = batch.num_graphs
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
            target = batch.y.view(batch_size, -1)
            loss1, cosine_similarities, mse_score = loss_fn_with_cosine(out, target)
            val_loss += loss1
            cos_sum += cosine_similarities.item()
            mse_sum += mse_score.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss.item())
        cos_sim = cos_sum / len(val_loader)
        mse_sum = mse_sum / len(val_loader)

    if (epoch > 1) and (epoch % 10 == 0):
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mean_cosine_similarity': cos_sim,  # 保存完整序列
        }, f"mpnn_with_base256_{epoch}+.pth")


    # Call the scheduler to adjust the learning rate based on the validation loss
    scheduler.step(val_loss)

    # Get the current learning rate and print it
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f} "
        f"- LR: {current_lr:.6f} - cos: {cos_sim:.4f} - mse: {mse_sum:.4f}")

    # If the learning rate drops to its minimum value, terminate training
    if current_lr <= scheduler.min_lrs[0]:
        print("Learning rate has reached its minimum value. Stopping early.")
        break

# Plotting the Loss Curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig('./figs/loss.png', format='png')

# Test the model's performance and calculate the cosine similarity
model.eval()
test_cosine_similarities = []
test_mse = []
with torch.no_grad():
    for batch in test_loader:
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.y = batch.y.to(device)

        batch_size = batch.num_graphs
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
        target = batch.y.view(batch_size, -1)
        loss_total, cos, mse = loss_fn_with_cosine(out, target)
        temperature=5
        cosine_sim = torch.sigmoid(cosine_similarity(out, target, dim=1)*temperature)
        test_cosine_similarities.append(cosine_sim.cpu().numpy())
        test_mse.append(mse.cpu().numpy())

mean_cosine_similarity = np.mean(test_cosine_similarities)
test_mse = np.mean(test_mse)
print(f"Mean Cosine Similarity on Test Set: {mean_cosine_similarity:.4f} - mse: {test_mse}")
x1 = f"Mean Cosine Similarity on Test Set: {mean_cosine_similarity:.4f} - mse: {test_mse}"

# Check if the file exists
if os.path.exists("output.txt"):
    # The file exists; write in append mode
    with open("output.txt", "a", encoding="utf-8") as file:
        file.write(x1)
    print("The file exists; write in append mode")
else:
    # The file does not exist; create the file and write to it
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(x1)
    print("The file does not exist; create the file and write to it")

def comput_cos(A, B):
    # Calculate the dot product
    dot_product = np.dot(A, B)
    # Calculate the magnitudes of two vectors
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # Calculate the cosine similarity
    return dot_product / (norm_A * norm_B)
# Comparison of Predicted Results with Actual Spectra
model.eval()
with torch.no_grad():
    cos_200 = []
    cos_mz = []
    num = 0
    for batch in test_loader:
        num += 1
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.y = batch.y.to(device)
        batch_size = batch.num_graphs
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
        target = batch.y.view(batch_size, -1)

        mz_range = (74, 400)
        bins = 326000 # 326000
        bin_size = (mz_range[1] - mz_range[0]) / bins
        mz_values = np.linspace(mz_range[0], mz_range[1], bins)

        for i in range(min(5, batch_size)):
            # Predicted and Actual Spectra
            predicted_spectrum = out[i].cpu().numpy()
            target_spectrum = target[i].cpu().numpy()

            # Normalization
            predicted_spectrum /= np.max(np.abs(predicted_spectrum))
            target_spectrum /= np.max(np.abs(target_spectrum))

            # Filter the 200 m/z values with the highest response intensity from the predicted values
            top_indices = np.argsort(predicted_spectrum)[-200:]  # 强度从小到大排序，取最后 200 个
            filtered_mz_values = mz_values[top_indices]
            filtered_predicted_intensities = predicted_spectrum[top_indices]

            # Filter the 200 m/z values with the highest response intensities from the raw spectra
            top_target_indices = np.argsort(target_spectrum)[-200:]
            filtered_target_mz_values = mz_values[top_target_indices]
            filtered_target_intensities = target_spectrum[top_target_indices]

            cos_200.append(comput_cos(filtered_target_intensities, filtered_predicted_intensities))
            cos_mz.append(comput_cos(filtered_target_mz_values, filtered_mz_values))

    # print('cos_200:', np.mean(cos_200), 'cos_mz:', np.mean(cos_mz))
            # Drawing
            plt.figure(figsize=(10, 6))
            plt.stem(filtered_mz_values, filtered_predicted_intensities, linefmt='b-', markerfmt='bo', label="Predicted Spectrum", basefmt=" ")
            plt.stem(filtered_target_mz_values, -filtered_target_intensities, linefmt='orange', markerfmt='ro', label="Target Spectrum (Inverted)", basefmt=" ")

            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel('m/z')
            plt.ylabel('Normalized Intensity')
            plt.title(f'Sample {i + 1} - Predicted vs Target Spectrum (Top 200 Peaks)')
            plt.legend()
            plt.ylim([-1, 1])
            plt.show()
            plt.savefig(f'./figs/fig{num}{i+1}.png', format='png')
    print('cos_200:', np.mean(cos_200), 'cos_mz:', np.mean(cos_mz))
    x2 = f"cos_200: {np.mean(cos_200)}, cos_mz: {np.mean(cos_mz)}"
    if os.path.exists("output.txt"):
        # The file exists; write in append mode.
        with open("output.txt", "a", encoding="utf-8") as file:
            file.write(x2)
        print("The file exists; write in append mode.")
    else:
        # The file does not exist; create the file and write to it
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(x2)
        print("The file does not exist; create the file and write to it")
