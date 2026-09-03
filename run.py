from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
# from gnn_sp import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
print('0000')
num = 150

# Define the Model
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

# Select a device, GPU first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize the model and optimizer
model = MPNNWithAttention(in_feats=12, h_feats=256, out_feats=326000, num_layers=3).to(device)
# model = GNN(num_layer=6, input_dim=12, emb_dim=1024, output_dim=326000, JK = "last", drop_ratio = 0, gnn_type = "gin", disable_fingerprint = False).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',  # 'min' 表示目标是最小化验证损失
                              factor=0.1,  # 学习率每次降低的比例，例如从 0.001 -> 0.0001
                              patience=10, # 容忍验证损失在 10 个 epoch 内不下降
                              verbose=True, # 输出学习率调整信息
                              min_lr=1e-6) # 最低学习率限制

# Initialize Variables
train_losses = []
val_losses = []
cosine_similarities = []
epoch = 0

# Loading the model's state
try:
    checkpoint = torch.load(f"./mpnn_with_base256.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epoch = checkpoint['epoch']

    # Fix the issue with `cosine_similarities`
    cosine_similarities = checkpoint.get('mean_cosine_similarity', [])
    if isinstance(cosine_similarities, (float, np.float32)):  # If it is a scalar
        cosine_similarities = [cosine_similarities]  # Switch to List View
    elif not isinstance(cosine_similarities, list):
        raise ValueError(`cosine_similarities` should be a list or an array.")

    print(f"The model has been loaded from ‘mpnn_with_base256_{num}+.pth’ and training has resumed from epoch {epoch + 1}.")
except FileNotFoundError:
    print("No checkpoint file found; initializing a new training state")
except KeyError as e:
    print(f"An error occurred while loading the checkpoint: Key {e} is missing.")

# Visualization of the change in cosine similarity over training iterations
if len(cosine_similarities) > 1:
    plt.plot(range(len(cosine_similarities)), cosine_similarities, label='Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Training Cosine Similarity Over Epochs')
    plt.legend()
    plt.show()
else:
    print(f"cosine_similarities: Insufficient data to plot: {cosine_similarities}")

# Print the status of the model, optimizer, and scheduler
print(model)
print(optimizer)
print(scheduler)
print(f"当前调度器状态: {scheduler.state_dict()}")


# Example of Training and Saving Logic
def train_model(num_epochs):
    global epoch, train_losses, val_losses, cosine_similarities

    for epoch in range(epoch, num_epochs):
        # 示例训练和保存逻辑
        train_loss = np.random.random()  # Simulation Training Loss
        val_loss = np.random.random()  # Simulation Verification Loss
        cosine_similarity = np.random.random()  # Simulated Cosine Similarity

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cosine_similarities.append(cosine_similarity)

        # Adjust the learning rate
        scheduler.step(val_loss)

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mean_cosine_similarity': cosine_similarities,  # 保存完整序列
        }, "mpnn_with_base256_{num}+.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Cosine Similarity: {cosine_similarity:.4f}")


# Sample Call Training
#train_model(num_epochs=5)


import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

mz_range = (74, 400)  # Adjust the scope based on actual circumstances

# Define the `mol_to_graph` function
def mol_to_graph(mol):
    node_features = []
    for atom in mol.GetAtoms():
        features = [0] * 12
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
    edge_attr = []

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
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_indices.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)  # An undirected graph requires bidirectional edges.

    # Convert `edge_index` and `edge_attr` to PyTorch tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Convert node features to PyTorch tensors
    node_features = torch.tensor(node_features, dtype=torch.float)

    return edge_index, node_features, edge_attr



# Define the CSV file path
input_csv_path = 'E:/predict.csv'
output_csv_path = 'E:/results_base256.csv'

# Read a CSV File
df = pd.read_csv(input_csv_path)

# Check for the ‘SMILES’ column
if 'SMILES' not in df.columns:
    raise ValueError("The CSV file must include a ‘SMILES’ column.")


# Storing Prediction Results
results = []

# Model Evaluation Methods
model.eval()

# Corrections to the Model Prediction Logic
with torch.no_grad():
    for index, row in df.iterrows():
        smiles = str(row['SMILES'])  # Ensure that SMILES is a string
        compound_id = str(row.get('id', index))  # Ensure that the id is a string
        # Convert to a molecular object
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"Invalid SMILES for row {index}: {smiles}")
            continue
        # Convert to graph data
        edge_index, node_features, edge_attr = mol_to_graph(mol)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        # Create a Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        # Model Predictions
        graph_data = graph_data.to(device)
        out = model(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
            torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=device)  # 假设 batch=0
        )
        # Check the Output Shape
        print(f"Output shape: {out.shape}")
        predicted_spectrum = out.squeeze().cpu().numpy()  # Remove the “batch” dimension
        bins = 326000  # Adjust to a smaller number of bins: 326,000
        mz_values = np.linspace(mz_range[0], mz_range[1], bins)
        if len(predicted_spectrum) != bins:
            raise ValueError(f"The predicted spectrum length ({len(predicted_spectrum)}) and the number of spectrum bins ({bins}) do not match!")
        # Retrieve the top 100 peaks with the highest intensity
        top_indices = np.argsort(predicted_spectrum)[-100:]
        filtered_mz_values = mz_values[top_indices]
        filtered_intensities = predicted_spectrum[top_indices]
        # Flatten m/z and intensity into rows
        result_row = {'id': compound_id, 'SMILES': smiles}
        for i, (mz, intensity) in enumerate(zip(filtered_mz_values, filtered_intensities)):
            result_row[f'mz_{i+1}'] = mz
            result_row[f'intensity_{i+1}'] = intensity
        results.append(result_row)

# Convert the results to a DataFrame and save them
results_df = pd.DataFrame(results)
results_df = results_df.astype(str)  # Ensure that all columns are of the string type
results_df.to_csv(output_csv_path, index=False)



print(f"Predicted MS2 spectra saved to {output_csv_path}")
