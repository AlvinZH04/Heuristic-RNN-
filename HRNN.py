import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time

# ----------------------------
# Synthetic Dataset Definition
# ----------------------------
class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=1024, input_dim=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        # Generate random sequences.
        self.data = torch.randn(num_samples, seq_len, input_dim)
        # Label is 1 if overall sum > 0, else 0.
        self.labels = (self.data.sum(dim=[1, 2]) > 0).long()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ----------------------------
# Model Definitions
# ----------------------------

# 1. Simple RNN (processing full sequence)
class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=2):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, h_n = self.rnn(x)  # h_n: (1, batch, hidden_dim)
        h_n = h_n.squeeze(0)    # (batch, hidden_dim)
        logits = self.fc(h_n)
        return logits

# 2. Hierarchical RNN (one-level divide and merge)
class HierarchicalRNNModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, num_classes=2, chunk_size=16):
        super(HierarchicalRNNModel, self).__init__()
        self.chunk_size = chunk_size
        self.local_rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.global_rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        num_chunks = seq_len // self.chunk_size
        # Reshape into chunks.
        x_chunks = x[:, :num_chunks * self.chunk_size, :].reshape(batch_size * num_chunks, self.chunk_size, -1)
        _, h_local = self.local_rnn(x_chunks)  # h_local: (1, batch*num_chunks, hidden_dim)
        h_local = h_local.squeeze(0).reshape(batch_size, num_chunks, -1)  # (batch, num_chunks, hidden_dim)
        _, h_global = self.global_rnn(h_local)  # h_global: (1, batch, hidden_dim)
        h_global = h_global.squeeze(0)
        logits = self.fc(h_global)
        return logits

# 3. Deep Hierarchical RNN (two-level divide and conquer)
class DeepHierarchicalRNNModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16, num_classes=2, chunk_size1=16, chunk_size2=4):
        """
        First level: divide into chunks of size chunk_size1.
        Second level: group chunk outputs (group size chunk_size2).
        Third level: merge with a global RNN.
        """
        super(DeepHierarchicalRNNModel, self).__init__()
        self.chunk_size1 = chunk_size1
        self.chunk_size2 = chunk_size2
        
        self.local_rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.middle_rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.global_rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # First level: process small chunks.
        num_chunks = seq_len // self.chunk_size1
        x_local = x[:, :num_chunks * self.chunk_size1, :].reshape(batch_size * num_chunks, self.chunk_size1, -1)
        _, h_local = self.local_rnn(x_local)  # (1, batch*num_chunks, hidden_dim)
        h_local = h_local.squeeze(0).reshape(batch_size, num_chunks, -1)  # (batch, num_chunks, hidden_dim)
        
        # Second level: group chunks.
        num_groups = num_chunks // self.chunk_size2
        h_local = h_local[:, :num_groups * self.chunk_size2, :].reshape(batch_size * num_groups, self.chunk_size2, -1)
        _, h_middle = self.middle_rnn(h_local)  # (1, batch*num_groups, hidden_dim)
        h_middle = h_middle.squeeze(0).reshape(batch_size, num_groups, -1)
        
        # Third level: merge group representations.
        _, h_global = self.global_rnn(h_middle)  # (1, batch, hidden_dim)
        h_global = h_global.squeeze(0)
        logits = self.fc(h_global)
        return logits

# 4. Transformer (pure transformer-based model)
class TransformerModel(nn.Module):
    def __init__(self, input_dim=10, model_dim=12, num_classes=2, seq_len=512, nhead=2, num_layers=1, dim_feedforward=24):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, 
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.seq_len = seq_len
        
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, model_dim)
        out = self.transformer_encoder(x)  # (batch, seq_len, model_dim)
        out = out.mean(dim=1)  # Global average pooling
        logits = self.fc(out)
        return logits

# 5. Hybrid RNN-Transformer Model
class HybridRNNTransformerModel(nn.Module):
    def __init__(self, input_dim=10, rnn_hidden_dim=12, num_classes=2, chunk_size=16, 
                 nhead=2, num_layers=1, dim_feedforward=24):
        """
        First level: divide the input sequence into chunks (e.g., chunk_size=16).
          Each chunk is processed with a simple RNN (local RNN) to yield a representation
          of dimension rnn_hidden_dim.
        Second level: the resulting sequence of chunk representations is fed into a Transformer encoder.
        """
        super(HybridRNNTransformerModel, self).__init__()
        self.chunk_size = chunk_size
        self.local_rnn = nn.RNN(input_dim, rnn_hidden_dim, batch_first=True)
        # Transformer encoder expects tokens of dimension rnn_hidden_dim.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rnn_hidden_dim, nhead=nhead, 
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(rnn_hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        num_chunks = seq_len // self.chunk_size
        # Process chunks: reshape x to (batch*num_chunks, chunk_size, input_dim)
        x_chunks = x[:, :num_chunks * self.chunk_size, :].reshape(batch_size * num_chunks, self.chunk_size, -1)
        _, h_local = self.local_rnn(x_chunks)  # h_local: (1, batch*num_chunks, rnn_hidden_dim)
        h_local = h_local.squeeze(0).reshape(batch_size, num_chunks, -1)  # (batch, num_chunks, rnn_hidden_dim)
        # Transformer on top of chunk representations.
        trans_out = self.transformer_encoder(h_local)  # (batch, num_chunks, rnn_hidden_dim)
        # Global pooling over chunk tokens.
        pooled = trans_out.mean(dim=1)  # (batch, rnn_hidden_dim)
        logits = self.fc(pooled)
        return logits

# ----------------------------
# Helper Functions
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# ----------------------------
# Main Comparison Function
# ----------------------------
def main():
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticSequenceDataset(num_samples=10000, seq_len=512, input_dim=10)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize models with similar parameter budgets.
    models = {
        "SimpleRNN": SimpleRNNModel(input_dim=10, hidden_dim=32, num_classes=2),
        "HierarchicalRNN": HierarchicalRNNModel(input_dim=10, hidden_dim=20, num_classes=2, chunk_size=16),
        "DeepHierarchicalRNN": DeepHierarchicalRNNModel(input_dim=10, hidden_dim=16, num_classes=2, chunk_size1=16, chunk_size2=4),
        "Transformer": TransformerModel(input_dim=10, model_dim=12, num_classes=2, seq_len=512, nhead=2, num_layers=1, dim_feedforward=24),
        "HybridRNNTransformer": HybridRNNTransformerModel(input_dim=10, rnn_hidden_dim=12, num_classes=2, chunk_size=16,
                                                          nhead=2, num_layers=1, dim_feedforward=24)
    }
    
    # Enable multi-GPU training if 2 or more GPUs are available.
    if torch.cuda.device_count() >= 2:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training.")
        for key in models:
            models[key] = nn.DataParallel(models[key])
    
    # Move models to device.
    for key in models:
        models[key] = models[key].to(device)
    
    # Print parameter counts.
    print("Parameter counts:")
    for name, model in models.items():
        count = count_parameters(model.module) if hasattr(model, 'module') else count_parameters(model)
        print(f"{name}: {count} parameters")
    
    results = {}
    num_epochs = 10
    for name, model in models.items():
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"\nTraining {name}...")
        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        elapsed_time = time.time() - start_time
        results[name] = {"time": elapsed_time, "val_accuracy": val_acc}
        print(f"{name} Total Training Time: {elapsed_time:.2f} seconds")
    
    print("\nComparison Results:")
    for name, res in results.items():
        print(f"{name}: Training Time = {res['time']:.2f}s, Validation Accuracy = {res['val_accuracy']:.4f}")

if __name__ == "__main__":
    main()
