import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TextClassificationMLP(nn.Module):
    """Deep Learning Multi-Layer Perceptron for processing fast NLP feature vectors."""
    def __init__(self, input_dim, output_dim):
        super(TextClassificationMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def train_dl_layer(X_train_sparse, y_train_encoded, output_classes):
    """Runs a lightning-fast Deep Learning routine to capture loss logs."""
    print("🧠 [Deep Learning] Initializing PyTorch Neural Network Weights...")
    
    # Take a representative slice to avoid long training wait times
    X_tensor = torch.FloatTensor(X_train_sparse.toarray()[:8000])
    y_tensor = torch.LongTensor(y_train_encoded[:8000])
    
    model = TextClassificationMLP(X_tensor.shape[1], output_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    loss_history = []
    print("🏋️‍♂️ [Deep Learning] Backpropagation active over target tensor blocks...")
    
    for epoch in range(15):  # Fast convergence iterations
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    # Generate mock probabilities for confidence distribution logs
    model.eval()
    with torch.no_grad():
        preds = torch.softmax(model(X_tensor), dim=1).numpy()
        confidences = np.max(preds, axis=1)
        
    print(f"✅ [Deep Learning] Loss successfully optimized down to: {loss_history[-1]:.4f}")
    return loss_history, confidences