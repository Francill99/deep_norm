import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm

import sys
sys.path.append("cnn")  

from model import CNNClassifier

def validate_model(model, criterion, dataloader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
    avg_loss = total_loss / total
    error_rate = 100.0 * (1 - correct / total)
    return error_rate, avg_loss

def train(model, 
        train_loader,
        vali_loader,
        test_loader,
        device,
        optimizer,
        criterion,
        T):

    # Create 20 logging epochs in a logspace from 1 to 1000.
    log_epochs = np.unique(np.logspace(0, np.log10(T), num=20, dtype=int))
    print("Logging at epochs:", log_epochs)
    
    # Lists to store logged data.
    log_val_error = []
    log_val_loss  = []
    log_model_norm = []
    log_margins = []  # Each element is a numpy array of margin values from one validation batch.
    
    # Main training loop.
    for epoch in range(1, T + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch in log_epochs:
            val_error, val_loss = validate_model(model, criterion, vali_loader, device)
            #norm_value = model.compute_model_norm().item()
            norm_value = model.compute_spectral_complexity_ALT.item()
            
            # Get one batch from the validation loader.
            batch_val = next(iter(vali_loader))
            inputs_val, targets_val = batch_val
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            #margins = model.compute_margin_distribution(inputs_val, targets_val)
            margins = model.compute_margin_distribution_ALT(inputs_val, targets_val)
            margins_np = margins.cpu().numpy()
            
            log_val_error.append(val_error)
            log_val_loss.append(val_loss)
            log_model_norm.append(norm_value)
            log_margins.append(margins_np)
            
            print(f"Epoch {epoch}: Val Error = {val_error:.2f}%, Val Loss = {val_loss:.4f}, "
                  f"Model Norm = {norm_value:.4f}, Margin Mean = {margins_np.mean():.4e}, Margin Min = {margins.min():.4e}")
    
    # Evaluate on test set at the end.
    test_error, test_loss = validate_model(model, criterion, test_loader, device)
    print(f"Test Error: {test_error:.2f}%, Test Loss: {test_loss:.4f}")
    
    # Convert lists to numpy arrays where appropriate.
    log_val_error = np.array(log_val_error)
    log_val_loss  = np.array(log_val_loss)
    log_model_norm = np.array(log_model_norm)
    # log_margins is a list of numpy arrays (one per logged epoch).
    
    logs = {
        "log_epochs": log_epochs,
        "val_error": log_val_error,
        "val_loss": log_val_loss,
        "model_norm": log_model_norm,
        "margins": log_margins
    }
    return logs


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Load CIFAR-10 data with basic transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_set, val_set = random_split(dataset, [40000, 10000])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
vali_loader = DataLoader(val_set, batch_size=128)
test_loader = DataLoader(test_set, batch_size=128)

# Instantiate model, criterion, optimizer
model = CNNClassifier(conv_channels = [3, 32, 64], kernel_size = 3, mlp_layers = [512, 128, 10]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train for 10 epochs
logs = train(
    model=model,
    train_loader=train_loader,
    vali_loader=vali_loader,
    test_loader=test_loader,
    device=device,
    optimizer=optimizer,
    criterion=criterion,
    T=10
)