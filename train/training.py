import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm

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
        P,
        T):

    # Create 50 logging epochs in a logspace from 1 to 1000.
    log_epochs = np.unique(np.logspace(np.log10(1), np.log10(T), num=50, dtype=int))
    print("Logging at epochs:", log_epochs)
    
    # Lists to store logged data.
    log_train_error = []
    log_train_loss  = []
    log_val_error = []
    log_val_loss  = []
    log_model_norm = []
    #log_margins = []  # Each element is a numpy array of margin values from one validation batch.
    
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
            model.eval()
            train_error, train_loss = validate_model(model, criterion, train_loader, device)
            val_error, val_loss = validate_model(model, criterion, test_loader, device)
            norm_value = model.compute_model_norm().item()
            
            # Get one batch from the train loader.
            '''
            margins_np = np.array([])
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
            
                margins = model.compute_margin_distribution(inputs, targets)
                margins_np = np.append(margins_np, margins.cpu().numpy())
            '''
            log_train_error.append(train_error)
            log_val_error.append(val_error)
            log_train_loss.append(train_loss)
            log_val_loss.append(val_loss)
            log_model_norm.append(norm_value)
            #log_margins.append(margins_np)
            
            print(f"P:{P} Epoch {epoch}: Train Error = {train_error:.2f}%, Val Error = {val_error:.2f}%, Val Loss = {val_loss:.4f}, "
                  f"Model Norm = {norm_value:.4f}")
    
    # Evaluate on test set at the end.
    test_error, test_loss = validate_model(model, criterion, test_loader, device)
    print(f"Test Error: {test_error:.2f}%, Test Loss: {test_loss:.4f}")
    
    # Convert lists to numpy arrays where appropriate.
    log_train_error = np.array(log_train_error)
    log_train_loss  = np.array(log_train_loss)
    log_val_error = np.array(log_val_error)
    log_val_loss  = np.array(log_val_loss)
    log_model_norm = np.array(log_model_norm)
    # log_margins is a list of numpy arrays (one per logged epoch).
    
    logs = {
        "log_epochs": log_epochs,
        "train_error": log_train_error,
        "train_loss": log_train_loss,
        "val_error": log_val_error,
        "val_loss": log_val_loss,
        "model_norm": log_model_norm
        #"margins": log_margins
    }
    return logs


    