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
        T,
        other_norms = False,
        norms_every_steps=None):

    # Create 50 logging epochs in a logspace from 1 to T.
    log_epochs = np.unique(np.logspace(np.log10(1), np.log10(T), num=50, dtype=int))
    print("Logging at epochs:", log_epochs)
    
    # Lists to store logged data.
    log_train_error = []
    log_train_loss  = []
    log_val_error = []
    log_val_loss  = []
    log_model_norm = []
    ###
    log_l1       = []
    log_fro      = []
    log_g_2_1    = []
    log_spectral  = []
    #log_path_n    =  []
    #log_f_rao     = []

    #log_margins = []  # Each element is a numpy array of margin values from one validation batch.
    j_index = 0
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
            if norms_every_steps is not None:
                j_index += 1
                if norms_every_steps % j_index == 0:
                    train_error, train_loss = validate_model(model, criterion, train_loader, device)
                    val_error, val_loss = validate_model(model, criterion, test_loader, device)
                    norm_value = model.compute_model_norm().item()
                    log_train_error.append(train_error)
                    log_val_error.append(val_error)
                    log_train_loss.append(train_loss)
                    log_val_loss.append(val_loss)
                    log_model_norm.append(norm_value)
                    j_index = 0
                    print(f"P:{P} Epoch {epoch}: Train Error = {train_error:.2f}%, Val Error = {val_error:.2f}%, Val Loss = {val_loss:.4f}, "
                  f"Model Norm = {norm_value:.4f}")
        
        if epoch in log_epochs and norms_every_steps is None:
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

            if other_norms==True:
                l1 = model.compute_l1_norm().item()
                fro = model.compute_frobenius_norm().item()
                g_2_1 = model.compute_group_2_1_norm().item()
                spectral = model.compute_spectral_norm().item()
                #path_n = model.compute_path_norm().item()
                #data_iter = iter(train_loader)
                #images, labels = next(data_iter)

                # 2) Move to device
                #images = images.to(device)
                #labels = labels.to(device)

                # 3) Compute Fisher–Rao norm (as a Python float)
                #f_rao = model.compute_fisher_rao_norm(images, labels).item()
                
                log_l1.append(l1)      
                log_fro.append(fro)     
                log_g_2_1.append(g_2_1)   
                log_spectral.append(spectral)
                #log_path_n.append(path_n)  
                #log_f_rao.append(f_rao)   

            #log_margins.append(margins_np)
            if other_norms==False:
                print(f"P:{P} Epoch {epoch}: Train Error = {train_error:.2f}%, Val Error = {val_error:.2f}%, Val Loss = {val_loss:.4f}, "
                  f"Model Norm = {norm_value:.4f}")
            else:
                print(f"P:{P} Epoch {epoch}: Train Error = {train_error:.2f}%, Val Error = {val_error:.2f}%, Val Loss = {val_loss:.4f}, "
                  f"Model Norm = {norm_value:.4f},"f"L1 = {l1:.4f}",f"fro = {fro:.4f}",f"g_2_1 = {g_2_1:.4f}",f"Spectral = {spectral:.4f}")
    
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
    
    if other_norms==False:
        logs = {
            "log_epochs": log_epochs,
            "train_error": log_train_error,
            "train_loss": log_train_loss,
            "val_error": log_val_error,
            "val_loss": log_val_loss,
            "model_norm": log_model_norm
            #"margins": log_margins
        }
    else:
        logs = {
            "log_epochs": log_epochs,
            "train_error": log_train_error,
            "train_loss": log_train_loss,
            "val_error": log_val_error,
            "val_loss": log_val_loss,
            "model_norm": log_model_norm,
            "l1": log_l1      ,
            "fro": log_fro     ,
            "g_2_1": log_g_2_1   ,
            "spectral": log_spectral,
            #"path_n": log_path_n  ,
            #"f_rao": log_f_rao   ,
        }
    return logs


    