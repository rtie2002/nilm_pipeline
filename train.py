import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import time
import copy
import pandas as pd
from matplotlib import pyplot as plt

# Assume your CNN model is defined in model.py
from model import CNN

def train_model(model, train_dataloader, val_dataloader, num_epochs, scaler_y=None, learning_rate=1e-4):
    """
    Train the model with proper validation.
    
    Args:
        model: PyTorch model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        num_epochs: Number of training epochs
        scaler_y: StandardScaler for inverse transforming predictions (optional)
        learning_rate: Initial learning rate (default: 1e-4, recommended for regression)
    """
    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use Adam as optimizer with configurable learning rate
    # Lower learning rate (1e-4) is recommended for regression tasks to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()  # Standard loss for regression

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')  # Initialize best loss to a very large number

    # Loss Lists
    train_loss_all = []
    val_loss_all = []
    val_rmse_all = []
    val_mae_all = []

    # Current Time
    since = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        # Initialize the Parameters
        train_loss = 0.0
        val_loss = 0.0
        train_num = 0
        val_num = 0

        # -------------------- Training Phase --------------------
        model.train()  # Set model to training mode
        train_bar = tqdm(
            train_dataloader,
            desc=f"Train Epoch {epoch}/{num_epochs}",
            leave=False,
        )
        for step, (b_x, b_y) in enumerate(train_bar, start=1):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # Forward Pass
            output = model(b_x)
            # Calculate Loss
            loss = criterion(output, b_y)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Backward Pass and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * b_x.size(0)  # Multiply by current batch size
            train_num += b_x.size(0)

        # -------------------- Validation Phase --------------------
        model.eval()  # Set model to evaluation mode
        val_preds = []
        val_targets = []
        with torch.no_grad():  # Disable gradient computation for efficiency
            val_bar = tqdm(
                val_dataloader,
                desc=f"Val   Epoch {epoch}/{num_epochs}",
                leave=False,
            )
            for step, (b_x, b_y) in enumerate(val_bar, start=1):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                # Forward Pass
                output = model(b_x)
                # Calculate Loss
                loss = criterion(output, b_y)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

                # Statistics
                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)
                if scaler_y is not None:
                    val_preds.append(output.detach().cpu())
                    val_targets.append(b_y.detach().cpu())

        # Calculate average losses for the epoch
        epoch_train_loss = train_loss / max(train_num, 1)
        epoch_val_loss = val_loss / max(val_num, 1)
        train_loss_all.append(epoch_train_loss)
        val_loss_all.append(epoch_val_loss)

        epoch_val_rmse = None
        epoch_val_mae = None
        if scaler_y is not None and val_preds:
            preds_np = torch.cat(val_preds).numpy()
            targets_np = torch.cat(val_targets).numpy()
            preds_real = scaler_y.inverse_transform(preds_np)
            targets_real = scaler_y.inverse_transform(targets_np)
            diff = preds_real - targets_real
            epoch_val_rmse = float(np.sqrt(np.mean(diff ** 2)))
            epoch_val_mae = float(np.mean(np.abs(diff)))
            val_rmse_all.append(epoch_val_rmse)
            val_mae_all.append(epoch_val_mae)

        print(f"{epoch} Train Loss: {epoch_train_loss:.4f}")
        print(f"{epoch} Val Loss: {epoch_val_loss:.4f}")
        if epoch_val_rmse is not None:
            print(f"{epoch} Val RMSE (real units): {epoch_val_rmse:.4f}")
            print(f"{epoch} Val MAE (real units): {epoch_val_mae:.4f}")

        # Deep copy the model if it has the best validation loss so far
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step(epoch_val_loss)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Time spent for training
    time_use = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    print("Best Val Loss: {:4f}".format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model
    torch.save(model.state_dict(), 'best_model.pth')

    # Create a DataFrame from the recorded losses
    process_dict = {"Epoch": list(range(1, num_epochs + 1)),
                    "Train_Loss": train_loss_all,
                    "Val_Loss": val_loss_all}
    if val_rmse_all:
        process_dict["Val_RMSE_real"] = val_rmse_all
        process_dict["Val_MAE_real"] = val_mae_all
    train_process = pd.DataFrame(data=process_dict)
    
    return model, train_process

def plot_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.plot(train_process["Epoch"], train_process["Train_Loss"], 'ro-', label="Train Loss")
    plt.plot(train_process["Epoch"], train_process["Val_Loss"], 'bs-', label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.show()


    