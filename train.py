import torch
import torch.nn as nn
import time
import copy
import pandas as pd
from matplotlib import pyplot as plt

# Assume your CNN model is defined in model.py
from model import CNN

def train_model(model, train_dataloader, val_dataloader, num_epochs):

    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use Adam as optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # Standard loss for regression

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')  # Initialize best loss to a very large number

    # Loss Lists
    train_loss_all = []
    val_loss_all = []

    # Current Time
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Initialize the Parameters
        train_loss = 0.0
        val_loss = 0.0
        train_num = 0
        val_num = 0

        # -------------------- Training Phase --------------------
        model.train()  # Set model to training mode
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # Forward Pass
            output = model(b_x)
            # Calculate Loss
            loss = criterion(output, b_y)

            # Backward Pass and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * b_x.size(0)  # Multiply by current batch size
            train_num += b_x.size(0)

        # -------------------- Validation Phase --------------------
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for efficiency
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                # Forward Pass
                output = model(b_x)
                # Calculate Loss
                loss = criterion(output, b_y)

                # Statistics
                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)

        # Calculate average losses for the epoch
        epoch_train_loss = train_loss / train_num
        epoch_val_loss = val_loss / val_num
        train_loss_all.append(epoch_train_loss)
        val_loss_all.append(epoch_val_loss)

        print(f"{epoch} Train Loss: {epoch_train_loss:.4f}")
        print(f"{epoch} Val Loss: {epoch_val_loss:.4f}")

        # Deep copy the model if it has the best validation loss so far
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Time spent for training
    time_use = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    print("Best Val Loss: {:4f}".format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model
    torch.save(model.state_dict(), 'NILM_PIPELINE/best_model.pth')

    # Create a DataFrame from the recorded losses
    train_process = pd.DataFrame(data={"Epoch": range(num_epochs),
                                       "Train_Loss": train_loss_all,
                                       "Val_Loss": val_loss_all})
    
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


    