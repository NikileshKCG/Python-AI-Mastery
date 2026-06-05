"""
Project 10: Neural Network from Scratch
Concepts: PyTorch tensors, autograd, nn.Module,
          forward pass, backpropagation, optimizers,
          loss functions, training loop, batch processing,
          DataLoader, dropout, batch normalization,
          learning rate scheduling, model saving/loading,
          multi-class classification & regression with deep learning
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import time
warnings.filterwarnings("ignore")

# ----- PyTorch ------------------------------------------------------
import torch
import torch.nn as nn               # neural network building blocks
import torch.nn.functional as F     # activation function, loss functions
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# ----- scikit-learn (for data prep only) ----------------------------
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Detect device - use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Running on: {DEVICE}")

#--------------------------------------------------------------
# SECTION 1: PyTorch Tensor Fundamentals
# Concepts: Tensor = the core data structure in PyTorch
#                    Like numpy arrays but with GPU support + autograd
#--------------------------------------------------------------

def tensor_basics():
    print("\n ===== PyTorch Tensor Basics ============================")

    # Creating tensors
    t1 = torch.tensor([1.0, 2.0, 3.0])              # from Python list
    t2 = torch.zeros(3, 4)                          # 3x4 zeros
    t3 = torch.ones(2, 3)                           # 2x3 ones
    t4 = torch.rand(3, 3)                           # random uniform 0-1
    t5 = torch.randn(3, 3)                          # random normal (mean=0, std=1)
    t6 = torch.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]

    print(f"\n tensor([1, 2, 3]) : {t1}")
    print(f" zeros(3, 4) shape : {t2.shape}")
    print(f" rand(3, 3) :\n {t4}")

    # Tensor operations - identical syntax to numpy
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print(f"\n a + b            : \n {a + b}")
    print(f" a @ b (matmul)     : \n {torch.matmul(a, b)}")    #matrix multiplication
    print(f" a.T (transpose)    : \n {a.T}")

    # Tensor - NumPy Conversion
    np_arr          = np.array([1.0, 2.0, 3.0])
    tensor_from_np  = torch.from_numpy(np_arr)          # numpy -> tensor
    back_to_np      = tensor_from_np.numpy()            # tensor -> numpy
    print(f"\n numpy -> temsor : {tensor_from_np}")
    print(f"  tensor -> temsor : {back_to_np}")

    # Reshape
    x = torch.arange(12).float()
    print(f"\n Original         : {x}")
    print(f" Reshaped (3x4)     :\n {x.reshape(3, 4)}")

    # ---- Autograd -- automatic differentiation -----------------------
    # Concept: requires_grad=True tells PyTorch to track all operations
    #          so it can compute gradients (derivatives) automatically.
    #          This is backpropagation
    print(f"\n ---- Autogard (Backpropagation Engine) -----\n")
    x = torch.tensor(3.0, requires_grad=True)      # x =3
    y = x ** 2 + 2 * x + 1                          # y = x² + 2x + 1
    y.backward()                                    # compute dy/dx
    print(f" y = x²+2x+1 at x=3 : y = {y.item():.1f}")
    print(f" dy/dx (gradient)   : {x.grad.item():.1f} (expected: 2x+2 =8)")

    # Multi-variable gradient
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    loss = (w * 3 + b - 10) ** 2
    loss.backward()
    print(f"\n loss = (w*3+b-10)² at w=2, b=1 : {loss.item():.1f}")
    print(f"\n dloss/dw                       : {w.grad.item():.1f}")
    print(f"\n dloss/db                       : {b.grad.item():.1f}")

#--------------------------------------------------------------
# SECTION 2: Building Blocks - Activation Functions 
# Concepts: Activation add non-Linearity - without them,
#           stacking layers is the same as one layer (just math)
#--------------------------------------------------------------

def plot_activations():
    """Visualize common activation functions."""
    x = torch.linspace(-5, 5, 200)

    activations = {
        "ReLu"          : F.relu(x),
        "Sigmoid"       : torch.sigmoid(x),
        "Tanh"          : torch.tanh(x),
        "Leaky ReLu"    : F.leaky_relu(x, 0.1),
        "ELU"           : F.elu(x),
        "Softmax"       : F.softmax(x.unsqueeze(0), dim=1).squeeze(),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Activation Functions", fontsize=14, color="white")
    fig.patch.set_facecolor("#0f0f0f")
    colors = ["#e94560", "#05c46b", "#ffd460", "#533483", "#0f3460", "#e94560"]

    for ax, (name, values), color in zip(axes.flat, activations.items(),  colors):
        ax.set_facecolor("#1a1a2e")
        ax.plot(x.numpy(), values.detach().numpy(), color=color, linewidth=2.5)
        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_title(name, color="white")
        ax.tick_params(colors="#aaa")
        ax.grid(alpha=0.2, color="#444")
    
    plt.tight_layout()
    plt.savefig("activations.png", dpi=130,
                bbox_inches="tight", facecolor="#0f0f0f")
    print(" Saved: activations.png")
    plt.show(); plt.close()


#--------------------------------------------------------------
# SECTION 3: Manual Neural Network (No PyTorch on)
# Concepts: Build forward pass & backprop by hand
#           so you truly understand what nn.Module does for you
#--------------------------------------------------------------

def manual_neural_network():
    """ 
    A single-hidden-layer network built from scratch using only tensors.
    Classifies XOR problem -- shows why non-linearity is needed.
    """
    print("\n ===== Manual Neural Network (XOR) ==================")

    # XOR dataset - not linearly separable (Linear Regressionn fails here!)
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # Network: 2 inputs -> 4 hidden -> 1 output
    # Initialize weights randomly with Xavier initialization
    W1  = torch.randn(2, 4, requires_grad=True) * 0.5
    b1  = torch.zeros(1, 4, requires_grad=True)
    W2  = torch.randn(4, 1, requires_grad=True) * 0.5
    b2  = torch.zeros(1, 1, requires_grad=True)

    lr     = 0.1
    losses = []

    for epoch in range(2000):
        # ---- Forward Pass ------------------------------------------
        # Layer 1: Linear transformation + ReLU activation
        z1 = X @ W1 + b1                # matrix multiplication + bias
        a1 = torch.relu(z1)             # activation function

        # Layer 2: Linear transformation + Sigmoid (output 0-1)
        z2 = a1 @ W2 + b2
        output = torch.sigmoid(z2)

        # ----- Loss Calculation -----------------------------------------
        # Binary Cross-Enntropy: measures how wrong our probabilities are
        loss = F.binary_cross_entropy(output, y)

        # ----- Backward Pass (Backpropagation) --------------------------
        loss.backward()                 # PyTorch computes. ALL gradients

        # ----- Weight: Update (Gradient Descent) ------------------------
        with torch.no_grad():           # don't track these updates
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad
        
        # Zero gradients - CRITICAL: must clear before next backward()
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

        losses.append(loss.item())

        if (epoch + 1) % 500 == 0:
            print(f" Epoch {epoch+1:>5} | Loss: {loss.item():.6f}")
    
    # Final predictions
    with torch.no_grad():
        z1      = X @ W1 + b1
        a1      = torch.result(z1)
        z2      = a1 @ W2 + b2
        preds   = torch.sigmoid(z2).round()
    
    print(f"\n XOR Predictions.")
    for i in range(4):
        correct = "✅" if preds[i].item() == y[i].item() else "❌"
        print(f" Input {X[i].tolist()} -> Pred: {preds[i].item():.0f} Target: {y[i].item():.0f} {correct}")

    return losses


#--------------------------------------------------------------
# SECTION 4: nn.Module - The PyTorch way
# Concepts: Subclass nn.Module, define layers in __init__,
#           implement forward() - same pattern as ALL PyTorch models
#           including ResNet, BERT, GPT
#--------------------------------------------------------------

class BinaryClassifier(nn.Module):
    """
    Simple feedforward network for binary classification.
    Inherits from nn.Module - same pattern as ALL PyTorch models
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super().__init__()          # MUST call parent __init__ (similar in project 3!)

        # Build layers dynamically from hidden_dim list
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),    # fully connected layer
                nn.BatchNorm1d(hidden_dim),         # normalize activations -> stable training
                nn.ReLU(),                          # non-linearity
                nn.Dropout(dropout_rate),           # randomly zero neurons -> prevents overfitting
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))           # output layer
        self.network = nn.Sequential(*layers)           # pack into one sequential block

    def forward(self, x):
        """ 
        Forward pass - called automatically wehn you do model(x).
        Define how data flows through the network.
        """
        return torch.sigmoid(self.network(x))

class MultiClassifier(nn.Module):
    """ Feedforward network for multi-class classification. """

    def __int__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),    # fully connected layer
                nn.BatchNorm1d(hidden_dim),         # normalize activations -> stable training
                nn.ReLU(),                          # non-linearity
                nn.Dropout(dropout_rate),           # randomly zero neurons -> prevents overfitting
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))           
        self.network = nn.Sequential(*layers)  

    def forward(self, x):
        return self.network(x)      # raw logits - CrossEntropyLoss applies softmax
    
class RegressionNet(nn.Module):
    """ Deep neural network for regression. """

    def __int__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),    
                nn.BatchNorm1d(hidden_dim),         
                nn.ReLU(),                          
                nn.Dropout(dropout_rate),           
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))           
        self.network = nn.Sequential(*layers)  

    
    def forward(self, x):
        return self.network(x)      # no activation - raw value for regression
        

#--------------------------------------------------------------
# SECTION 5: Custom Dataset Class
# Concepts: Subclass torch.utils.data.Dataset
#           Enables batching, shuffling, parallel loading
#--------------------------------------------------------------

class TabularDataset(Dataset):
    """ Custom Dataset for tabular (structured) data. """

    def __init__(self, X, y):
        # Convert numpy arrays to float32 tensors
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """Return total number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return one sample by index - DataLoader calls this automatically. """
        return self.X[idx], self.y[idx]

#--------------------------------------------------------------
# SECTION 6: Training Loop
# Concepts: The core of ALL deep learning
#           1. Forward pass     -> get predictions
#           2. Compute loss     -> measure error
#           3. Backward pass    -> compute gradients
#           4. Update weights   -> optimizer.step()
#           5. Zero gradients   -> optimizer.zero_grad()
#--------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    """Run one full epoch of training."""
    model.train()           # training mode - enables dropout, batchnorm
    total_loss = 0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # ----- The 5-step training loop -----------------------------
        optimizer.zero_grad()                   # 5. Zero gradients (do FIRST to avoid accumulation)
        output = model(X_batch)                 # 1. Forward pass
        loss   = criterion(output, y_batch)     # 2. Compute loss
        loss.backward()                         # 3. Backward pass (compute gradients)
        optimizer.step()                        # 4. Update weights 
        # ------------------------------------------------------------

        total_loss += loss.item()

        # Accuracy (for classifiers)
        if output.shape[1] if output.dim() > 1 else 1 > 1:
            preds    = output.argmax(dim=1)
            correct += (preds == y_batch.long()).sum().item()
        else:
            preds    = (output.squeeze() > 0.5).float()
            correct += (preds == y_batch.squeeze()).sum().item()
        total += len(y_batch)
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    """Evaluate model on validation/test set."""
    model.eval()            # eval mode - disables dropout, batchnorm uses running stats
    total_loss = 0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []
    
    with torch.no_grad():                   # no gradient computation needed for evaluation
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            output      = model(X_batch)
            loss        = criterion(output, y_batch)
            total_loss += loss.item()

            if output.shape[1] if output.dim() > 1 else 1 > 1:
                preds    = output.argmax(dim=1)
                correct += (preds == y_batch.long()).sum().item()
                all_preds.extend(preds.cpu().numpy())
            else:
                preds    = (output.squeeze() > 0.5).float()
                correct += (preds == y_batch.squeeze()).sum().item()
                all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(y_batch.squeeze().cpu().numpy())
            total += len(y_batch)

    return total_loss / len(loader), correct / total, all_preds, all_labels


#--------------------------------------------------------------
# SECTION 7: Full Traning with History
# Concepts: Track loss/accuracy over epochs -> learning curve
#--------------------------------------------------------------

def full_traning(model, train_loader, val_loader, optimizer,
                 criterion, scheduler=None, epochs=50, task_name="Model"):
    """Complete traning loop with validation and history tacking."""

    history         = {"traning_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_loss   = float("inf")
    best_state      = None

    print(f"\n Training {task_name}....")
    print(f" {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Train Acc':>10} {'Val Acc':>9}")
    print(" " + "-" * 52)

    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc   = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Learning rate scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f" {epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
                  f" {train_acc:>10.4f} {val_acc:>9.4f}")
        
    elapsed = time.time() - start
    print(f"\n Training complete in {elapsed:.1f}s")
    print(f" Best val loss: {best_val_loss:.4f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return history


#--------------------------------------------------------------
# SECTION 8: Learning curve visualization
#--------------------------------------------------------------

def plot_training_history(history, title="Training History", filename="training_history.png"):
    """Plot loss and accuracy curves over epochs."""
    fig, axes = plt.subplot(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, color="white")
    fig.patch.set_facecolor("#0f0f0f")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.grid(alpha=0.3, color="#444")
    
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"],
                 color="#e94560", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, history["Val_loss"],
                 color="#05c46b", linewidth=2, linestyle="--", label="Val Loss")
    axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend()

    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"],
                 color="#ffd460", linewidth=2, label="Train Loss")
    axes[1].plot(epochs, history["Val_acc"],
                 color="#533483", linewidth=2, linestyle="--", label="Val Acc")
    axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
    print(" Saved: {filename}")
    plt.show(); plt.close()


#--------------------------------------------------------------
# SECTION 9: Task 1 - Binary Calssification
# Concepts: sklearn make_classification (2 classes)
#--------------------------------------------------------------

def task_binary_classification():
    print("\n ===== Task 1: Binary Classification ==================")

    #Generate dataset
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )

    # Preprocess
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Dataset & DataLoader
    train_ds = TabularDataset(X_train, y_train)
    val_dss  = TabularDataset(X_val, y_val)
    test_ds  = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)
    test_loader   = DataLoader(tes_ds, batch_size=64)

    # Model
    model = BinaryClassifier(
        input_dim= 20,
        hidden_dims= [128, 64, 32],
        dropout_rate= 0.3
    ).to(DEVICE)

    print(f"\n Model Architecture.")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Total parameters: {total_params:,}")

    # Optimizer + Loss + Scheduler
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    # Train
    history = full_traning(
        model, train_loader, val_loader,
        optimizer, criterion, scheduler,
        epochs=60, task_name="Binary Classifier"
    )

    # Test evaluation
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\n ----- Test Results -----------------------------------")
    print(f" Test Accuracy : {test_acc:.4f}")
    print(f" Test Loss     : {test_loss:.4f}")

    plot_training_history(history, "Binary Classifier -- Training History",
                          "binary_training.png")

    torch.save(model.state_dict(), "binary_classifier.pth")
    print(" Saved: binary_classifier.pth")

    return model

#--------------------------------------------------------------
# SECTION 10: Task 2 - Multi-Class Classification
# Concepts: Iris (4 features - 3 classes)
#--------------------------------------------------------------

def task_multiclass_classification():
    print("\n ===== Task 2: Multi-Class Classification (Iris)==========")

    # Load Iris dataset
    iris        = load_iris()
    X, y        = iris.data.astype(np.float32), iris.target.astype(np.int64)
    classes     = iris.target_names

    # Preprocess
    scaler = StandardScaler()
    X      = scaler.fit_transform(X).astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds   = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)
    test_loader  = DataLoader(test_ds, batch_size=16)

    # Model
    model       = MultiClassifier(4, [64, 32], num_classes=3, dropout_rate=0.2).to(DEVICE)
    optimizer   = Adam(model.parameters(), lr=0.001)
    criterion   = nn.CrossEntropyLoss()         # inncludes softmax  -  use with raw logits
    scheduler   = StepLR(optimizer, step_size=20, gamma=0.5)

    history     = full_training(
        model, train_loader, val_loader,
        optimizer, criterion, scheduler,
        epochs=80, task_name="Iris Classifier"
    )

    # Final test
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            out     = model(X_b.to(DEVICE))
            preds   = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_test, all_preds)
    print(f"\n Test Accuracy: {acc:.4f}")
    print(f"\n Calssification Report:")
    print(classification_report(y_test, all_preds, target_names=classes))

    plot_training_history(history, "Iris Multi-Class Classifier", "iris_training.png")

    torch.save(model.state_dict(), "iris_classifier.pth")
    print(" Saved: iris_classifier.pth")

    return model, classes, scaler

#--------------------------------------------------------------
# SECTION 11: Task 3 - Regression
# Concepts: Synthetic house-price-like data
#--------------------------------------------------------------

def task_regression():
    print("\n ===== Task 3: Regression with Neural Network ============")

    # Generate regression data
    X, y = make_regression(n_samples=2000, n_features=15,
                           n_informative=12, noise=20, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Normalize target
    y_mean, y_std = y.mean(), y.std()
    y_norm = ((y - y_mean) / y_std).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_norm, test_size=0.3,
                                                        random_state=42)
    
    train_ds    = TabularDataset(X_train, y_train)
    val_ds      = TabularDataset(X_val, y_val)
    test_ds     = TabularDataset(X_test, y_test)

    train_loader  = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader    = DataLoader(val_ds, batch_size=64)
    test_loader   = DataLoader(test_ds, batch_size=64)

    model       = RegressionNet(15, [128, 64, 32], dropout_rate=0.2).to(DEVUCE)
    optimizer   = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion   = nn.MSELoss()
    scheduler    = ReduceLROnPlateau(optimizer, patience=0, factor=0.5)

    history     = full_training(
        model, train_loader, val_loader,
        optimizer, criterion, scheduler,
        epochs=80, task_name="Regression Net"
    )

    # Test: compute R2 manually
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            out = model(X_b, to(DEVICE))
            all_preds.extend(out.squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    ss_res    = np.sum((y_test - all_preds) ** 2)
    ss_tot    = np.sum((y_test - y_test.mean()) ** 2)
    r2        = 1 - ss_res / ss_tot
    print(f"\n Test R² : {r2:.4f}")

    plot_training_history(history, "Regression Neural Network", "regression_trainning.png")

    torch.save(model.sate.dict(), "regression_net.pth")
    print(" Saved: regression_net.pth")
 

#--------------------------------------------------------------
# SECTION 12: Model Inspector
# Concepts: Inspect weights, count parameters,
#           understand what the model learned
#--------------------------------------------------------------

def inspect_model(model):
    """Print detailed model architecture and parameter info. """
    print("\n ===== Model Inspector ====================")
    print(f"\n Architecture:\n{model}\n")

    total = 0
    print(f" {'Layer':<35} {'Shape':<20} {'Params':>8}")
    print(" " + "-" * 65)
    for name, param in model.named_parameters():
        count  = param.numel()           # total elements in this tensor
        total += count
        print(f" {name:<35} {str(list(param.shape)):<20} {count:>8,}")
    
    print(f"\n Total Parameters : {total:,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Trainable Params : {trainable:,}")

#--------------------------------------------------------------
# SECTION 13: Predict on Custom Input
# Concepts:
#--------------------------------------------------------------

def predict_iris(model, scaler, classes):
    """Predict Iris species for custom flower measurements."""
    print("\n Iris Species Predictor")
    print(" Enter flower measurements: \n")

    try:
        sl = float(input(" Sepal length (cm, e.g. 5.1): "))
        sw = float(input(" Sepal width (cm, e.g. 3.5): "))
        pl = float(input(" Petal length (cm, e.g. 1.4): "))
        pw = float(input(" Petal width (cm, e.g. 0.2): "))

        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs  = F.softmax(logits, dim=1).squeeze()
            pred   = probs.argmax().item()
        
        print(f"\n  ||=======================================================||")
        print(f"    || Predicted Species: {classes[pred]:<19} || ")
        print(f"    || Confidence :                           || ")
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            bar = " " * int(prob.item() * 20)
            print(f"|| {cls:<12} {prob.item()*100:>5.1f}% {bar:<20} || ")
        print(f"\n  ||=======================================================||")

    except Exception as e:
        print(f" Error: {e}")

#--------------------------------------------------------------
# SECTION 14: Main Menu
# Concepts:
#--------------------------------------------------------------

iris_model_ref      = None
iris_scaler_ref     = None
iris_classes_ref    = None

def main():
    global iris_model_ref, iris_scaler_ref, iris_classes_ref

    print("\n" + "="* 54)
    print(" NEURAL NETWORK FROM SCRATCH - PyTorch ")
    print("="*54)

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Tensor Basics & Autograd
              [2]  Plot Activation Functions
              [3]  Manual Neural Net (XOR - no, nn module)
              [4]  Task 1: Binary Classification
              [5]  Task 2: Multi-Class (Iris)
              [6]  Task 3: Regression
              [7]  Inspect a Model
              [8]  Predict Iris Species (after Task 2)
              [9]  Run All Tasks
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        try:
            if choice == "1":
                tensor_basics()

            elif choice == "2":
                plot_activations()

            elif choice == "3":
                losses = manual_neural_network()
                # Plot XOR loss curve
                fig, ax = plt.subplot(figsize=(8, 4))
                fig.patch.set_facecolor("#0f0f0f")
                ax.set_facecolor("#1a1a2e")
                ax.plot(losses, color="#e94560", linewidth=1.5)
                ax.set_title("Manual XOR Network -- Loss Curve", color="white")
                ax.set_xlabel("Epoch", color="#aaa")
                ax.set_ylabel("BCE Loss", color="#aaa")
                ax.tick_params(colors="#aaa")
                ax.grid(alpha=0.3, color="#444")
                plt.tight_layout()
                plt.savefig("xor_loss.png", dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
                print(" Saved: xor_loss.png")
                plt.show(); plt.close()

            elif choice == "4":
                model = task_binary_classification()
                inspect_model(model)

            elif choice == "5":
                iris_model_ref, iris_classes_ref, iris_scaler_ref = task_multiclass_classification()
                inspect_model(model)
            
            elif choice == "6":
                task_regression()
            
            elif choice == "7":
                print(" Build a model first (option 4, 5, or 6).")
            
            elif choice == "8":
                if iris_model_ref is None:
                    print(" Run Task 2 (option 5) first.")
                else:
                    predict_iris(iris_model_ref, iris_scaler_ref, iris_classes_ref)
            
            elif choice == "9":
                print("\n Running All Tasks...\n")
                tensor_basics()
                plot_activations()
                manual_neural_network()
                task_binary_classification()
                iris_model_ref, iris_classes_ref, iris_scaler_ref = task_multiclass_classification()
                task_regression()
                print("\n All Tasks completed Running...! \n")

            elif choice == "0":
                print("\n GoodBye!!!\m")
                break

            else:
                print(" Invalid Choice, try again!")
        
        except Exception as e:
            print(f" Error: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()