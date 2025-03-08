import pandas as pd
import mlcroissant as mlc
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Downloading and Cleaning Data
matplotlib.use('TkAgg')
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/asinow/laptop-price-dataset/croissant/download')
df = pd.DataFrame(croissant_dataset.records(record_set='laptop_prices.csv'))

df.columns = [col.split('/')[-1].replace('+', ' ') for col in df.columns]
string_cols = df.select_dtypes(include=['object']).columns
for col in string_cols:
    df[col] = df[col].str.decode('utf-8')

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[string_cols])
feature_names = encoder.get_feature_names_out(string_cols)
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
num_df = df.select_dtypes(exclude=['object'])
final_df = pd.concat([num_df, encoded_df], axis=1)

# Splitting Data into train/test, then scaling and setting tensors
Y = final_df['Price (%24)']
X = final_df.drop('Price (%24)', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42
)

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
price_scaler = StandardScaler()
Y_train_scaled = price_scaler.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_scaled = price_scaler.transform(Y_test.values.reshape(-1, 1))

X_train_tensor = torch.FloatTensor(X_train_scaled)
Y_train_tensor = torch.FloatTensor(Y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
Y_test_tensor = torch.FloatTensor(Y_test_scaled)

# Neural Network
class LaptopPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(LaptopPricePredictor, self).__init__()
        # First complex neural network I built!
        # Works by using a series of linear layers with batch normalization, ReLU activation, and dropout
        # The linear layers use a funnel design to narrow input features before making a prediction to improve performance
        # batch normalization improves speed and stability
        # ReLU activation introduces non-linearity so that the model can develop relationships between different features
        # dropout prevents overfitting by randomly setting some inputs to 0, creating randomness
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.27),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.23),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        return self.network(x)

# Initializing Model
model = LaptopPricePredictor(X_train_tensor.shape[1])
# HuberLoss chosen since it handles outliers better than MSE
loss_function = nn.HuberLoss(delta=1.0)
# Chosen for adaptive learning rates for different params to prioritize things that would contribute more to price
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# Reduces learning rate when plateau is reached
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.3,
    patience=7,
    threshold=0.005,
    min_lr=1e-6
)
# Training parameters
batch_size = 16
epochs = 50
patience = 10
best_loss = float('inf')
no_improve = 0

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    n_samples = len(X_train_tensor)
    
    indices = torch.randperm(n_samples)
    X_train_shuffled = X_train_tensor[indices]
    Y_train_shuffled = Y_train_tensor[indices]
    
    for i in range(0, n_samples, batch_size):
        batch_X = X_train_shuffled[i:i+batch_size]
        batch_y = Y_train_shuffled[i:i+batch_size]
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = loss_function(predictions, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        epoch_loss += loss.item() * len(batch_X)
    
    epoch_loss /= n_samples
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor)
        val_loss = loss_function(val_predictions, Y_test_tensor)
        
        val_pred_orig = price_scaler.inverse_transform(val_predictions.numpy())
        val_true_orig = price_scaler.inverse_transform(Y_test_tensor.numpy())
        val_mae = mean_absolute_error(val_true_orig, val_pred_orig)
        
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: ${val_mae:.2f}')
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered")
            break

# Evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    Y_pred = model(X_test_tensor)

    Y_pred_orig = price_scaler.inverse_transform(Y_pred.numpy())
    Y_test_orig = price_scaler.inverse_transform(Y_test_tensor.numpy())
    
    mse = mean_squared_error(Y_test_orig, Y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
    r2 = r2_score(Y_test_orig, Y_pred_orig)
    
    print("\nModel Evaluation Metrics:")
    print(f"Test Loss (MSE): {best_loss:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R² Score: {r2:.3f}")

# Visualization
Y_test_np = Y_test_orig
Y_pred_np = Y_pred_orig

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(Y_test_np, Y_pred_np, alpha=0.5)
ax1.plot([Y_test_np.min(), Y_test_np.max()], 
         [Y_test_np.min(), Y_test_np.max()], 
         'r--', label='Perfect Prediction')
ax1.set_xlabel('Actual Price ($)')
ax1.set_ylabel('Predicted Price ($)')
ax1.set_title('Model Prediction Accuracy')
ax1.legend()

errors = Y_pred_np - Y_test_np
ax2.hist(errors.ravel(), bins=20, edgecolor='black')
ax2.axvline(x=0, color='r', linestyle='--', label='Zero Error')
ax2.set_xlabel('Prediction Error ($)')
ax2.set_ylabel('Count')
ax2.set_title(f'Error Distribution (Mean: ${errors.mean():.2f})')
ax2.legend()

plt.tight_layout()
plt.show()

print("\nError Statistics:")
print("═" * 50)
print(f"Mean Error:         ${errors.mean():>10.2f}")
print(f"Std Dev of Error:   ${errors.std():>10.2f}")
print(f"Max Underestimate:  ${errors.min():>10.2f}")
print(f"Max Overestimate:   ${errors.max():>10.2f}")
print("═" * 50)

results_df = pd.DataFrame({
    'Actual Price': Y_test_np.flatten(),
    'Predicted Price': Y_pred_np.flatten(),
    'Error': errors.flatten()
})

results_df = pd.concat([results_df, X_test.reset_index(drop=True)], axis=1)

# Filter large errors (abs > 500)
large_errors = results_df[abs(results_df['Error']) > 500].sort_values('Error')

print(f"\nTotal number of large errors: {len(large_errors)} out of {len(results_df)} samples")
print(f"Percentage of large errors: {(len(large_errors)/len(results_df))*100:.1f}%")
