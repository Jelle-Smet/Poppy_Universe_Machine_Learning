import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# 1) Load Data Logic
DATA_SOURCE = os.getenv("DATA_SOURCE", "fictional")

if DATA_SOURCE == "database":
    # Path to temp file created by Master_Layer4.py
    input_path = os.path.join(os.path.dirname(__file__), '../../Files/temp_interactions_L4.csv')
    print(f"Moon NN: Reading real data from {input_path}")
    interactions = pd.read_csv(input_path)
else:
    # Standalone/Fictional fallback
    input_path = os.path.join(os.path.dirname(__file__), "../../../Input_Data/NN_Semantic_Interactions.csv")
    print(f"Moon NN: Reading fictional data from {input_path}")
    interactions = pd.read_csv(input_path)

# Ensure Timestamp is datetime
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'], errors='coerce')

# 2) Filter Moon Parent Data
moon_interactions = interactions[interactions['Category_Type'] == 'Moon'].copy()

# Safety Check: If no moon interactions, exit early with dummy list for the master merge
if moon_interactions.empty:
    print("No Moon interactions found. Skipping NN training.")
    u_ids = interactions['User_ID'].unique()
    dummy_df = pd.DataFrame({'User_ID': u_ids})
    output_file = os.path.join(os.path.dirname(__file__), '../Files/Layer4_Moon_Predictions.csv')
    dummy_df.to_csv(output_file, index=False)
    sys.exit(0)

# 3) Encode Users and Moon Parents
user_encoder = LabelEncoder()
moon_encoder = LabelEncoder()

moon_interactions['user_encoded'] = user_encoder.fit_transform(moon_interactions['User_ID'])
moon_interactions['moon_encoded'] = moon_encoder.fit_transform(moon_interactions['Category_Value'])

num_users = moon_interactions['user_encoded'].nunique()
num_moons = moon_interactions['moon_encoded'].nunique()

# 4) Prepare Training Data
X_user = moon_interactions['user_encoded'].values
X_moon = moon_interactions['moon_encoded'].values
y = moon_interactions['Strength'].values

scaler = MinMaxScaler()
y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split
X_user_train, X_user_val, X_moon_train, X_moon_val, y_train, y_val = train_test_split(
    X_user, X_moon, y_normalized, test_size=0.2, random_state=42
)

# --- NN HELPER FUNCTIONS (Kept your logic intact) ---
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

def forward_propagation(X, parameters, activation='relu'):
    caches = []; A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        Z = parameters[f'W{l}'] @ A_prev + parameters[f'b{l}']
        A = relu(Z) if activation == 'relu' else sigmoid(Z)
        caches.append((Z, A_prev))
    A_prev = A
    Z = parameters[f'W{L}'] @ A_prev + parameters[f'b{L}']
    AL = sigmoid(Z)
    caches.append((Z, A_prev))
    return AL, caches

def backward_propagation(AL, Y, caches, parameters, activation='relu'):
    grads = {}; L = len(caches); m = AL.shape[1]
    dZ = AL - Y
    grads[f'dW{L}'] = (1/m) * dZ @ caches[L-1][1].T
    grads[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    for l in reversed(range(1, L)):
        Z, A_prev = caches[l-1]
        dA = parameters[f'W{l+1}'].T @ dZ
        dZ = dA * relu_derivative(Z) if activation == 'relu' else dA * sigmoid_derivative(Z)
        grads[f'dW{l}'] = (1/m) * dZ @ A_prev.T
        grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'b{l}']
    return parameters

def create_one_hot(indices, num_classes):
    one_hot = np.zeros((num_classes, len(indices)))
    one_hot[indices, np.arange(len(indices))] = 1
    return one_hot

# 5) One-Hot Prep
X_train = np.vstack([create_one_hot(X_user_train, num_users), create_one_hot(X_moon_train, num_moons)])
X_val = np.vstack([create_one_hot(X_user_val, num_users), create_one_hot(X_moon_val, num_moons)])
Y_train = y_train.reshape(1, -1)
Y_val = y_val.reshape(1, -1)

# 6) Train
layer_dims = [num_users + num_moons, 64, 32, 16, 1]
parameters = initialize_parameters(layer_dims)
learning_rate, num_epochs = 0.01, 500 # Reduced epochs for faster backend response

for epoch in range(num_epochs):
    AL_train, caches = forward_propagation(X_train, parameters, activation='relu')
    grads = backward_propagation(AL_train, Y_train, caches, parameters, activation='relu')
    parameters = update_parameters(parameters, grads, learning_rate)

# 7) Prediction Matrix
all_users = np.arange(num_users)
all_moons = np.arange(num_moons)
user_grid, moon_grid = np.meshgrid(all_users, all_moons, indexing='ij')
user_flat, moon_flat = user_grid.flatten(), moon_grid.flatten()

X_all = np.vstack([create_one_hot(user_flat, num_users), create_one_hot(moon_flat, num_moons)])
predictions_normalized, _ = forward_propagation(X_all, parameters, activation='relu')
predictions = scaler.inverse_transform(predictions_normalized.T).flatten()
predictions = np.clip(predictions, 1, 5)

# 8) Convert to DataFrame
prediction_matrix = predictions.reshape(num_users, num_moons)
user_ids = user_encoder.inverse_transform(all_users)
moon_values_decoded = moon_encoder.inverse_transform(all_moons)

R_hat_df = pd.DataFrame(prediction_matrix, columns=moon_values_decoded)
R_hat_df.insert(0, 'User_ID', user_ids)

# 9) Save Output
output_file = os.path.join(os.path.dirname(__file__), '../Files/Layer4_Moon_Predictions.csv')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
R_hat_df.to_csv(output_file, index=False)

print(f"Moon NN Complete. Saved: {output_file}")