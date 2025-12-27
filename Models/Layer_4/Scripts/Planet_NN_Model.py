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
    print(f"Planet NN: Reading real data from {input_path}")
    interactions = pd.read_csv(input_path)
else:
    # Standalone/Fictional fallback
    input_path = os.path.join(os.path.dirname(__file__), "../../../Input_Data/NN_Semantic_Interactions.csv")
    print(f"Planet NN: Reading fictional data from {input_path}")
    interactions = pd.read_csv(input_path)

# Ensure Timestamp is datetime
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'], errors='coerce')

# 2) Filter Planet Data
planet_interactions = interactions[interactions['Category_Type'] == 'Planet'].copy()

# Safety Check: If no planet data in this batch, exit with a dummy User_ID list
if planet_interactions.empty:
    print("No Planet interactions found. Skipping NN training.")
    u_ids = interactions['User_ID'].unique()
    dummy_df = pd.DataFrame({'User_ID': u_ids})
    output_file = os.path.join(os.path.dirname(__file__), '../Files/Layer4_Planet_Predictions.csv')
    dummy_df.to_csv(output_file, index=False)
    sys.exit(0)

# 3) Encode Users and Planets
user_encoder = LabelEncoder()
planet_encoder = LabelEncoder()

planet_interactions['user_encoded'] = user_encoder.fit_transform(planet_interactions['User_ID'])
planet_interactions['planet_encoded'] = planet_encoder.fit_transform(planet_interactions['Category_Value'])

num_users = planet_interactions['user_encoded'].nunique()
num_planets = planet_interactions['planet_encoded'].nunique()

# 4) Prepare Training Data
X_user = planet_interactions['user_encoded'].values
X_planet = planet_interactions['planet_encoded'].values
y = planet_interactions['Strength'].values

scaler = MinMaxScaler()
y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_user_train, X_user_val, X_planet_train, X_planet_val, y_train, y_val = train_test_split(
    X_user, X_planet, y_normalized, test_size=0.2, random_state=42
)

# 5) Neural Network Helper Functions (Manual implementation)

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

# 6) Prepare Features
X_train = np.vstack([create_one_hot(X_user_train, num_users), create_one_hot(X_planet_train, num_planets)])
X_val = np.vstack([create_one_hot(X_user_val, num_users), create_one_hot(X_planet_val, num_planets)])
Y_train = y_train.reshape(1, -1)
Y_val = y_val.reshape(1, -1)

# 7) Train Neural Network
layer_dims = [num_users + num_planets, 64, 32, 16, 1]
parameters = initialize_parameters(layer_dims)
learning_rate, num_epochs = 0.01, 500 # Optimized epochs

for epoch in range(num_epochs):
    AL_train, caches = forward_propagation(X_train, parameters, activation='relu')
    grads = backward_propagation(AL_train, Y_train, caches, parameters, activation='relu')
    parameters = update_parameters(parameters, grads, learning_rate)

# 8) Generate Predictions
all_users, all_planets = np.arange(num_users), np.arange(num_planets)
user_grid, planet_grid = np.meshgrid(all_users, all_planets, indexing='ij')
X_all = np.vstack([create_one_hot(user_grid.flatten(), num_users), create_one_hot(planet_grid.flatten(), num_planets)])

predictions_normalized, _ = forward_propagation(X_all, parameters, activation='relu')
predictions = scaler.inverse_transform(predictions_normalized.T).flatten()
predictions = np.clip(predictions, 1, 5)

# 9) Convert to DataFrame
prediction_matrix = predictions.reshape(num_users, num_planets)
R_hat_df = pd.DataFrame(prediction_matrix, columns=planet_encoder.inverse_transform(all_planets))
R_hat_df.insert(0, 'User_ID', user_encoder.inverse_transform(all_users))

# 10) Save Output
output_file = os.path.join(os.path.dirname(__file__), '../Files/Layer4_Planet_Predictions.csv')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
R_hat_df.to_csv(output_file, index=False)

print(f"Planet NN predictions saved. Shape: {R_hat_df.shape}")