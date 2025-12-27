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
    print(f"Star NN: Reading real data from {input_path}")
    interactions = pd.read_csv(input_path)
else:
    # Standalone/Fictional fallback
    input_path = os.path.join(os.path.dirname(__file__), "../../../Input_Data/NN_Semantic_Interactions.csv")
    print(f"Star NN: Reading fictional data from {input_path}")
    interactions = pd.read_csv(input_path)

# Ensure Timestamp is datetime
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'], errors='coerce')

# 2) Filter Star Data
star_interactions = interactions[interactions['Category_Type'] == 'Star'].copy()

# Safety Check: If no star data, exit with dummy list
if star_interactions.empty:
    print("No Star interactions found. Creating fallback CSV.")
    u_ids = interactions['User_ID'].unique()
    dummy_df = pd.DataFrame({'User_ID': u_ids})
    output_file = os.path.join(os.path.dirname(__file__), '../Files/Layer4_Star_Predictions.csv')
    dummy_df.to_csv(output_file, index=False)
    sys.exit(0)

# 3) Encode Users and Stars
user_encoder = LabelEncoder()
star_encoder = LabelEncoder()

star_interactions['user_encoded'] = user_encoder.fit_transform(star_interactions['User_ID'])
star_interactions['star_encoded'] = star_encoder.fit_transform(star_interactions['Category_Value'])

num_users = star_interactions['user_encoded'].nunique()
num_star = star_interactions['star_encoded'].nunique()

# 4) Prepare Training Data
X_user = star_interactions['user_encoded'].values
X_star = star_interactions['star_encoded'].values
y = star_interactions['Strength'].values

scaler = MinMaxScaler()
y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_user_train, X_user_val, X_star_train, X_star_val, y_train, y_val = train_test_split(
    X_user, X_star, y_normalized, test_size=0.2, random_state=42
)

# 5) Neural Network Helper Functions (Optimized)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)

def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

def forward_propagation(X, parameters, activation='relu'):
    caches = []; A = X; L = len(parameters) // 2
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
    for l in range(1, (len(parameters) // 2) + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'b{l}']
    return parameters

def create_one_hot(indices, num_classes):
    one_hot = np.zeros((num_classes, len(indices)))
    one_hot[indices, np.arange(len(indices))] = 1
    return one_hot

# 6) Train Prep
X_train = np.vstack([create_one_hot(X_user_train, num_users), create_one_hot(X_star_train, num_star)])
Y_train = y_train.reshape(1, -1)

# 7) Train
layer_dims = [num_users + num_star, 64, 32, 16, 1]
parameters = initialize_parameters(layer_dims)
learning_rate, num_epochs = 0.01, 500

for epoch in range(num_epochs):
    AL_train, caches = forward_propagation(X_train, parameters, activation='relu')
    grads = backward_propagation(AL_train, Y_train, caches, parameters, activation='relu')
    parameters = update_parameters(parameters, grads, learning_rate)

# 8) Predict Full Matrix
all_users, all_star = np.arange(num_users), np.arange(num_star)
user_grid, star_grid = np.meshgrid(all_users, all_star, indexing='ij')
X_all = np.vstack([create_one_hot(user_grid.flatten(), num_users), create_one_hot(star_grid.flatten(), num_star)])

predictions_norm, _ = forward_propagation(X_all, parameters, activation='relu')
predictions = scaler.inverse_transform(predictions_norm.T).flatten()
predictions = np.clip(predictions, 1, 5)

# 9) Save
prediction_matrix = predictions.reshape(num_users, num_star)
R_hat_df = pd.DataFrame(prediction_matrix, columns=star_encoder.inverse_transform(all_star))
R_hat_df.insert(0, 'User_ID', user_encoder.inverse_transform(all_users))

output_file = os.path.join(os.path.dirname(__file__), '../Files/Layer4_Star_Predictions.csv')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
R_hat_df.to_csv(output_file, index=False)

print(f"Star NN predictions saved: {output_file}")