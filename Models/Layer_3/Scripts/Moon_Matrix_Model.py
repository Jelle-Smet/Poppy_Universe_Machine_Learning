import pandas as pd
import numpy as np
import os
import sys

# 1) Load Data Logic
DATA_SOURCE = os.getenv("DATA_SOURCE", "fictional")

if DATA_SOURCE == "database":
    # Path to temp file created by Master_Layer3.py
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Files/temp_interactions.csv'))
    print(f"Moon Model: Reading real data from {input_path}")
    interactions = pd.read_csv(input_path)
else:
    # Standalone/Fictional fallback
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Input_Data/MF_Semantic_Type_Interactions.csv"))
    print(f"Moon Model: Reading fictional data from {input_path}")
    interactions = pd.read_csv(input_path)

# Ensure Timestamp is datetime
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'], errors='coerce')

# 2) Filter for Moon data
moon_interactions = interactions[interactions['Category_Type'] == 'Moon'].copy()

# Safety Check: If no moon interactions, exit early with dummy list
if moon_interactions.empty:
    print("No Moon interactions found. Skipping MF calculations.")
    u_ids = interactions['User_ID'].unique()
    dummy_df = pd.DataFrame({'User_ID': u_ids})
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Files/Layer3_Moon_Predictions.csv'))
    dummy_df.to_csv(output_path, index=False)
    sys.exit(0)

# 3) Create User × Category Matrix
user_category_matrix = moon_interactions.pivot_table(
    index='User_ID', 
    columns='Category_Value', 
    values='Strength', 
    aggfunc='max',
    fill_value=0
)

# 4) Matrix Factorization with SGD
R = user_category_matrix.values
num_users, num_items = R.shape
K = 3  # Latent features

np.random.seed(42)
U = np.random.rand(num_users, K)
V = np.random.rand(num_items, K)

alpha = 0.01   # learning rate
beta = 0.02    # regularization term
iterations = 500 # Optimized for speed

for it in range(iterations):
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                pred = U[i, :].dot(V[j, :].T)
                e_ij = R[i, j] - pred
                U[i, :] += alpha * (2 * e_ij * V[j, :] - beta * U[i, :])
                V[j, :] += alpha * (2 * e_ij * U[i, :] - beta * V[j, :])

R_hat = U.dot(V.T)

# 5) Convert Approximated Matrix Back to DataFrame
R_hat_df = pd.DataFrame(R_hat, columns=user_category_matrix.columns)
R_hat_df['User_ID'] = user_category_matrix.index

cols = ['User_ID'] + [c for c in R_hat_df.columns if c != 'User_ID']
R_hat_df = R_hat_df[cols]

# 6) Save Predicted Matrix to CSV
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Files/Layer3_Moon_Predictions.csv'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
R_hat_df.to_csv(output_path, index=False)

print(f"Moon matrix factorization complete. Saved: {output_path}")