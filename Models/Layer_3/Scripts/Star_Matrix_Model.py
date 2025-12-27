import pandas as pd
import numpy as np
import os
import sys

# Removed matplotlib/seaborn to keep it fast for backend execution

# 1) Load Data Logic
DATA_SOURCE = os.getenv("DATA_SOURCE", "fictional")

if DATA_SOURCE == "database":
    # Read the temp file created by the Master script
    # Path is relative to the Scripts folder where this runs
    input_path = os.path.join(os.path.dirname(__file__), '../../Files/temp_interactions.csv')
    print(f"Star Model: Reading real data from {input_path}")
    interactions = pd.read_csv(input_path)
else:
    # Fallback to the original simulated CSV
    input_path = os.path.join(os.path.dirname(__file__), "../../../Input_Data/MF_Semantic_Type_Interactions.csv")
    print(f"Star Model: Reading fictional data from {input_path}")
    interactions = pd.read_csv(input_path)

# Ensure Timestamp is datetime
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'], errors='coerce')

# 2) Filter for Star Data
# Note: Your query provides 'Category_Type', so we filter for 'Star'
star_interactions = interactions[interactions['Category_Type'] == 'Star'].copy()

if star_interactions.empty:
    print("No Star interactions found. Creating empty fallback predictions.")
    # Create a dummy DataFrame so the Master script merge doesn't crash
    # Assuming User_ID exists in the original interactions
    u_ids = interactions['User_ID'].unique()
    dummy_df = pd.DataFrame({'User_ID': u_ids})
    output_path = os.path.join(os.path.dirname(__file__), '../Files/Layer3_Moon_Predictions.csv')
    dummy_df.to_csv(output_path, index=False)
    sys.exit(0)

# 3) Create User × Category Matrix
user_category_matrix = star_interactions.pivot_table(
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

alpha = 0.01   # Learning rate
beta = 0.02    # Regularization
iterations = 500 # Slightly reduced for faster backend response

for it in range(iterations):
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                pred = U[i, :].dot(V[j, :].T)
                e_ij = R[i, j] - pred
                U[i, :] += alpha * (2 * e_ij * V[j, :] - beta * U[i, :])
                V[j, :] += alpha * (2 * e_ij * U[i, :] - beta * V[j, :])

R_hat = U.dot(V.T)

# 5) Convert Back to DataFrame
R_hat_df = pd.DataFrame(R_hat, columns=user_category_matrix.columns)
R_hat_df['User_ID'] = user_category_matrix.index

cols = ['User_ID'] + [c for c in R_hat_df.columns if c != 'User_ID']
R_hat_df = R_hat_df[cols]

# 6) Save Predicted Matrix
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Files/Layer3_Star_Predictions.csv'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
R_hat_df.to_csv(output_path, index=False)

print(f"Star matrix complete. Saved: {output_path}")