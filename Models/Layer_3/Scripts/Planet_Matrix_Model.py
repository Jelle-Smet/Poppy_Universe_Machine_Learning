# %% [markdown]
# # Poppy Universe – Layer 3: Planet Matrix Model
# 
# Welcome to the **Poppy Universe Layer 3 – Planet Matrix notebook**!  
# The planet dataset is already fully correct. Here, we focus on **building a matrix-based recommendation model** using simulated user interactions and planet types. This is a **sandbox environment** to test collaborative filtering before the engine consumes it.
# 
# > Note: This notebook currently uses **simulated user interactions** to test the Planet matrix.  
# > Once we have enough real interactions, the same pipeline will process actual user data for production recommendations.
# 
# ---
# 
# ## Goals
# 
# 1. **Prepare interaction data for matrix factorization**  
#    - Map users to planet types  
#    - Include weighted interactions (views, clicks, favorites, ratings)  
#    - Normalize scores for ML input
# 
# 2. **Build the User × Planet_Type matrix**  
#    - Users in rows, planet types in columns  
#    - Populate with interaction strengths  
# 
# 3. **Perform matrix factorization / prediction**  
#    - Generate predicted scores for each user × planet_type  
#    - Save intermediate CSV for engine integration
# 
# 4. **Analyze results**  
#    - Identify top planet types per user  
#    - Visualize patterns across users and planet types
# 
# ---
# 
# ## Folder & File References
# 
# - **../../Input_Data/Planets.csv** → Planet dataset  
# - **../../Input_Data/Semantic_Type_Interactions.csv** → User interaction dataset  
# - **../../Output_Data/Layer3_Planet_Predictions.csv** → Final predictions for engine  
# - **Plots/** → Optional heatmaps or visualizations
# 
# ---
# 
# > Note: This notebook focuses **on the planet component** of Layer 3. Stars and moons will have separate notebooks, then merged later.
# 

# %% [markdown]
# ## 0) Imports

# %%
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

# %% [markdown]
# ## 1) Load Data

# %%
# --- Load interaction dataset ---
# 'backend_df' is injected via papermill by the master notebook if backend data passed the checks
try:
    interactions = backend_df
    print("Using backend-provided interactions")
except NameError:
    # fallback to CSV if running standalone
    interactions = pd.read_csv("../../../Input_Data/MF_Semantic_Type_Interactions.csv")
    print("Using simulated CSV interactions")

# Ensure Timestamp is datetime
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'])

# Preview
interactions.head()

# %% [markdown]
# **Explanation:**  
# We’re loading the simulated user × type interaction data to see what we have. The key columns are:
#  
# - `Interaction_ID`: unique identifier for each interaction  
# - `User_ID`: the user who performed the interaction  
# - `Category_Type`: the type of category the interaction belongs to (e.g., Star_Type, Planet_Type, Moon_Parent)  
# - `Category_Value`: the specific value within the category (e.g., G for Star_Type, Dwarf Planet for Planet_Type)  
# - `Strength`: numerical interaction strength (1–5), used as a matrix factorization target  
# - `Timestamp`: when the interaction occurred  
#  
# This gives us the base data we’ll use to compute features like user-type preferences, recency-weighted strengths, and the semantic matrices for the third layer of the recommendation engine.

# %% [markdown]
# ## 2) Filter out Star and moon data

# %%
# Keep only rows where Category_Type is "Planet"
planet_interactions = interactions[interactions['Category_Type'] == 'Planet']

planet_interactions.head()

# %% [markdown]
# ## 3) Create User × Category Matrix

# %%
# Pivot: rows = users, cols = category values, values = max strength (or sum/mean if multiple)
user_category_matrix = planet_interactions.pivot_table(
    index='User_ID', 
    columns='Category_Value', 
    values='Strength', 
    aggfunc='max',   # could also be sum or mean
    fill_value=0     # fills missing interactions with 0
)

# Optional: reset column names if you want a flat DataFrame
user_category_matrix = user_category_matrix.reset_index()

print(user_category_matrix.head())


# %% [markdown]
# ## 4) Matrix Factorization with SGD

# %%
# Only run this check if you want
run_K_check = True  # set to False to skip

if run_K_check:
    R = user_category_matrix.drop('User_ID', axis=1).values
    num_users, num_items = R.shape
    
    K_values = [ 3 , 4, 5]  # candidate latent features
    alpha = 0.01
    beta = 0.02
    iterations = 1500  # shorter for quick test

    final_sse = []

    for K in K_values:
        np.random.seed(42)
        U = np.random.rand(num_users, K)
        V = np.random.rand(num_items, K)
        sse = 0

        for it in range(iterations):
            total_error = 0
            for i in range(num_users):
                for j in range(num_items):
                    if R[i, j] > 0:
                        pred = U[i, :].dot(V[j, :].T)
                        e_ij = R[i, j] - pred
                        total_error += e_ij**2
                        U[i, :] += alpha * (2 * e_ij * V[j, :] - beta * U[i, :])
                        V[j, :] += alpha * (2 * e_ij * U[i, :] - beta * V[j, :])
            total_error += (beta/2) * (np.sum(U**2) + np.sum(V**2))
            sse = total_error  # keep last SSE

        final_sse.append(sse)

    # Plot SSE vs K
    plt.figure()
    plt.plot(K_values, final_sse, marker='o')
    plt.xlabel("K (latent features)")
    plt.ylabel("Final SSE after training")
    plt.title("MF: SSE vs K")
    plt.grid(True)
    plt.show()

# %%
# Convert pivot table to numpy array (exclude User_ID column)
R = user_category_matrix.drop('User_ID', axis=1).values
num_users, num_items = R.shape
K = 4 # check code above 

# Initialize user and item latent matrices
np.random.seed(42)
U = np.random.rand(num_users, K)
V = np.random.rand(num_items, K)

# Hyperparameters
alpha = 0.01
beta = 0.02
iterations = 2000

# Store loss
sse_history = []

# SGD loop
for it in range(iterations):
    total_error = 0

    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                pred = U[i, :].dot(V[j, :].T)
                e_ij = R[i, j] - pred

                total_error += e_ij ** 2

                U[i, :] += alpha * (2 * e_ij * V[j, :] - beta * U[i, :])
                V[j, :] += alpha * (2 * e_ij * U[i, :] - beta * V[j, :])

    sse_history.append(total_error)

# Reconstruct matrix
R_hat = U.dot(V.T)

print("Original matrix:\n", R)
print("Approximated matrix:\n", R_hat)

# %%
start = 200
end = len(sse_history)

plt.figure()
plt.plot(range(start, end), sse_history[start:end])
plt.xlabel("Iteration")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("MF Training Loss (zoomed in)")
plt.grid(True)
plt.show()


# %% [markdown]
# ### SSE in Matrix Factorization
# **What it is:** Sum of Squared Errors (SSE) measures how far the model's predictions are from the actual values. Lower SSE = better fit.
# 
# **How it's calculated:** For each known value, take the difference between the real value and the prediction, square it, and sum all these squared differences.
# 
# **How to interpret it:** Low SSE → model captures patterns well. High SSE → data is noisy, sparse, or too complex for the chosen number of latent features.
# 

# %% [markdown]
# ## 5) Convert Approximated Matrix Back to DataFrame

# %%
# Convert R_hat back to DataFrame
R_hat_df = pd.DataFrame(R_hat, columns=user_category_matrix.columns[1:])  # skip User_ID
R_hat_df['User_ID'] = user_category_matrix['User_ID'].values

# Optional: reorder columns so User_ID is first
cols = ['User_ID'] + [c for c in R_hat_df.columns if c != 'User_ID']
R_hat_df = R_hat_df[cols]

R_hat_df.head()


# %% [markdown]
# ## 6) Save Predicted Matrix to CSV

# %%
# Save as CSV for master notebook
R_hat_df.to_csv('../Files/Layer3_Planet_Predictions.csv', index=False)

# Optional: preview
print(R_hat_df.head())


