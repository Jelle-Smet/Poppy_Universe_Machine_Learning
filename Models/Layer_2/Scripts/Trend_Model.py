import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

print("=== Layer 2 Trend Model started ===")

# 1. Configuration
DATA_SOURCE = os.getenv("DATA_SOURCE", "fictional")

# 2. Load Data
if DATA_SOURCE == "database":
    print("Reading data from Node.js (stdin)...")
    try:
        # Read the JSON string sent from Node
        input_data = sys.stdin.read()
        if not input_data:
            print("No data received via stdin!")
            sys.exit(1)
        
        # Load JSON into DataFrame
        raw_json = json.loads(input_data)
        interactions = pd.DataFrame(raw_json)
    except Exception as e:
        print(f"Error parsing JSON from Node: {e}")
        sys.exit(1)
else:
    print("Reading backup fictional CSV...")
    data_path = os.path.join(os.path.dirname(__file__), "../../../Input_data/Simulated_User_Interactions.csv")
    interactions = pd.read_csv(data_path)

# ---------------- PREP ----------------
interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'], errors='coerce')
interactions = interactions.dropna(subset=['Timestamp'])

recent_window = datetime.now() - timedelta(days=14)
recent_interactions = interactions[interactions['Timestamp'] >= recent_window].copy()

if recent_interactions.empty:
    recent_interactions = interactions.copy()

# ---------------- FEATURES ----------------
max_time = recent_interactions['Timestamp'].max()
min_time = recent_interactions['Timestamp'].min()

if max_time == min_time:
    recent_interactions['decay'] = 1.0
else:
    recent_interactions['decay'] = (
        (recent_interactions['Timestamp'] - min_time) / (max_time - min_time)
    ).fillna(1.0)

features = recent_interactions.groupby(['Object_Type', 'Object_ID']).apply(
    lambda x: pd.Series({
        'total_interactions': len(x),
        'num_views': x.loc[x['Interaction_Type'] == 'View', 'decay'].sum(),
        'num_likes': x.loc[x['Interaction_Type'] == 'Like', 'decay'].sum(),
        'num_rates': x.loc[x['Interaction_Type'] == 'Rate', 'decay'].sum(),
    })
).reset_index()

features['trending_score'] = (
    features['num_views'] * 1 +
    features['num_likes'] * 2 +
    features['num_rates'] * 3
).round(3)

# ---------------- OUTPUT ----------------
top_per_type = features.sort_values('trending_score', ascending=False).groupby('Object_Type').head(100)

output_path = os.path.join(os.path.dirname(__file__), "../../../Output_data/Layer_2_Top_Trending_Per_Type.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
top_per_type.to_csv(output_path, index=False)

print(f"CSV overwritten successfully at: {output_path}")
print("=== Layer 2 Trend Model finished ===")