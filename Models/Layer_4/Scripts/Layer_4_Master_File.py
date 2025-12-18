import pandas as pd
import numpy as np
import os
import subprocess
import sys
import json
from datetime import datetime

def log(msg):
    print(msg, flush=True)

log("=== Layer 4 Master NN Model started ===")

DATA_SOURCE = os.getenv("DATA_SOURCE", "fictional")
input_temp_path = '../../Files/temp_interactions_L4.csv'

# 1. Handle Input Data
if DATA_SOURCE == "database":
    try:
        input_data = sys.stdin.read()
        raw_json = json.loads(input_data)
        interactions_df = pd.DataFrame(raw_json)
        
        # Save for NN sub-scripts to read
        os.makedirs(os.path.dirname(input_temp_path), exist_ok=True)
        interactions_df.to_csv(input_temp_path, index=False)
        log(f"TOTAL_ROWS_PROCESSED: {len(interactions_df)}")
    except Exception as e:
        log(f"Error loading stdin: {e}")
        sys.exit(1)
else:
    log("Running in fictional mode. TOTAL_ROWS_PROCESSED: 0")

# 2. Helper to run NN scripts
def run_nn_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if os.path.exists(script_path):
        log(f"Executing {script_name}...")
        subprocess.check_call([sys.executable, script_path], env={**os.environ, "DATA_SOURCE": DATA_SOURCE})
    else:
        log(f"Error: {script_name} not found.")

# 3. Trigger NN Sub-models
run_nn_script('Star_NN_Model.py')
run_nn_script('Planet_NN_Model.py')
run_nn_script('Moon_NN_Model.py')

# 4. Load & Merge Predictions
try:
    # Relative paths to the prediction files
    star_df   = pd.read_csv('../Files/Layer4_Star_Predictions.csv')
    planet_df = pd.read_csv('../Files/Layer4_Planet_Predictions.csv')
    moon_df   = pd.read_csv('../Files/Layer4_Moon_Predictions.csv')

    # Use merge to ensure User_IDs stay aligned correctly
    merged_df = star_df.merge(planet_df, on='User_ID', how='outer').merge(moon_df, on='User_ID', how='outer')
    merged_df = merged_df.fillna(0)

    # Save Final Data
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Output_Data/Layer4_Final_Predictions.csv'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    log(f"Layer 4 Integration Complete. Merged file saved to: {output_path}")
except Exception as e:
    log(f"Merge error: {e}")
    sys.exit(1)