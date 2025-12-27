import pandas as pd
import numpy as np
import os
import subprocess
import sys
import json

def log(msg):
    print(msg, flush=True)

log("=== Layer 3 Master Model started ===")

DATA_SOURCE = os.getenv("DATA_SOURCE", "fictional")
input_temp_path = os.path.join(os.path.dirname(__file__), '../../Files/temp_interactions.csv') # Temp file for sub-scripts to read

# 1. Load Data
if DATA_SOURCE == "database":
    log("Reading data from Node.js (stdin)...")
    try:
        input_data = sys.stdin.read()
        raw_json = json.loads(input_data)
        interactions_df = pd.DataFrame(raw_json)
        # Save to a temp CSV so Star/Planet/Moon scripts can read it
        os.makedirs(os.path.dirname(input_temp_path), exist_ok=True)
        interactions_df.to_csv(input_temp_path, index=False)
        log(f"TOTAL_ROWS_PROCESSED: {len(interactions_df)}")
    except Exception as e:
        log(f"Error: {e}")
        sys.exit(1)
else:
    log("Using fictional fallback logic.")
    # In fictional mode, we assume the sub-scripts use their own internal fallbacks
    log("TOTAL_ROWS_PROCESSED: 0") # Indicating fallback

# --- HELPER TO RUN SUB-SCRIPTS ---
def run_python_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if os.path.exists(script_path):
        log(f"Executing {script_name}...")
        # We pass DATA_SOURCE so sub-scripts know whether to look for the temp CSV
        subprocess.check_call([sys.executable, script_path], env={**os.environ, "DATA_SOURCE": DATA_SOURCE})
    else:
        log(f"Error: {script_name} not found.")

# 2. Run the 3 sub-models
run_python_script('Star_Matrix_Model.py')
run_python_script('Planet_Matrix_Model.py')
run_python_script('Moon_Matrix_Model.py')

# 3. Load & Merge results
try:
    # Adjusting paths to match your provided structure
    sim_star_path   = os.path.join(os.path.dirname(__file__), '../Files/Layer3_Star_Predictions.csv')
    sim_planet_path = os.path.join(os.path.dirname(__file__), '../Files/Layer3_Planet_Predictions.csv')
    sim_moon_path   = os.path.join(os.path.dirname(__file__), '../Files/Layer3_Moon_Predictions.csv')

    star_df   = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), sim_star_path)))
    planet_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), sim_planet_path)))
    moon_df   = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), sim_moon_path)))

    # Concatenate horizontally on User_ID
    merged_df = star_df.merge(planet_df, on='User_ID').merge(moon_df, on='User_ID')

    # Save final merged matrix
    output_path = os.path.join(os.path.dirname(__file__), '../../../Output_Data/Layer_3_Final_Predictions.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    log(f"Final merged matrix saved: {output_path}")
except Exception as e:
    log(f"Merging Error: {e}")
    sys.exit(1)

log("=== Layer 3 Master Model finished ===")