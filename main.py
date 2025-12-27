from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import subprocess
import os
import pandas as pd

app = FastAPI(title="Poppy Universe ML Service")
security = HTTPBearer()

# --- CONFIGURATION (Ensure paths match GitHub exactly) ---
CONFIG = {
    2: {
        "script": "Models/Layer_2/Scripts/Trend_Model.py",
        "output": "Models/Layer_2/Output_Data/Layer_2_Top_Trending_Per_Type.csv"
    },
    3: {
        "script": "Models/Layer_3/Scripts/Layer_3_Master_File.py",
        "output": "Models/Layer_3/Output_Data/Layer_3_Final_Predictions.csv"
    },
    4: {
        "script": "Models/Layer_4/Scripts/Master_Layer4.py",
        "output": "Models/Layer_4/Output_Data/Layer4_Final_Predictions.csv"
    }
}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != os.getenv("HF_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid Token")
    return credentials.credentials

@app.post("/run-layer/{layer_num}")
async def run_layer(layer_num: int, request: Request, token: str = Depends(verify_token)):
    if layer_num not in CONFIG:
        raise HTTPException(status_code=404, detail="Layer not found")
    
    body = await request.json()
    mode = body.get("mode", "cached") # 'run' or 'cached'
    layer = CONFIG[layer_num]
    
    # 1. RUN MODEL (Only if Node.js said 'run')
    if mode == "run":
        script_path = os.path.join(os.path.dirname(__file__), layer["script"])
        # We pass DATA_SOURCE=database so the Python script knows to hit Aiven
        result = subprocess.run(["python", script_path], env={**os.environ, "DATA_SOURCE": "database"}, capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "error", "trace": result.stderr}

    # 2. READ OUTPUT FILE (Always from Huggy's storage)
    csv_path = os.path.join(os.path.dirname(__file__), layer["output"])
    if not os.path.exists(csv_path):
        return {"status": "error", "message": "CSV output not found on cloud"}

    df = pd.read_csv(csv_path)
    # Convert CSV to list of objects (JSON format)
    data_json = df.to_dict(orient='records')

    return {
        "status": "success",
        "total_rows": len(df),
        "data": data_json
    }