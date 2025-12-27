from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import subprocess
import os
import pandas as pd
import uvicorn

app = FastAPI(title="Poppy Universe ML Service")
security = HTTPBearer()

# --- CONFIGURATION (Ensure paths match GitHub exactly) ---
CONFIG = {
    2: {
        "script": "Models/Layer_2/Scripts/Trend_Model.py",
        "output": "Output_Data/Layer_2_Top_Trending_Per_Type.csv"
    },
    3: {
        "script": "Models/Layer_3/Scripts/Layer_3_Master_File.py",
        "output": "Output_Data/Layer_3_Final_Predictions.csv"
    },
    4: {
        "script": "Models/Layer_4/Scripts/Master_Layer4.py",
        "output": "Output_Data/Layer4_Final_Predictions.csv"
    }
}

# --- HEALTH CHECK (Required for Hugging Face) ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "Poppy Universe ML Service is active"}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # This checks the HF_TOKEN secret you added to Hugging Face
    if credentials.credentials != os.getenv("HF_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid Token")
    return credentials.credentials

@app.post("/run-layer/{layer_num}")
async def run_layer(layer_num: int, request: Request, token: str = Depends(verify_token)):
    if layer_num not in CONFIG:
        raise HTTPException(status_code=404, detail="Layer not found")
    
    body = await request.json()
    mode = body.get("mode", "cached") 
    layer = CONFIG[layer_num]
    
    python_output = ""
    # 1. RUN MODEL (Only if mode is 'run')
    if mode == "run":
        script_path = os.path.join(os.path.dirname(__file__), layer["script"])
        # We pass DATA_SOURCE=database so your Python scripts know to hit Aiven
        result = subprocess.run(["python", script_path], env={**os.environ, "DATA_SOURCE": "database"}, capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "error", "trace": result.stderr}
        python_output = result.stdout

    # 2. READ OUTPUT FILE (The CSV stored on Huggy)
    csv_path = os.path.join(os.path.dirname(__file__), layer["output"])
    if not os.path.exists(csv_path):
        return {"status": "error", "message": f"CSV not found at {layer['output']}"}

    df = pd.read_csv(csv_path)
    data_json = df.to_dict(orient='records')

    return {
        "status": "success",
        "total_rows": len(df),
        "output": python_output,
        "data": data_json
    }

# --- STARTUP COMMAND (Required for Hugging Face) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)