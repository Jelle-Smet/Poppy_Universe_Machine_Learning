from fastapi import FastAPI, HTTPException
import subprocess
import os
import uvicorn

app = FastAPI(title="Poppy Universe ML Service")

@app.get("/")
def home():
    return {"status": "online", "message": "ML Service is running"}

@app.post("/run-layer/{layer_num}")
def run_layer(layer_num: int):
    # Mapping request to your specific file names
    script_map = {
        2: "Models/Layer_2/Scripts/Trend_Model.py", 
        3: "Models/Layer_3/Scripts/Layer_3_Master_File.py",
        4: "Models/Layer_4/Scripts/Layer_4_Master_File.py"
    }
    
    if layer_num not in script_map:
        raise HTTPException(status_code=404, detail="Layer script not found")
        
    script_path = os.path.join(os.path.dirname(__file__), script_map[layer_num])
    
    try:
        # Runs the script just like a terminal command
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "error", "trace": result.stderr}
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)