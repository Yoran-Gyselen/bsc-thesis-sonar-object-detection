from datetime import datetime
from pathlib import Path
import os

def create_model_dir(project_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    working_dir = os.path.join("models", project_name)
    
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    
    log_path = os.path.join(working_dir, f"{project_name}_{timestamp}.txt")
    model_path = log_path.replace(".txt", ".pth")

    return log_path, model_path