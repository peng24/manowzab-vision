from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import subprocess
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scripts", tags=["Scripts"])

class AnnotateRequest(BaseModel):
    limit: int = 0

def run_annotate_script(limit: int):
    logger.info(f"[Script Runner] Started auto_annotate.py with limit={limit}")
    cmd = ["python", "scripts/auto_annotate.py"]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    try:
        subprocess.run(cmd, check=True)
        logger.info("[Script Runner] ✅ auto_annotate.py completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[Script Runner] ❌ auto_annotate.py failed: {e}")

def run_yolo_train_script():
    logger.info(f"[Script Runner] Started YOLO training")
    cmd = ["yolo", "task=detect", "mode=train", "model=yolov8n.pt", "data=data/data.yaml", "epochs=20", "imgsz=640"]
    try:
        subprocess.run(cmd, check=True)
        logger.info("[Script Runner] ✅ YOLO training completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[Script Runner] ❌ YOLO training failed: {e}")

@router.post("/annotate")
async def start_annotation(req: AnnotateRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_annotate_script, req.limit)
    return {"status": "started", "message": "Auto-Annotation script is running in the background. Check terminal for progress."}

@router.post("/train")
async def start_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_yolo_train_script)
    return {"status": "started", "message": "YOLO Training script is running in the background. ETA up to 10-30 mins."}
