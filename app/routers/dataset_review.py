from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/review", tags=["Dataset Review"])

class DeleteRequest(BaseModel):
    filenames: List[str]

@router.get("/images")
async def get_review_images():
    """Returns a list of all auto-annotated images available for review."""
    img_dir = Path("data/train/images")
    if not img_dir.exists():
        return {"images": []}
        
    images = []
    for f in img_dir.glob("*.jpg"):
        images.append(f.name)
    return {"images": sorted(images)}

@router.post("/delete")
async def delete_rejected_images(req: DeleteRequest):
    """Deletes the rejected .jpg and corresponding .txt label files permanently."""
    img_dir = Path("data/train/images")
    lbl_dir = Path("data/train/labels")
    
    deleted_count = 0
    for filename in req.filenames:
        # Delete image
        img_path = img_dir / filename
        if img_path.exists():
            try:
                img_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete image {img_path}: {e}")
                
        # Delete corresponding label
        base_name = Path(filename).stem
        lbl_path = lbl_dir / f"{base_name}.txt"
        if lbl_path.exists():
            try:
                lbl_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete label {lbl_path}: {e}")
                
    return {"status": "success", "deleted": deleted_count}
