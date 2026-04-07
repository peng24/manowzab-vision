"""
app/routers/dataset_review.py
─────────────────────────────────────────────────────────────────
Dataset Curation API:
  - Source: data/needs_training/
  - Accept  → move to data/train/images/
  - Reject  → delete immediately
  - Session state persisted to data/curation_state.json
    (so the user can resume across browser refreshes)
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/review", tags=["Dataset Review"])

# ── Directories ──────────────────────────────────────────────────────────────
SRC_DIR   = Path("data/needs_training")
DEST_DIR  = Path("data/train/images")
STATE_FILE = Path("data/curation_state.json")

# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load persistent curation state, or return a fresh default."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"decisions": {}}   # {"filename": "accept" | "reject" | "pending"}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Request Schemas ───────────────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    filename: str
    action: Literal["accept", "reject"]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/images")
async def get_review_images():
    """
    Return all images from data/needs_training/ together with any
    saved decisions so the frontend can resume mid-session.
    """
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()

    images = []
    for f in sorted(SRC_DIR.glob("*.jpg")):
        decision = state["decisions"].get(f.name, "pending")
        images.append({"name": f.name, "decision": decision})

    return {"images": images, "total": len(images)}


@router.get("/image/{filename}")
async def serve_image(filename: str):
    """Serve a single source image for the review UI."""
    path = SRC_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/decide")
async def decide(req: DecisionRequest):
    """
    Accept or reject a single image.
    - accept → move jpg (+ txt label if present) to data/train/images/
    - reject → delete jpg (+ txt label) immediately
    Decision is also persisted to curation_state.json.
    """
    src_path = SRC_DIR / req.filename
    state = _load_state()

    if req.action == "accept":
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        if src_path.exists():
            dest = DEST_DIR / req.filename
            shutil.move(str(src_path), str(dest))
            logger.info("[Curation] ✅ Accept → %s", dest)
        state["decisions"][req.filename] = "accept"
        _save_state(state)
        return {"status": "ok", "action": "accept", "filename": req.filename}

    elif req.action == "reject":
        if src_path.exists():
            src_path.unlink()
            logger.info("[Curation] 🗑️ Reject → deleted %s", req.filename)
        # Also remove a matching label if one exists in data/needs_training/labels/
        lbl = SRC_DIR / "labels" / (Path(req.filename).stem + ".txt")
        if lbl.exists():
            lbl.unlink()
        state["decisions"][req.filename] = "reject"
        _save_state(state)
        return {"status": "ok", "action": "reject", "filename": req.filename}

    raise HTTPException(status_code=400, detail="Invalid action")


@router.get("/stats")
async def get_stats():
    """Summary counts for the curation session."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    decisions = state.get("decisions", {})

    remaining = sum(
        1 for f in SRC_DIR.glob("*.jpg")
        if decisions.get(f.name, "pending") == "pending"
    )
    accepted = sum(1 for v in decisions.values() if v == "accept")
    rejected = sum(1 for v in decisions.values() if v == "reject")

    return {
        "remaining": remaining,
        "accepted": accepted,
        "rejected": rejected,
        "total_decided": accepted + rejected,
    }


@router.post("/reset")
async def reset_state():
    """Clear the saved session (start fresh — does NOT move or delete files)."""
    _save_state({"decisions": {}})
    return {"status": "reset"}
