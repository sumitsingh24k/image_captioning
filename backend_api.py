from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from caption_service import CaptionService


PROJECT_DIR = Path(r"e:\dl\image_captioning")
service = CaptionService(PROJECT_DIR)

app = FastAPI(title="Image Captioning API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    try:
        generated = service.caption(image, mode="blip")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"caption": generated, "mode": "blip", "filename": file.filename}
