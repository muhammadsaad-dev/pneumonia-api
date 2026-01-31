from fastapi import FastAPI, File, UploadFile, HTTPException
from model_service import PneumoniaModel
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pneumonia Detection API")

# Global variable to hold the model
model_wrapper = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pneumonia_resnet18_weights.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown tasks"""
    global model_wrapper
    try:
        print("Loading model...")
        model_wrapper = PneumoniaModel(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e  # Stop server if model fails

    yield  # App is running

    # Shutdown tasks (if needed)
    print("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running. Use /predict to test."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file, returns Normal/Pneumonia prediction.
    """
    # 1. Validate File Type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images allowed.")
    
    # 2. Read Image
    image_bytes = await file.read()
    
    # 3. Get Prediction from Service
    try:
        probability = model_wrapper.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return probability
    