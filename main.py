from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import io
from ultralytics import YOLO  # Updated YOLOv8 import
from pydantic import BaseModel
import uvicorn
import os
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
YOLO_CLASSES = {"desi_food", "fast_food", "fruit"}  # Your trained classes
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

# Initialize Models
try:
    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Load YOLOv8 model
    yolo_model = YOLO("best.pt")  # Your custom trained model
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise RuntimeError("Failed to initialize models")

# Enhanced Databases
FOOD_DB: Dict[str, Dict] = {
    "biryani": {
        "category": "Desi Food",
        "nutrition": {"calories": 350, "protein": 15, "carbs": 45},
        "recipe": ["Basmati rice", "Chicken/Beef", "Yogurt", "Spices"]
    },
    # Add more items
}

FRUIT_DB: Dict[str, Dict] = {
    "apple": {
        "category": "Fruit",
        "nutrition": {"calories": 95, "vitamin_c": "14%", "fiber": "17%"}
    },
    # Add more fruits
}

class AnalysisResponse(BaseModel):
    model_used: str
    object_name: str
    category: str
    confidence: Optional[float] = None
    description: str
    details: Dict
    nutrition: Optional[Dict] = None
    recipe: Optional[List[str]] = None

def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, "Unsupported file type")
    if file.size > MAX_IMAGE_SIZE:
        raise HTTPException(400, "File too large")

async def get_image_bytes(file: UploadFile) -> bytes:
    return await file.read()

def is_yolo_target(label: str) -> bool:
    """Check if detection matches trained classes"""
    return label.lower() in YOLO_CLASSES

async def process_yolo_detection(image_bytes: bytes) -> Optional[AnalysisResponse]:
    """Enhanced YOLO processing with error handling"""
    try:
        results = yolo_model.predict(image_bytes)
        if not results:
            return None

        detection = results[0]
        if len(detection.boxes.cls) == 0:
            return None

        label = detection.names[int(detection.boxes.cls[0])]
        confidence = detection.boxes.conf[0].item()

        if not is_yolo_target(label):
            return None

        # Get additional details
        if label.lower() in FRUIT_DB:
            details = FRUIT_DB[label.lower()]
            return AnalysisResponse(
                model_used="YOLOv8 Custom",
                object_name=label,
                category=details["category"],
                confidence=confidence,
                description=f"Nutritional information for {label}",
                details=details,
                nutrition=details.get("nutrition", {})
            )
        else:
            details = FOOD_DB.get(label.lower(), {})
            return AnalysisResponse(
                model_used="YOLOv8 Custom",
                object_name=label,
                category=details.get("category", "Food"),
                confidence=confidence,
                description=f"Details for {label}",
                details=details,
                recipe=details.get("recipe", []),
                nutrition=details.get("nutrition", {})
            )
            
    except Exception as e:
        logger.error(f"YOLO processing error: {str(e)}")
        return None

async def process_gemini_analysis(image_bytes: bytes) -> AnalysisResponse:
    """Enhanced Gemini analysis with structured output"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        structured_prompt = """Analyze this image and provide STRICT JSON format response with these fields:
        - object_name: (non-food item name)
        - category: (technical category)
        - technical_description: (detailed technical specs)
        - materials: (list of materials)
        - typical_usage: (common usage scenarios)"""

        response = gemini_model.generate_content([structured_prompt, img])
        
        try:
            parsed = parse_gemini_json(response.text)
            return AnalysisResponse(
                model_used="Gemini 1.5 Flash",
                object_name=parsed.get("object_name", "Unknown Object"),
                category=parsed.get("category", "General"),
                description=parsed.get("technical_description", ""),
                details={
                    "materials": parsed.get("materials", []),
                    "usage": parsed.get("typical_usage", [])
                }
            )
        except Exception as e:
            logger.error(f"Gemini parse error: {str(e)}")
            raise HTTPException(500, "Failed to parse AI response")

    except Exception as e:
        logger.error(f"Gemini analysis failed: {str(e)}")
        raise HTTPException(500, "General analysis failed")

def parse_gemini_json(text: str) -> Dict:
    """Extract JSON from Gemini response"""
    start = text.find('{')
    end = text.rfind('}') + 1
    json_str = text[start:end]
    return json.loads(json_str)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    validate_image(file)
    image_bytes = await get_image_bytes(file)
    
    try:
        # First try YOLO detection
        yolo_response = await process_yolo_detection(image_bytes)
        if yolo_response:
            return yolo_response
            
        # Fallback to Gemini
        return await process_gemini_analysis(image_bytes)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(500, "Analysis process failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)