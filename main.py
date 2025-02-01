from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import io
import yolov8  # Assuming you have YOLOv8 installed
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load YOLOv8 model
yolo_model = yolov8.YOLO("best.pt")  # Your custom trained model

# Food category database
FOOD_DB = {
    "biryani": {
        "category": "Desi Food",
        "nutrition": {"calories": 350, "protein": 15, "carbs": 45},
        "recipe": ["Basmati rice", "Chicken/Beef", "Yogurt", "Spices"]
    },
    "pizza": {
        "category": "Fast Food",
        "nutrition": {"calories": 285, "protein": 12, "carbs": 36},
        "recipe": ["Dough", "Cheese", "Tomato sauce", "Toppings"]
    },
    # Add more food items
}

# Nutrition database for fruits
FRUIT_DB = {
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
    details: dict
    description: str = None
    recipe: list = None
    nutrition: dict = None

def is_food_item(label: str) -> bool:
    food_categories = ["desi", "fast", "food", "fruit"]
    return any(cat in label.lower() for cat in food_categories)

async def gemini_analysis(image_bytes: bytes) -> AnalysisResponse:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = """Analyze this image and provide:
        - Object name (exclude desi food, fast food, and fruits)
        - Category
        - Detailed technical description
        - Material composition
        - Typical usage scenarios"""

        response = gemini_model.generate_content([prompt, img])
        
        # Parse Gemini response
        # You'll need to implement proper response parsing based on your format
        parsed_response = parse_gemini_output(response.text)
        
        return AnalysisResponse(
            model_used="Gemini 1.5 Flash",
            **parsed_response
        )
        
    except Exception as e:
        raise HTTPException(500, f"Gemini analysis failed: {str(e)}")

async def yolo_analysis(image_bytes: bytes) -> AnalysisResponse:
    try:
        # Perform YOLO detection
        results = yolo_model.predict(image_bytes)
        
        # Get top detection
        detection = results[0]
        label = detection.names[int(detection.boxes.cls[0])]
        confidence = detection.boxes.conf[0].item()
        
        # Get additional details
        if label.lower() in FRUIT_DB:
            details = FRUIT_DB[label.lower()]
            details["confidence"] = confidence
            return AnalysisResponse(
                model_used="YOLOv8 Custom",
                object_name=label,
                category=details["category"],
                nutrition=details["nutrition"],
                description=f"Nutritional information for {label}"
            )
        else:
            details = FOOD_DB.get(label.lower(), {})
            details["confidence"] = confidence
            return AnalysisResponse(
                model_used="YOLOv8 Custom",
                object_name=label,
                category=details.get("category", "Food"),
                recipe=details.get("recipe", []),
                nutrition=details.get("nutrition", {}),
                description=f"Details for {label}"
            )
            
    except Exception as e:
        raise HTTPException(500, f"YOLO analysis failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")
    
    image_bytes = await file.read()
    
    try:
        # First check if it's food using YOLO quick check
        is_food = await check_food_quick(image_bytes)
        
        if is_food:
            return await yolo_analysis(image_bytes)
        else:
            return await gemini_analysis(image_bytes)
            
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

async def check_food_quick(image_bytes: bytes) -> bool:
    """Quick check using YOLO to determine if food item"""
    try:
        results = yolo_model.predict(image_bytes)
        detection = results[0]
        label = detection.names[int(detection.boxes.cls[0])]
        return is_food_item(label)
    except:
        return False

def parse_gemini_output(text: str) -> dict:
    """Implement your custom parsing logic here"""
    # Example implementation:
    lines = text.split("\n")
    return {
        "object_name": lines[0].split(": ")[1],
        "category": lines[1].split(": ")[1],
        "description": lines[2].split(": ")[1],
        "details": {
            "material": lines[3].split(": ")[1],
            "usage": lines[4].split(": ")[1]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
