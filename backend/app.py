from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from model import SoilCropModel
from pest_model import predict_from_pil, predict_pest_detailed, get_pest_classes
from PIL import Image
import os
import json

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Initialize models
model = SoilCropModel('data_core.csv')

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "message": "Pest Detection API",
        "version": "1.0.0",
        "endpoints": {
            "crop_recommendation": "/recommend_crops",
            "fertilizer_recommendation": "/recommend_fertilizer", 
            "pest_detection": "/detect_pest",
            "pest_detection_detailed": "/detect_pest_detailed",
            "pest_classes": "/pest_classes",
            "model_info": "/model_info"
        }
    })

@app.route('/recommend_crops', methods=['POST'])
def recommend_crops():
    """Get crop recommendations based on soil and environmental data"""
    data = request.json
    required = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
    
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400
    
    try:
        inputs = {k: float(data[k]) for k in required}
    except Exception:
        return jsonify({"error": "Invalid input"}), 400
    
    result = model.recommend_crops(
        inputs['Temparature'], inputs['Humidity'], inputs['Moisture'],
        inputs['Nitrogen'], inputs['Phosphorous'], inputs['Potassium']
    )
    return jsonify(result)

@app.route('/recommend_fertilizer', methods=['POST'])
def recommend_fertilizer():
    """Get fertilizer recommendations for selected crop"""
    data = request.json
    required = ['Crop', 'Nitrogen', 'Phosphorous', 'Potassium']
    
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400
    
    crop = data.get('Crop')
    try:
        n = float(data.get('Nitrogen'))
        p = float(data.get('Phosphorous'))
        k = float(data.get('Potassium'))
    except Exception:
        return jsonify({"error": "Invalid input"}), 400
    
    result = model.recommend_fertilizer(crop, n, p, k)
    return jsonify(result)

@app.route('/detect_pest', methods=['POST'])
def detect_pest():
    """Detect pest from uploaded image (simple response)"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided (field name 'image')"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    try:
        result = predict_from_pil(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Pest detection failed", "details": str(e)}), 500

@app.route('/detect_pest_detailed', methods=['POST'])
def detect_pest_detailed():
    """Detect pest from uploaded image (detailed response)"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided (field name 'image')"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    try:
        result = predict_pest_detailed(img, top_k=5)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Pest detection failed", "details": str(e)}), 500

@app.route('/pest_classes', methods=['GET'])
def pest_classes():
    """Get list of available pest classes"""
    try:
        result = get_pest_classes()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Failed to get pest classes", "details": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information and statistics"""
    from pest_model import get_detector
    info = {
        "model_loaded": False,
        "labels_loaded": False,
        "num_classes": 0,
        "class_names": [],
        "status": "degraded"
    }

    # Try to load detector, but don't fail the endpoint if it errors
    try:
        detector = get_detector()
        model_info = detector.get_model_info()
        info.update(model_info)
        info["status"] = "healthy" if model_info.get("model_loaded") and model_info.get("labels_loaded") else "degraded"
    except Exception as e:
        info["error"] = str(e)

    # Add training summary if available
    summary_path = 'pest_model/training_summary.json'
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                training_summary = json.load(f)
            info['training_summary'] = training_summary
        except Exception as e:
            info['training_summary_error'] = str(e)

    return jsonify(info)

@app.route('/training_visualizations/<filename>')
def get_training_visualization(filename):
    """Serve training visualization images"""
    try:
        file_path = os.path.join('pest_model', filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": "Failed to serve file", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    from pest_model import get_detector
    resp = {
        "status": "unhealthy",
        "model_loaded": False,
        "labels_loaded": False,
        "num_classes": 0
    }
    try:
        detector = get_detector()
        resp["model_loaded"] = detector.model is not None
        resp["labels_loaded"] = detector.labels is not None
        resp["num_classes"] = len(detector.labels) if resp["labels_loaded"] else 0
        resp["status"] = "healthy" if resp["model_loaded"] and resp["labels_loaded"] else "degraded"
    except Exception as e:
        resp["error"] = str(e)

    return jsonify(resp)

if __name__ == '__main__':
    print("Starting Pest Detection API...")
    print("Available endpoints:")
    print("- POST /detect_pest - Simple pest detection")
    print("- POST /detect_pest_detailed - Detailed pest detection")
    print("- GET /pest_classes - List available pest classes")
    print("- GET /model_info - Model information and statistics")
    print("- GET /health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5000)