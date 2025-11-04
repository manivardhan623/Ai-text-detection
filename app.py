"""
AI Text Detection - Web Service API
Flask application with local fine-tuned RoBERTa model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

app = Flask(__name__)
CORS(app)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables
model = None
tokenizer = None

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_DIR = "ensemble_models/roberta-base"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")


def load_model():
    """Load the fine-tuned AI text detection model from local directory"""
    global model, tokenizer

    print("\n" + "=" * 70)
    print("LOADING AI TEXT DETECTION MODEL")
    print("=" * 70)

    # Step 1: Verify model directory exists
    if not os.path.exists(MODEL_DIR):
        error_msg = f"Model directory not found: {MODEL_DIR}"
        print(f"✗ {error_msg}")
        return False, error_msg

    if not os.path.exists(MODEL_PATH):
        error_msg = f"Model file not found: {MODEL_PATH}"
        print(f"✗ {error_msg}")
        return False, error_msg

    # Get model size
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n✓ Model file found ({model_size:.2f} MB)")
    print(f"Location: {MODEL_PATH}")

    # Step 2: Load tokenizer from local directory
    try:
        print("\nLoading tokenizer from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        error_msg = f"Error loading tokenizer: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg

    # Step 3: Load fine-tuned model from local directory
    try:
        print("Loading fine-tuned model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=2,
            local_files_only=True
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg

    # Step 4: Prepare model for inference
    model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("✓ MODEL LOADED SUCCESSFULLY!")
    print(f"Device: {device}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Expected Accuracy: 98.69%")
    print("=" * 70 + "\n")

    return True, "Model loaded successfully"


def predict_text(text, min_words=50):
    """Predict if text is AI-generated or human-written"""

    if model is None or tokenizer is None:
        return {
            'error': 'Model not loaded. Please wait for model initialization.',
            'success': False
        }

    # Check word count
    word_count = len(text.split())
    if word_count < min_words:
        return {
            'error': f'Text too short. Minimum {min_words} words required. You provided {word_count} words.',
            'success': False,
            'word_count': word_count
        }

    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            ai_probability = probs[0, 1].item()
            prediction = 1 if ai_probability >= 0.5 else 0

        result = {
            'success': True,
            'prediction': 'AI-GENERATED' if prediction == 1 else 'HUMAN-WRITTEN',
            'prediction_label': prediction,
            'word_count': word_count,
            'text_length': len(text),
            'ai_confidence': f"{ai_probability * 100:.2f}%",
            'human_confidence': f"{(1 - ai_probability) * 100:.2f}%"
        }

        return result

    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'success': False
        }


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Home endpoint"""
    model_status = "loaded" if model is not None else "loading..."

    return jsonify({
        'message': 'AI Text Detection API - Fine-tuned RoBERTa Model',
        'status': 'running',
        'model_loaded': model is not None,
        'device': str(device),
        'accuracy': '98.69%',
        'model_source': 'Local fine-tuned model',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Detect AI-generated text',
            '/model-info': 'GET - Model details'
        },
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'body': {
                'text': 'Your text to analyze (minimum 50 words)',
                'min_words': 50
            }
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'model_file_exists': os.path.exists(MODEL_PATH)
    })


@app.route('/model-info')
def model_info():
    """Model information endpoint"""

    model_size = "N/A"
    if os.path.exists(MODEL_PATH):
        size_bytes = os.path.getsize(MODEL_PATH)
        model_size = f"{size_bytes / (1024 * 1024):.2f} MB"

    # Count parameters if model is loaded
    total_params = "N/A"
    if model is not None:
        total_params = f"{sum(p.numel() for p in model.parameters()):,}"

    return jsonify({
        'model': 'RoBERTa-base (fine-tuned)',
        'parameters': total_params,
        'accuracy': '98.69%',
        'precision': '97.76%',
        'recall': '99.66%',
        'f1_score': '98.70%',
        'model_size': model_size,
        'training_samples': '40,000',
        'model_loaded': model is not None,
        'source': 'Local fine-tuned model',
        'model_path': MODEL_DIR
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for text detection"""

    if model is None:
        return jsonify({
            'error': 'Model is still loading. Please wait a moment and try again.',
            'success': False
        }), 503

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.',
                'success': False,
                'example': {
                    'text': 'Your text here (minimum 50 words)',
                    'min_words': 50
                }
            }), 400

        text = data['text']
        min_words = data.get('min_words', 50)

        result = predict_text(text, min_words)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500


# ============================================
# APPLICATION STARTUP
# ============================================

# Load model on startup
print("\n" + "=" * 70)
print("AI TEXT DETECTION API - STARTING UP")
print("=" * 70)
print(f"Python working directory: {os.getcwd()}")
print(f"Model directory: {MODEL_DIR}")
print(f"Model path: {MODEL_PATH}")
print("=" * 70)

with app.app_context():
    success, message = load_model()
    if success:
        print(f"\n🚀 API READY TO SERVE REQUESTS!")
    else:
        print(f"\n⚠️  WARNING: {message}")
        print("The API will run but predictions will fail.")
        print("\nPlease check:")
        print("1. Model directory exists: ensemble_models/roberta-base")
        print("2. Model file exists: ensemble_models/roberta-base/model.safetensors")
        print("3. All required model files are present (config.json, tokenizer files, etc.)")

print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)