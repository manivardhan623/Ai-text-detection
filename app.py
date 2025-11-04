from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class AITextDetector:
    def __init__(self, model_dir="ensemble_models/roberta-base"):
        """Initialize the detector with the trained model"""
        print("Loading model from", model_dir)

        # Load transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, text):
        """Make predictions on text input"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prob = probs[0, 1].item()  # Probability of being AI-generated
            prediction = 1 if prob >= 0.5 else 0

        return prediction, prob


# Initialize detector
model_dir = "ensemble_models/roberta-base"
detector = AITextDetector(model_dir)
print("AI Text Detector initialized successfully!")


@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "AI Text Detection API is running"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Make prediction
        prediction, confidence = detector.predict(text)

        return jsonify({
            "prediction": int(prediction),
            "confidence": float(confidence),
            "text_length": len(text),
            "word_count": len(text.split())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
