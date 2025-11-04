# AI Text Detection API

A Flask-based REST API for detecting AI-generated text using a fine-tuned RoBERTa model.

## 🎯 Model Performance

- **Accuracy**: 98.69%
- **Precision**: 97.76%
- **Recall**: 99.66%
- **F1 Score**: 98.70%
- **Parameters**: 124,647,170 (~125M)
- **Model Size**: 475.51 MB

## 🚀 Features

- Fine-tuned RoBERTa-base model for AI text detection
- REST API with multiple endpoints
- Real-time text classification
- Confidence scores for predictions
- Minimum word count validation

## 📋 Requirements

- Python 3.11+
- PyTorch
- Transformers
- Flask
- See `requirements.txt` for complete list

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the model files are present in `ensemble_models/roberta-base/`

## 🏃 Running Locally

```bash
python app.py
```

The API will start on `http://localhost:10000`

## 📡 API Endpoints

### 1. Home - `GET /`
Get API information and usage instructions.

### 2. Health Check - `GET /health`
Check if the API and model are loaded.

### 3. Model Info - `GET /model-info`
Get detailed model information and metrics.

### 4. Predict - `POST /predict`
Detect if text is AI-generated or human-written.

**Request Body:**
```json
{
  "text": "Your text to analyze (minimum 50 words)",
  "min_words": 50
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "AI-GENERATED",
  "prediction_label": 1,
  "word_count": 75,
  "text_length": 450,
  "ai_confidence": "99.75%",
  "human_confidence": "0.25%"
}
```

## 🧪 Testing

Run the test script to verify model loading:
```bash
python test_model_load.py
```

## 🌐 Deployment to Render

This project is configured for easy deployment to Render.

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Render will automatically detect `render.yaml` and deploy

**Important**: The model files (~476 MB) must be included in your repository.

## 📁 Project Structure

```
.
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
├── render.yaml                     # Render service configuration
├── ensemble_models/
│   └── roberta-base/              # Fine-tuned model files
│       ├── model.safetensors      # Model weights (475 MB)
│       ├── config.json            # Model configuration
│       ├── tokenizer.json         # Tokenizer
│       └── ...                    # Other model files
└── README.md                       # This file
```

## 🔧 Configuration

- **Port**: Default 10000 (configurable via `PORT` environment variable)
- **Device**: Automatically uses CUDA if available, otherwise CPU
- **Min Words**: Default 50 words for text analysis

## 📝 License

[Add your license here]

## 👥 Authors

[Add your name/team here]

## 🙏 Acknowledgments

- Model: RoBERTa-base (fine-tuned)
- Framework: Hugging Face Transformers
- Training samples: 40,000
