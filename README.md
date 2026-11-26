# PneumoGuard AI 🫁

**AI-Powered Pneumonia Detection from Chest X-Ray Images**

PneumoGuard AI is a deep-learning-based web application that analyzes chest X-ray images and predicts whether the image indicates **NORMAL** or **PNEUMONIA**. It uses MobileNetV2 transfer learning and integrates AI-generated explanations and downloadable PDF reports.

## 🚀 Features

- Upload chest X-ray images for prediction.
- Detect NORMAL and PNEUMONIA cases.
- MobileNetV2-based deep learning model.
- AI-generated explanations using Google Gemini.
- Clinical notes and prediction results.
- Downloadable PDF prediction reports.
- React-based user interface.
- FastAPI backend for model inference.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Node.js, Express.js |
| AI Backend | Python, FastAPI |
| Deep Learning | TensorFlow, Keras, MobileNetV2 |
| AI Integration | Google Gemini |
| Image Processing | Pillow, NumPy |
| Reports | PDF generation |

## 🧠 Model Information

- **Architecture:** MobileNetV2
- **Approach:** Transfer Learning
- **Input Image Size:** 224 × 224
- **Output Classes:** NORMAL and PNEUMONIA
- **Dataset:** Chest X-Ray Images (Pneumonia) dataset

### Model Performance

The selected V3 model achieved the following results on the held-out test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | 86.70% |
| Pneumonia Precision | 84.34% |
| Pneumonia Recall | 96.67% |
| Pneumonia F1-Score | 90.08% |

**Confusion Matrix**

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 164 | 70 |
| PNEUMONIA | 13 | 377 |

These results are from the project's evaluation run and do not represent clinical validation.

## 📂 Project Structure

```text
pneumoguard-ai/
│
├── src/
│   ├── backend_python/
│   │   ├── main.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── requirements.txt
│   │
│   ├── components/
│   │   ├── PredictionCard.tsx
│   │   └── UploadZone.tsx
│   │
│   ├── App.tsx
│   ├── main.tsx
│   ├── index.css
│   ├── server.ts
│   └── types.ts
│
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kumarr-shubham/pneumoguard-ai.git
cd pneumoguard-ai
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Set up Python environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r src/backend_python/requirements.txt
```

### 4. Prepare the trained model

The trained `.keras` model is not included in this repository.

Download and prepare the Chest X-Ray Images (Pneumonia) dataset, then train the model using:

```bash
python src/backend_python/train.py
```

Ensure the generated model is saved at:

```text
src/backend_python/pneumonia_model_v3.keras
```

### 5. Configure Gemini API

Configure your Gemini API key using the environment variable expected by the Node.js backend:

```text
GEMINI_API_KEY=your_api_key_here
```

Keep your actual API key private. Never commit your `.env` file.

### 6. Start the Python backend

```bash
python -m uvicorn src.backend_python.main:app --reload --port 8000
```

### 7. Start the application

Open another terminal in the project directory:

```bash
npm run dev
```

Open the application at:

```text
http://localhost:3000
```

## 📊 Dataset

This project uses the Chest X-Ray Images (Pneumonia) dataset by Paul Mooney, available on Kaggle.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## 🔮 Future Improvements

- Improve model generalization and validation reliability.
- Reduce false positive predictions.
- Expand testing with additional X-ray datasets.
- Improve model interpretability.
- Optimize application deployment.

## ⚠️ Disclaimer

PneumoGuard AI is an educational project developed for learning and demonstration purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice or diagnosis.

## 👨‍💻 Author

**Shubham Kumar**

GitHub: https://github.com/kumarr-shubham

---

⭐ If you find this project interesting, consider giving the repository a star!