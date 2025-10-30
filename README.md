# 🗑️ Real-Time Waste Monitoring System

A smart waste classification system using deep learning (MobileNetV2) with real-time camera monitoring and database tracking.

## 🌟 Features

### 📸 Live Camera Capture
- Real-time waste detection using your device camera
- Instant classification with confidence scores
- Automatic saving to database

### 📤 Image Upload
- Upload waste images for classification
- Optional database storage
- Detailed probability breakdown for all classes

### 📊 Dashboard
- Total items captured statistics
- Most common waste type
- Average confidence scores
- Visual charts and graphs
- Confidence trends over time

### 🗂️ History
- View all captured waste items
- Filter by waste type
- Sort by date or confidence
- View stored images
- Delete records

## 🚀 How to Use

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy from `app.py`

## 📦 Technical Stack
- **Framework:** Streamlit
- **ML Model:** MobileNetV2 (Fine-tuned)
- **Database:** SQLite
- **Image Processing:** PIL/Pillow
- **Deep Learning:** TensorFlow/Keras

## 🎯 Waste Categories
The system classifies 9 types of waste:
- 🔋 Battery
- 🍎 Biological
- 📦 Cardboard
- 👕 Clothes
- 🥤 Glass
- 🔩 Metal
- 📄 Paper
- ♻️ Plastic
- 👟 Shoes

## 📁 Database Schema
```sql
waste_records (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    classification TEXT,
    confidence REAL,
    image_data TEXT (base64),
    all_probabilities TEXT
)
```

## 🔒 Data Storage
- All captured images are stored in SQLite database
- Images are encoded as base64 strings
- Each record includes timestamp, classification, and confidence
- Complete probability distribution saved for analysis

## 📱 Camera Support
Works with:
- Desktop webcams
- Laptop cameras
- Mobile device cameras (when deployed on Streamlit Cloud)

## 🎨 UI Features
- Multi-page navigation
- Real-time statistics
- Interactive charts
- Image gallery
- Record management

---
Built with ❤️ using Streamlit and TensorFlow
