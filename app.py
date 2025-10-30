import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from datetime import datetime
import os

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'fine_tuned_mobilenet_model.h5'

# Waste classification classes (9 categories)
CLASS_LABELS = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
                'metal', 'paper', 'plastic', 'shoes']

# Load the model
@st.cache_resource
def load_classification_model():
    """Load the pre-trained model"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for MobileNetV2"""
    # Resize image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    # Convert to array
    img_array = np.array(img)
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    # Apply MobileNetV2 preprocessing
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return processed_img

def predict_waste_type(model, image):
    """Make prediction on the uploaded image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions) * 100)
    predicted_class = CLASS_LABELS[predicted_class_index]
    return predicted_class, confidence, predictions[0]

# Streamlit UI
def main():
    st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")
    
    st.title("♻️ Waste Classification System")
    st.write("Upload an image to classify the type of waste material")
    
    # Load model
    model = load_classification_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'fine_tuned_mobilenet_model.h5' exists in the same directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load and display the image
        image = Image.open(uploaded_file)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Classification Results")
            
            # Make prediction
            with st.spinner('Classifying...'):
                predicted_class, confidence, all_predictions = predict_waste_type(model, image)
            
            # Display results
            st.markdown(f"**🕒 Timestamp:** `{timestamp}`")
            st.markdown(f"**📋 Classification Label:** `{predicted_class.upper()}`")
            st.markdown(f"**🎯 Confidence:** `{confidence:.2f}%`")
            
            # Visual confidence bar
            st.progress(confidence / 100)
            
            # Show all predictions for debugging
            st.markdown("---")
            st.markdown("### 🔍 All Class Probabilities")
            prob_data = {CLASS_LABELS[i]: f"{all_predictions[i]*100:.2f}%" for i in range(len(CLASS_LABELS))}
            sorted_probs = sorted(prob_data.items(), key=lambda x: float(x[1].strip('%')), reverse=True)
            for class_name, prob in sorted_probs:
                st.text(f"{class_name.capitalize()}: {prob}")
            
            # Category information
            st.markdown("---")
            st.markdown("### Category Information")
            category_info = {
                'battery': '🔋 Battery - Hazardous waste requiring special disposal',
                'biological': '🍎 Biological - Organic/food waste for composting',
                'cardboard': '📦 Cardboard - Recyclable paper product',
                'clothes': '👕 Clothes - Textile waste for donation or recycling',
                'glass': '🥤 Glass - Recyclable material',
                'metal': '🔩 Metal - Recyclable material',
                'paper': '📄 Paper - Recyclable material',
                'plastic': '♻️ Plastic - Recyclable material',
                'shoes': '👟 Shoes - Textile/mixed material waste'
            }
            st.info(category_info.get(predicted_class, "Unknown category"))
        
        # Summary section
        st.markdown("---")
        st.subheader("📊 Classification Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Timestamp", timestamp)
        with summary_col2:
            st.metric("Waste Type", predicted_class.upper())
        with summary_col3:
            st.metric("Confidence", f"{confidence:.2f}%")

if __name__ == "__main__":
    main()
