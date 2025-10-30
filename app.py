import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from datetime import datetime
import os
import sqlite3
import pandas as pd
import io
import base64

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'fine_tuned_mobilenet_model.h5'
DB_PATH = 'waste_monitoring.db'

# Waste classification classes (9 categories)
CLASS_LABELS = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
                'metal', 'paper', 'plastic', 'shoes']

# Initialize Database
def init_database():
    """Initialize SQLite database for waste monitoring"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waste_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_data TEXT NOT NULL,
            all_probabilities TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_database(timestamp, classification, confidence, image, all_predictions):
    """Save classification result to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert image to base64 for storage
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Store all probabilities as JSON-like string
    probs_str = ','.join([f"{CLASS_LABELS[i]}:{all_predictions[i]:.4f}" for i in range(len(CLASS_LABELS))])
    
    cursor.execute('''
        INSERT INTO waste_records (timestamp, classification, confidence, image_data, all_probabilities)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, classification, confidence, img_str, probs_str))
    
    conn.commit()
    conn.close()

def get_all_records():
    """Retrieve all records from database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, timestamp, classification, confidence FROM waste_records ORDER BY id DESC", conn)
    conn.close()
    return df

def get_record_image(record_id):
    """Retrieve image for a specific record"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_data FROM waste_records WHERE id=?", (record_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        img_data = base64.b64decode(result[0])
        return Image.open(io.BytesIO(img_data))
    return None

def delete_record(record_id):
    """Delete a record from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM waste_records WHERE id=?", (record_id,))
    conn.commit()
    conn.close()

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
    st.set_page_config(page_title="Waste Monitoring System", page_icon="♻️", layout="wide")
    
    # Initialize database
    init_database()
    
    st.title("♻️ Real-Time Waste Monitoring System")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["📸 Live Capture", "📤 Upload Image", "📊 Dashboard", "🗂️ History"])
    
    # Load model
    model = load_classification_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'fine_tuned_mobilenet_model.h5' exists in the same directory.")
        return
    
    if page == "📸 Live Capture":
        live_capture_page(model)
    elif page == "📤 Upload Image":
        upload_image_page(model)
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "🗂️ History":
        history_page()

def live_capture_page(model):
    """Real-time camera capture page"""
    st.header("📸 Live Camera Waste Detection")
    st.write("Use your camera to capture waste items in real-time")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        # Camera input
        camera_image = st.camera_input("Capture waste item")
        
        if camera_image is not None:
            # Process the captured image
            image = Image.open(camera_image)
            
            # Get timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Make prediction
            with st.spinner('Classifying...'):
                predicted_class, confidence, all_predictions = predict_waste_type(model, image)
            
            # Save to database
            save_to_database(timestamp, predicted_class, confidence, image, all_predictions)
            
            st.success(f"✅ Captured and classified as: **{predicted_class.upper()}** ({confidence:.2f}%)")
            
            # Display result
            with col2:
                st.subheader("Classification Result")
                st.metric("Waste Type", predicted_class.upper())
                st.metric("Confidence", f"{confidence:.2f}%")
                st.metric("Timestamp", timestamp)
                
                # Progress bar
                st.progress(confidence / 100)
                
                # Category info
                category_info = {
                    'battery': '🔋 Battery - Hazardous waste',
                    'biological': '🍎 Biological - Organic waste',
                    'cardboard': '📦 Cardboard - Recyclable',
                    'clothes': '👕 Clothes - Textile waste',
                    'glass': '🥤 Glass - Recyclable',
                    'metal': '🔩 Metal - Recyclable',
                    'paper': '📄 Paper - Recyclable',
                    'plastic': '♻️ Plastic - Recyclable',
                    'shoes': '👟 Shoes - Mixed waste'
                }
                st.info(category_info.get(predicted_class, "Unknown"))

def upload_image_page(model):
    """Upload image page (original functionality)"""
    st.header("📤 Upload Waste Image")
    st.write("Upload an image to classify the type of waste material")
    
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
            
            # Save to database
            if st.button("💾 Save to Database"):
                save_to_database(timestamp, predicted_class, confidence, image, all_predictions)
                st.success("✅ Saved to database!")
            
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

def dashboard_page():
    """Dashboard with statistics"""
    st.header("📊 Waste Monitoring Dashboard")
    
    # Get all records
    df = get_all_records()
    
    if len(df) == 0:
        st.info("No records yet. Start capturing waste items!")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items Captured", len(df))
    
    with col2:
        most_common = df['classification'].mode()[0] if len(df) > 0 else "N/A"
        st.metric("Most Common Type", most_common.upper())
    
    with col3:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
    
    with col4:
        latest = df.iloc[0]['timestamp'] if len(df) > 0 else "N/A"
        st.metric("Latest Capture", latest)
    
    # Classification distribution chart
    st.subheader("📈 Waste Type Distribution")
    class_counts = df['classification'].value_counts()
    st.bar_chart(class_counts)
    
    # Confidence over time
    st.subheader("🎯 Confidence Levels Over Time")
    st.line_chart(df.set_index('id')['confidence'])
    
    # Recent captures
    st.subheader("🕒 Recent Captures")
    st.dataframe(df.head(10), use_container_width=True)

def history_page():
    """View and manage historical records"""
    st.header("🗂️ Classification History")
    
    # Get all records
    df = get_all_records()
    
    if len(df) == 0:
        st.info("No records yet. Start capturing waste items!")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        filter_class = st.multiselect("Filter by Waste Type", 
                                       options=df['classification'].unique(),
                                       default=df['classification'].unique())
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"])
    
    # Apply filters
    filtered_df = df[df['classification'].isin(filter_class)]
    
    # Apply sorting
    if sort_by == "Newest First":
        filtered_df = filtered_df.sort_values('id', ascending=False)
    elif sort_by == "Oldest First":
        filtered_df = filtered_df.sort_values('id', ascending=True)
    elif sort_by == "Highest Confidence":
        filtered_df = filtered_df.sort_values('confidence', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('confidence', ascending=True)
    
    st.write(f"Showing {len(filtered_df)} records")
    
    # Display records with images
    for idx, row in filtered_df.iterrows():
        with st.expander(f"�️ {row['classification'].upper()} - {row['timestamp']} (Confidence: {row['confidence']:.2f}%)"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Load and display image
                img = get_record_image(row['id'])
                if img:
                    st.image(img, use_column_width=True)
            
            with col2:
                st.write(f"**ID:** {row['id']}")
                st.write(f"**Timestamp:** {row['timestamp']}")
                st.write(f"**Classification:** {row['classification'].upper()}")
                st.write(f"**Confidence:** {row['confidence']:.2f}%")
                
                if st.button(f"🗑️ Delete Record #{row['id']}", key=f"delete_{row['id']}"):
                    delete_record(row['id'])
                    st.success(f"Deleted record #{row['id']}")
                    st.rerun()

if __name__ == "__main__":
    main()
