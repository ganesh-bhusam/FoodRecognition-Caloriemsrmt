import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu

# --- Load Models and Data ---
MODEL_PATH = "Model/best.pt"
CLASS_FILE = "items.txt"

# Calorie database
CALORIE_DB = {
    "burger_261kcal": 261,
    "chicken_biryani_292kcal": 292,
    "chicken_fride_rice_343kcal": 343,
    "white_rice_242kcal": 242,
    "apple_52kcal": 52,
    "banana_89kcal": 89,
    "onion_40kcal": 40,
    "chicken_curry_110kcal": 110,
    "chicken_fry_246kcal": 246,
    "veg_birayani_206kcal": 206,
    "lemon_29kcal": 29,
    "spoone": 0,
    "bowl": 0,
    "plate": 0
}

# Load class names
with open(CLASS_FILE, "r") as f:
    class_list = [line.strip() for line in f if line.strip()]

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# --- Helper Functions ---
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == "male":
        return 10*weight + 6.25*height - 5*age + 5
    else:
        return 10*weight + 6.25*height - 5*age - 161

def calculate_tdee(bmr, activity):
    activity_factors = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9
    }
    return bmr * activity_factors.get(activity.lower(), 1.2)

def detect_food(image):
    try:
        results = model(image)
        annotated_image = results[0].plot()
        detections = []
        
        for box in results[0].boxes:
            confidence = box.conf.item()
            if confidence < 0.5:
                continue
                
            class_name = results[0].names[int(box.cls.item())]
            if class_name in class_list:
                detections.append({
                    "Class": class_name,
                    "Calories": CALORIE_DB.get(class_name, 0),
                    "Confidence": round(confidence, 2),
                })
        
        return annotated_image, detections
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None, []

# --- Streamlit UI ---
st.set_page_config(page_title="Food Calorie Tracker", page_icon="🍎")

# Initialize session state
if 'calories_detected' not in st.session_state:
    st.session_state.calories_detected = 0
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Sidebar - User Inputs
with st.sidebar:
    st.header("User Profile 👤")
    name = st.text_input("Your Name", "Guest")
    weight = st.number_input("Weight (kg)", 30, 150, 70)
    height = st.number_input("Height (cm)", 100, 250, 170)
    age = st.number_input("Age", 10, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    activity = st.selectbox("Activity Level", [
        "Sedentary", "Lightly Active", "Moderately Active",
        "Very Active", "Extra Active"
    ])
    
    health = st.multiselect("Health Conditions", ["Diabetes", "High Blood Pressure"])
    
    # Calculate calories
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity)
    if "Diabetes" in health:
        tdee *= 0.9
    if "High Blood Pressure" in health:
        tdee *= 0.95
    
    st.success(f"### Daily Calories: {tdee:.0f} kcal")

# Main Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Webcam", "About", "Information"],
    icons=["house", "camera", "info-circle", "book"],
    orientation="horizontal"
)

# Home Page
if selected == "Home":
    st.title(f"🍽️ Hello {name}! Food Calorie Tracker")
    st.write("Upload a food image to analyze its calories")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Your Meal", use_container_width=True)
            
            # Convert PIL image to numpy array
            img_np = np.array(img)
            annotated_img, detections = detect_food(img_np)
            
            if annotated_img is not None:
                st.image(annotated_img, caption="Analysis", use_container_width=True)
                
                if detections:
                    st.subheader("🍎 Detection Results")
                    st.table(detections)
                    
                    total_cal = sum(item["Calories"] for item in detections)
                    st.session_state.calories_detected = total_cal
                    remaining = tdee - total_cal
                    
                    if remaining > 0:
                        st.success(f"✅ You have {remaining:.0f} calories remaining today!")
                    else:
                        st.warning(f"⚠️ You're over by {-remaining:.0f} calories!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Webcam Page
elif selected == "Webcam":
    st.title("🎥 Real-Time Food Detection")
    st.write("Use your webcam to detect food items and calculate calories")
    
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.empty()
    cap = cv2.VideoCapture(0)
    
    # Detection results placeholder
    results_placeholder = st.empty()
    calories_placeholder = st.empty()
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
            
        # Convert frame to RGB (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        annotated_frame, detections = detect_food(frame_rgb)
        
        if annotated_frame is not None:
            # Display annotated frame
            FRAME_WINDOW.image(annotated_frame, channels="RGB")
            
            if detections:
                # Update detection results
                with results_placeholder.container():
                    st.subheader("Detected Items")
                    st.table(detections)
                
                # Update calorie count
                total_cal = sum(item["Calories"] for item in detections)
                st.session_state.calories_detected = total_cal
                remaining = tdee - total_cal
                
                with calories_placeholder.container():
                    if remaining > 0:
                        st.success(f"✅ Total calories: {total_cal} | Remaining: {remaining:.0f} kcal")
                    else:
                        st.warning(f"⚠️ Total calories: {total_cal} | Over by {-remaining:.0f} kcal")
        else:
            FRAME_WINDOW.image(frame, channels="BGR")
    
    cap.release()
    if not run:
        FRAME_WINDOW.write("Webcam is off")
        if st.session_state.calories_detected > 0:
            st.success(f"Last detected calories: {st.session_state.calories_detected}")

# About and Information pages remain the same...

    # ... (keep all your previous imports and setup code)

# About Page
elif selected == "About":
    st.title("About ℹ️")
    st.write("""
    ## Food Calorie Detection App
    
    This application uses YOLO object detection to:
    - Identify food items in images/webcam
    - Calculate calorie content
    - Provide personalized nutrition advice
    
    ### Project Repository
    Check out the source code on GitHub:
    [GitHub Repository](https://github.com/subramanyamrekhandar/Road-Accident-Detection-By-using-AI.git)
    
    ### Features
    - Image upload analysis
    - Real-time webcam detection
    - Personalized calorie recommendations
    - Health condition considerations
    """)
    st.markdown("[![GitHub](https://img.shields.io/badge/View_on_GitHub-181717?logo=github)](https://github.com/subramanyamrekhandar/Road-Accident-Detection-By-using-AI.git)")

# ... (rest of your code remains the same)
elif selected == "Information":
    st.title("Information 📚")
    st.write(f"Supported food items: {', '.join([k for k in CALORIE_DB.keys() if CALORIE_DB[k] > 0])}")

# Footer
st.sidebar.title("Contact 📞")
st.sidebar.info("Created by [BATCH - 01](https://ibb.co/GfDzNdBG)")