import streamlit as st
import cv2
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from PIL import Image
import os
from datetime import datetime
import pandas as pd
import time

# Set up page configuration
st.set_page_config(
    page_title="Advanced Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Create directories if they don't exist
EMPLOYEE_DIR = 'employees'
ATTENDANCE_DIR = 'attendance'
if not os.path.exists(EMPLOYEE_DIR):
    os.makedirs(EMPLOYEE_DIR)
if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

# Initialize session state variables
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'ensemble' not in st.session_state:
    st.session_state.ensemble = None
if 'employee_names' not in st.session_state:
    st.session_state.employee_names = {}
if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False

def create_ensemble():
    """
    Creates an ensemble classifier using only SVM and MLP classifiers.
    """
    # Create base classifiers with optimized parameters
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale')
    mlp = MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', 
                       solver='adam', alpha=0.0001, max_iter=1000)
    
    # Create and return a simple voting classifier
    return VotingClassifier(
        estimators=[('svm', svm), ('mlp', mlp)],
        voting='soft'
    )

def train_model():
    """
    Trains the recognition model using SVM and MLP ensemble.
    """
    with st.spinner("Training the recognition model..."):
        # Load employee images and prepare training data
        X_train = []
        y_train = []
        employee_names = {}
        
        progress_bar = st.progress(0)
        employee_folders = [f for f in os.listdir(EMPLOYEE_DIR) if os.path.isdir(os.path.join(EMPLOYEE_DIR, f))]
        
        if not employee_folders:
            st.error("No employees found. Please add employees first.")
            return False
        
        # Load and process images
        for idx, employee_name in enumerate(employee_folders):
            employee_path = os.path.join(EMPLOYEE_DIR, employee_name)
            image_files = [f for f in os.listdir(employee_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                st.warning(f"No valid images found for employee {employee_name}")
                continue
            
            # Process each image
            for image_name in image_files:
                image_path = os.path.join(employee_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        features = extract_features(image)
                        X_train.append(features)
                        y_train.append(idx)
                        employee_names[idx] = employee_name
                except Exception as e:
                    st.warning(f"Couldn't process {image_path}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(employee_folders))
        
        # Check if we have enough data
        if not X_train:
            st.error("No valid images found for training.")
            return False
        
        # Train the ensemble
        try:
            ensemble = create_ensemble()
            ensemble.fit(X_train, y_train)
            
            # Save to session state
            st.session_state.ensemble = ensemble
            st.session_state.employee_names = employee_names
            st.session_state.model_trained = True
            
            st.success(f"Model trained successfully with {len(X_train)} images from {len(employee_names)} employees")
            return True
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return False

def extract_features(image):
    # Enhanced feature extraction method that only uses OpenCV
    if isinstance(image, (bytes, np.ndarray)):
        if isinstance(image, bytes):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Convert PIL Image to numpy array
        image = np.array(image)
        
    # Resize for consistency
    image = cv2.resize(image, (128, 128))
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 1. HOG Features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    
    # 2. Histogram features
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # 3. Edge features using Sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_hist = cv2.calcHist([sobel_mag.astype(np.uint8)], [0], None, [16], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    
    # 4. Basic statistical features
    mean = np.mean(gray)
    std = np.std(gray)
    median = np.median(gray)
    
    # Combine all features
    combined_features = np.concatenate([
        hog_features.flatten(),
        hist,
        edge_hist,
        [mean, std, median]
    ])
    
    return combined_features

def detect_faces(image):
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    original_image = image.copy()
    
    # Convert to BGR for OpenCV if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Load face detection model
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        st.error("Error loading face detection model")
        return [], original_image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return faces, original_image

def log_attendance(name, confidence):
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create attendance file for today if it doesn't exist
    attendance_file = os.path.join(ATTENDANCE_DIR, f"{today}.csv")
    
    if not os.path.exists(attendance_file):
        # Create new file with headers
        df = pd.DataFrame(columns=['Name', 'Time', 'Confidence'])
        df.to_csv(attendance_file, index=False)
    
    # Read existing attendance
    df = pd.read_csv(attendance_file)
    
    # Current time
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Check if this person has already been logged today
    if name in df['Name'].values:
        # Only update if the last entry was more than 1 hour ago
        last_entry_time = df[df['Name'] == name].iloc[-1]['Time']
        last_entry_dt = datetime.strptime(last_entry_time, '%H:%M:%S')
        current_dt = datetime.strptime(current_time, '%H:%M:%S')
        
        # Calculate time difference in hours
        time_diff = (current_dt - last_entry_dt).total_seconds() / 3600
        
        if time_diff > 1:
            # Add new entry
            new_entry = pd.DataFrame({'Name': [name], 'Time': [current_time], 'Confidence': [confidence]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(attendance_file, index=False)
            return f"Attendance logged at {current_time}"
        else:
            return "Already logged within the last hour"
    else:
        # Add new entry
        new_entry = pd.DataFrame({'Name': [name], 'Time': [current_time], 'Confidence': [confidence]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        return f"First attendance logged at {current_time}"

def webcam_capture():
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    capture_btn = st.button("Capture Image")
    stop_btn = st.button("Stop Webcam")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam")
        return None
    
    captured_image = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Detect faces
        faces, _ = detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the frame - maintain original colors (BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed")
        
        if capture_btn and len(faces) == 1:
            x, y, w, h = faces[0]
            captured_image = frame[y:y+h, x:x+w]
            st.success("Face captured!")
            break
            
        if stop_btn:
            break
            
        # Short delay to reduce CPU usage
        time.sleep(0.1)
    
    # Release webcam
    cap.release()
    
    return captured_image

def add_employee_ui():
    st.subheader("Add New Employee")
    
    # Two methods to add employee
    method = st.radio("Choose Image Source", ["Upload Image", "Capture from Webcam"])
    
    employee_name = st.text_input("Employee Name")
    if not employee_name:
        st.warning("Please enter employee name")
        return
    
    captured_image = None
    uploaded_image = None
    
    if method == "Upload Image":
        uploaded_file = st.file_uploader("Upload Employee Face Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", width=300)
    else:
        st.write("Click 'Capture' to take a photo when your face is detected")
        captured_image = webcam_capture()
        if captured_image is not None:
            # Convert to RGB for display
            display_img = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            st.image(display_img, caption="Captured Image", width=300)
    
    if st.button("Save Employee"):
        if not employee_name:
            st.error("Employee name is required!")
            return
            
        # Create directory for employee if it doesn't exist
        employee_path = os.path.join(EMPLOYEE_DIR, employee_name)
        if not os.path.exists(employee_path):
            os.makedirs(employee_path)
        
        image_path = os.path.join(employee_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        
        if captured_image is not None:
            # Save captured image
            cv2.imwrite(image_path, captured_image)
            st.success(f"Employee {employee_name} added successfully!")
            # Reset model training flag as we have new data
            st.session_state.model_trained = False
            
        elif uploaded_image is not None:
            # Process uploaded image to detect face
            faces, _ = detect_faces(uploaded_image)
            
            if len(faces) == 0:
                st.error("No face detected in the uploaded image.")
                return
            elif len(faces) > 1:
                st.error("Multiple faces detected. Please upload an image with only one face.")
                return
            else:
                # Extract and save the face
                np_image = np.array(uploaded_image)
                x, y, w, h = faces[0]
                face_img = np_image[y:y+h, x:x+w]
                
                # Convert to BGR for OpenCV save
                if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    
                cv2.imwrite(image_path, face_img)
                st.success(f"Employee {employee_name} added successfully!")
                # Reset model training flag as we have new data
                st.session_state.model_trained = False
        else:
            st.error("No image captured or uploaded.")

def recognize_face_ui():
    st.subheader("Face Recognition")
    
    # Check if model is trained
    if not st.session_state.model_trained:
        if st.button("Train Recognition Model"):
            success = train_model()
            if success:
                st.success("Model trained successfully!")
            else:
                st.error("Failed to train model. Make sure you have added employees.")
        return
    
    # Two methods for recognition
    recognition_method = st.radio(
        "Choose Recognition Method", 
        ["Upload Image", "Live Recognition (Webcam)"]
    )
    
    if recognition_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image for Recognition", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Recognize Face"):
                # Convert PIL image to numpy array for processing
                np_image = np.array(image)
                
                # Detect faces
                faces, display_image = detect_faces(np_image)
                
                if len(faces) == 0:
                    st.error("No faces detected in the image.")
                else:
                    # Create a copy for drawing recognition results
                    result_image = display_image.copy()
                    
                    for (x, y, w, h) in faces:
                        face_image = np_image[y:y+h, x:x+w]
                        
                        # Extract features and predict
                        try:
                            features = extract_features(face_image)
                            prediction = st.session_state.ensemble.predict([features])[0]
                            probabilities = st.session_state.ensemble.predict_proba([features])[0]
                            probability = max(probabilities)
                            
                            # Get name with confidence threshold
                            if probability > 0.65:
                                name = st.session_state.employee_names[prediction]
                                confidence = f"{probability*100:.1f}%"
                                text_color = (0, 255, 0)  # Green for high confidence
                                # Log attendance
                                log_msg = log_attendance(name, confidence)
                                st.success(log_msg)
                            else:
                                name = "Unknown"
                                confidence = f"{probability*100:.1f}%"
                                text_color = (255, 0, 0)  # Red for low confidence
                            
                            # Draw rectangle and name
                            cv2.rectangle(result_image, (x, y), (x+w, y+h), text_color, 2)
                            cv2.putText(result_image, f"{name} ({confidence})", 
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, text_color, 2)
                            
                        except Exception as e:
                            st.error(f"Error during recognition: {str(e)}")
                    
                    # Display result
                    st.image(result_image, caption="Recognition Result", channels="RGB")
    
    else:  # Live Recognition
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("Start Live Recognition")
        
        with col2:
            stop_button = st.button("Stop Recognition")
        
        # Define placeholders for frame and status
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Handle start/stop button clicks
        if start_button:
            st.session_state.recognition_active = True
            st.session_state.camera_started = False
        
        if stop_button:
            st.session_state.recognition_active = False
            return
        
        # Perform live recognition if active
        if st.session_state.recognition_active:
            try:
                # Initialize camera
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not open webcam")
                    st.session_state.recognition_active = False
                    return
                
                # Get a frame
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    cap.release()
                    return
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces, _ = detect_faces(frame_rgb)
                
                # Draw rectangles and recognize faces
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    
                    try:
                        # Extract features and recognize
                        features = extract_features(face_img)
                        prediction = st.session_state.ensemble.predict([features])[0]
                        probabilities = st.session_state.ensemble.predict_proba([features])[0]
                        probability = max(probabilities)
                        
                        if probability > 0.65:
                            name = st.session_state.employee_names[prediction]
                            confidence = f"{probability*100:.1f}%"
                            color = (0, 255, 0)  # Green
                            
                            # Log attendance for recognized person
                            log_result = log_attendance(name, f"{probability*100:.1f}")
                            status_text = f"Recognized: {name} ({confidence}) - {log_result}"
                        else:
                            name = "Unknown"
                            confidence = f"{probability*100:.1f}%"
                            color = (0, 0, 255)  # Red
                            status_text = "Unknown person detected"
                        
                        # Draw rectangle and text
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{name}", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, f"{confidence}", (x, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Show status
                        status_placeholder.info(status_text)
                    except Exception as e:
                        status_placeholder.error(f"Recognition error: {str(e)}")
                
                # Display the frame in original colors
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", caption="Live Recognition")
                
                # Release camera
                cap.release()
                
                # Add continue button
                if st.button("Capture Next Frame"):
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                st.session_state.recognition_active = False

def view_employees_ui():
    st.subheader("View Registered Employees")
    
    if not os.path.exists(EMPLOYEE_DIR) or not os.listdir(EMPLOYEE_DIR):
        st.warning("No employees registered yet.")
        return
    
    employees = [f for f in os.listdir(EMPLOYEE_DIR) if os.path.isdir(os.path.join(EMPLOYEE_DIR, f))]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, employee in enumerate(employees[:len(employees)//2 + len(employees)%2]):
            st.subheader(employee)
            employee_path = os.path.join(EMPLOYEE_DIR, employee)
            image_files = [f for f in os.listdir(employee_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                # Display first image as example
                img_path = os.path.join(employee_path, image_files[0])
                img = Image.open(img_path)
                st.image(img, width=200, caption=f"{employee} ({len(image_files)} images)")
            else:
                st.warning(f"No images found for {employee}")
                
            # Option to delete employee
            if st.button(f"Delete {employee}", key=f"del_{employee}"):
                try:
                    import shutil
                    shutil.rmtree(employee_path)
                    st.success(f"Deleted {employee}")
                    st.session_state.model_trained = False
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error deleting employee: {str(e)}")
    
    with col2:
        for i, employee in enumerate(employees[len(employees)//2 + len(employees)%2:]):
            st.subheader(employee)
            employee_path = os.path.join(EMPLOYEE_DIR, employee)
            image_files = [f for f in os.listdir(employee_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                # Display first image as example
                img_path = os.path.join(employee_path, image_files[0])
                img = Image.open(img_path)
                st.image(img, width=200, caption=f"{employee} ({len(image_files)} images)")
            else:
                st.warning(f"No images found for {employee}")
                
            # Option to delete employee
            if st.button(f"Delete {employee}", key=f"del_{employee}"):
                try:
                    import shutil
                    shutil.rmtree(employee_path)
                    st.success(f"Deleted {employee}")
                    st.session_state.model_trained = False
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error deleting employee: {str(e)}")

def view_attendance_ui():
    st.subheader("Attendance Records")
    
    if not os.path.exists(ATTENDANCE_DIR) or not os.listdir(ATTENDANCE_DIR):
        st.warning("No attendance records found.")
        return
    
    # Get all attendance files
    attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.csv')]
    attendance_files.sort(reverse=True)  # Most recent first
    
    # Select date
    selected_date = st.selectbox("Select Date", attendance_files)
    
    if selected_date:
        # Load attendance data
        attendance_path = os.path.join(ATTENDANCE_DIR, selected_date)
        try:
            df = pd.read_csv(attendance_path)
            
            # Display attendance table
            st.write(f"### Attendance for {selected_date.split('.')[0]}")
            st.dataframe(df)
            
            # Display statistics
            st.write("### Attendance Statistics")
            total_unique = df['Name'].nunique()
            total_entries = len(df)
            
            col1, col2 = st.columns(2)
            col1.metric("Total Unique Employees", total_unique)
            col2.metric("Total Check-ins", total_entries)
            
            # Show who is present today
            st.write("### Currently Present Employees")
            present_employees = df['Name'].unique()
            if len(present_employees) > 0:
                for i, employee in enumerate(present_employees):
                    employee_entries = df[df['Name'] == employee]
                    latest_entry = employee_entries.iloc[-1]
                    st.write(f"**{employee}** - Last check-in: {latest_entry['Time']} (Confidence: {latest_entry['Confidence']})")
            else:
                st.write("No employees checked in yet.")
                
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Attendance Report",
                data=csv,
                file_name=f"attendance_{selected_date}",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error loading attendance data: {str(e)}")

def main():
    st.title("Advanced Face Recognition System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page", 
        ["Home", "Add Employee", "Face Recognition", "View Employees", "Attendance Records"]
    )
    
    # Display system status in sidebar
    st.sidebar.subheader("System Status")
    employee_count = 0
    if os.path.exists(EMPLOYEE_DIR):
        employee_count = len([f for f in os.listdir(EMPLOYEE_DIR) if os.path.isdir(os.path.join(EMPLOYEE_DIR, f))])
    
    st.sidebar.info(f"Registered Employees: {employee_count}")
    st.sidebar.info(f"Model Status: {'Trained' if st.session_state.model_trained else 'Not Trained'}")
    
    # Display current date and time in sidebar
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.info(f"Current Time: {current_time}")
    
    # Main area content based on selected page
    if page == "Home":
        st.write("""
        ## Welcome to the Advanced Face Recognition System
        
        This application allows you to:
        - Register new employees with facial recognition
        - Perform real-time face recognition
        - Track attendance with timestamp logging
        - View and manage employee database
        
        ### How to Use
        1. First, add employees to the system using the 'Add Employee' page
        2. Once you have added employees, train the recognition model
        3. Use the 'Face Recognition' page to identify faces in images or via webcam
        4. Check attendance records in the 'Attendance Records' page
        
        ### Technical Features
        - Multi-model ensemble classification (SVM, KNN, Neural Network)
        - Advanced feature extraction for improved accuracy
        - Attendance tracking with timestamps
        - Confidence-based recognition results
        """)
        
        # Quick access buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Add New Employee"):
                st.session_state.page = "Add Employee"
                st.experimental_rerun()
        with col2:
            if st.button("Start Recognition"):
                st.session_state.page = "Face Recognition"
                st.experimental_rerun()
        with col3:
            if st.button("View Attendance"):
                st.session_state.page = "Attendance Records"
                st.experimental_rerun()
                
    elif page == "Add Employee":
        add_employee_ui()
        
    elif page == "Face Recognition":
        recognize_face_ui()
        
    elif page == "View Employees":
        view_employees_ui()
        
    elif page == "Attendance Records":
        view_attendance_ui()

if __name__ == "__main__":
    main()
