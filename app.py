import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🧑",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Model paths
# -----------------------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# -----------------------------
# Download model files if not present
# -----------------------------
@st.cache_resource
def download_model_files():
    """
    Downloads the pre-trained face detection model files if they are not already present.
    """
    import requests
    
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists(PROTOTXT_PATH):
        st.info("Downloading deploy.prototxt...")
        with open(PROTOTXT_PATH, "wb") as f:
            f.write(requests.get(prototxt_url).content)
    
    if not os.path.exists(CAFFEMODEL_PATH):
        st.info("Downloading res10_300x300_ssd_iter_140000.caffemodel (10.7 MB)...")
        with open(CAFFEMODEL_PATH, "wb") as f:
            f.write(requests.get(caffemodel_url).content)

# -----------------------------
# Load the face detection model
# -----------------------------
@st.cache_resource
def load_face_detection_model():
    """
    Loads the pre-trained OpenCV DNN face detection model.
    
    Returns:
        cv2.dnn_Net: Loaded neural network.
    """
    download_model_files()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    return net

face_net = load_face_detection_model()

# -----------------------------
# Face detection function
# -----------------------------
def detect_faces(image, net, confidence_threshold=0.5):
    """
    Detects faces in the input image/frame.
    
    Args:
        image (np.ndarray): Input in BGR format.
        net (cv2.dnn_Net): Loaded model.
        confidence_threshold (float): Minimum confidence.
    
    Returns:
        np.ndarray: Processed image with boxes.
        int: Number of detected faces.
    """
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            face_count += 1
    
    return image, face_count

# -----------------------------
# UI
# -----------------------------
st.title("🧑 Face Detection using OpenCV DNN")
st.markdown("Upload an **image** or **video** to detect faces using a pre-trained deep learning model.")

tab1, tab2 = st.tabs(["🖼 Image", "🎥 Video"])

with tab1:
    st.header("Image Detection")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    conf_img = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, key="img")
    
    if uploaded_image:
        img = Image.open(uploaded_image)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        result_img, count = detect_faces(img_cv.copy(), face_net, conf_img)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        st.image(result_rgb, caption=f"Detected {count} face(s)", use_column_width=True)
        st.success(f"Found {count} face(s)!") if count > 0 else st.warning("No faces detected. Try lowering the threshold.")

with tab2:
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    conf_vid = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, key="vid")
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        st.info("Processing video...")
        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0
        
        total_faces = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, faces = detect_faces(frame.copy(), face_net, conf_vid)
            total_faces += faces
            out.write(processed_frame)
            
            processed += 1
            progress.progress(processed / frame_count if frame_count > 0 else 1)
        
        cap.release()
        out.release()
        
        st.success("Processing complete!")
        st.video(output_path)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • OpenCV DNN (Res10 SSD) • Pre-trained Model")
