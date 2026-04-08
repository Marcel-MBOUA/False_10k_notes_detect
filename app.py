import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from joblib import load
import numpy as np
from PIL import Image

import cv2
import numpy as np


# ---------------------------------------------------------
# Helper 1: Order 4 points consistently
# ---------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect


# ---------------------------------------------------------
# Helper 2: Perspective transform
# ---------------------------------------------------------
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# ---------------------------------------------------------
# Main Function: Robust Tight Crop (High-Res Preserving)
# ---------------------------------------------------------
def tight_crop_banknote(img):
    """
    Input:
        img (BGR OpenCV image, high resolution)

    Output:
        Cropped + perspective-corrected banknote (original resolution)
        or None if detection fails
    """

    original = img.copy()

    # ------------------------------
    # 1. Resize only for detection
    # ------------------------------
    target_height = 800
    ratio = img.shape[0] / float(target_height)

    resized = cv2.resize(
        img,
        (int(img.shape[1] / ratio), target_height)
    )

    image_area = resized.shape[0] * resized.shape[1]

    # ------------------------------
    # 2. Preprocessing
    # ------------------------------
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    # ------------------------------
    # 3. Find contours
    # ------------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # ------------------------------
    # 4. Smart contour selection
    # ------------------------------
    for c in contours:

        area = cv2.contourArea(c)

        # Reject small rectangles inside note
        if area < 0.15 * image_area:
            continue

        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]

        if w == 0 or h == 0:
            continue

        # Aspect ratio filtering (adjust range if needed)
        aspect_ratio = max(w, h) / min(w, h)

        if aspect_ratio < 1.3 or aspect_ratio > 2.0:
            continue

        # ------------------------------
        # 5. Get rotated bounding box
        # ------------------------------
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        # Scale box back to original resolution
        box *= ratio

        # ------------------------------
        # 6. Perspective correction on ORIGINAL image
        # ------------------------------
        warped = four_point_transform(original, box)

        # Ensure horizontal orientation
        h_warp, w_warp = warped.shape[:2]
        if h_warp > w_warp:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    return None
 
# --- UI Streamlit ---
st.set_page_config(page_title="False 10K_Detect", layout="wide")

col1, col2 = st.columns([10, 1])

with col2:
    language = st.selectbox('Choisissez votre langue \n Select your language :', ['Francais', 'English'])

LANGUAGES = {
    "Francais": {
        "button_label": "Analyser le billet",
        "title": "Détecteur de faux billets (10 000 FCFA)",
        'upload label': "Choisissez une image du billet :",
        'contour fail': "Contour du billet non détecté, veuillez reprendre l'image.",
        'img upload success': "Image Chargée",
        'analysis_spinner': 'Analyse en cours...',
        "success": "Le billet semble authentique.",
        "warning": "Attention : Risque de contrefaçon !"
    },
    "English": {
        "button_label": "Analyze banknote",
        "title": "Counterfeit Note Detector (10,000 FCFA)",
        'upload label': "Upload an image of the bank note :",
        'contour fail': "Banknote contour not detected, Please retake Image.",
        'img upload success': 'Image uploaded successfully',
        'analysis_spinner': 'Analysis in progress...',
        "success": "The note appears to be authentic.",
        "warning": "Warning: High risk of counterfeit!"
    }
}

texts = LANGUAGES[language]

st.title(texts["title"]) 

model_type = "Fine-Tuned ResNet"

uploaded_file = st.file_uploader(texts["upload label"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped = tight_crop_banknote(img_cv)
    
    if cropped is None:
        st.error(texts['contour fail'])
    else:
        # Convert back to RGB for Streamlit display
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        st.image(cropped_rgb, caption=texts['img upload success'], width=400)

        if st.button(texts['button_label']):
            with st.spinner(texts['analysis_spinner']):
                device = torch.device("cpu")
                
                num_classes = 1
                model = models.resnet18(weights=None)
                model.fc = nn.Sequential(
                    nn.Linear(model.fc.in_features, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, num_classes)
                )
                model.load_state_dict(torch.load("banknote_classifier.pth", map_location=device))
                model.to(device)
                model.eval()

                val_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                image = Image.fromarray(cropped_rgb)
                image = val_transform(image).unsqueeze(0)
                image = image.to(device)

                with torch.no_grad():
                    # threshold = st.slider("Seuil de prediction :", 0.0, 1.0, step=0.01)
                    outputs = model(image)
                    prob = torch.sigmoid(outputs).numpy()
                    prediction = (prob > 0.99)
                    if prediction[0] == 1:
                        st.success(texts['success'])
                    else:
                        st.error(texts['warning'])