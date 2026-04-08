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
col1, col2 = st.columns([8, 1])
with col2:
    lang = st.radio("🌐 Langue / Language", ["Français", "English"], horizontal=True)

LANGUAGES = {
    "Français": {
        "sidebar_title": "À propos de l'auteur",
        "role": "**Data Scientist & AI Engineer**",
        "bio": "Passionné par l'intersection de l'IA et de la finance. Ce projet démontre l'application de la **Vision par ordinateur** pour sécuriser les transactions en zone CEMAC.",
        "linkedin_btn": "Mon Profil LinkedIn",
        "lang_label": "Choisir la langue",
        "button_label": "Analyser le billet",
        "title": "Détecteur de faux billets (10 000 FCFA)",
        "upload_label": "Choisissez une image du billet :",
        "contour_fail": "Contour du billet non détecté, veuillez reprendre l'image.",
        "img_upload_success": "Image Chargée",
        "analysis_spinner": "Analyse en cours...",
        "success": ["Le billet semble authentique.", "Confiance"],
        "warning": "Attention : Risque de contrefaçon !"
    },
    "English": {
        "sidebar_title": "About the Author",
        "role": "**Data Scientist & AI Engineer**",
        "bio": "Passionate about the intersection of AI and Finance. This project demonstrates the application of **Computer Vision** to secure transactions in the CEMAC zone.",
        "linkedin_btn": "My LinkedIn Profile",
        "lang_label": "Choose Language",
        "button_label": "Analyze banknote",
        "title": "Counterfeit Note Detector (10,000 FCFA)",
        "upload_label": "Upload an image of the bank note:",
        "contour_fail": "Banknote contour not detected, please retake image.",
        "img_upload_success": "Image uploaded successfully",
        "analysis_spinner": "Analysis in progress...",
        "success": ["The note appears to be authentic.", "Confidence"],
        "warning": "Warning: High risk of counterfeit!"
    }
}

texts = LANGUAGES[lang]

with st.sidebar:
    st.image("pp_.jpeg", width=150)
    st.title(texts["sidebar_title"])
    st.markdown(f"""
    ## **Marcel MBOUA**
    {texts["role"]}  
    *Quant Enthusiast*
    
    ---
    {texts["bio"]}
    """)
    
    linkedin_url = "https://linkedin.com/in/marcel-mboua-285882237"
    st.markdown(f'''
        <a href="{linkedin_url}" target="_blank" style="text-decoration: none;">
            <button style="width: 100%; background-color: #0077B5; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                {texts["linkedin_btn"]}
            </button>
        </a>
    ''', unsafe_allow_html=True)

st.title(texts["title"]) 

model_type = "Fine-Tuned ResNet"

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    num_classes = 1
    # On définit l'architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes)
    )
    # Chargement des poids
    model.load_state_dict(torch.load("banknote_classifier.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Appel de la fonction de chargement
model, device = load_model()

uploaded_file = st.file_uploader(texts["upload_label"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped = tight_crop_banknote(img_cv)
    
    if cropped is None:
        st.error(texts['contour_fail'])
    else:
        # Convert back to RGB for Streamlit display
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        st.image(cropped_rgb, caption=texts['img_upload_success'], width=400)

        if st.button(texts['button_label']):
            with st.spinner(texts['analysis_spinner']):

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
                    prob = torch.sigmoid(outputs).cpu().item()
                    if prob > 0.99:
                        st.success(f"{texts['success'][0]} ({texts['success'][1]}: {prob:.2%})")
                    else:
                        st.error(f"{texts['warning']} (Score: {prob:.2%})")