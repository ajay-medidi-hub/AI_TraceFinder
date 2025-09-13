# feature_extractor.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis, entropy

# 🔹 Streamlit page setup
st.set_page_config(page_title="📊 Dataset Feature Extractor", layout="wide")
st.title("📊 Image Dataset Feature Extractor")

# --- Feature extractor function ---
def extract_features(image_path, class_label):
    try:
        # Load image using PIL for better handling of various formats
        img_pil = Image.open(image_path)
        img_np = np.array(img_pil)

        # Handle potential empty or invalid image files
        if img_np.size == 0:
            return {
                "file_name": os.path.basename(image_path),
                "class": class_label,
                "error": "Unreadable file or empty image"
            }

        # Handle different channel counts (RGB, B&W, etc.)
        if len(img_np.shape) == 2:
            # Grayscale image
            gray = img_np
            img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # RGB image
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            return {
                "file_name": os.path.basename(image_path),
                "class": class_label,
                "error": "Unsupported image format"
            }

        # Basic image metadata
        file_name = os.path.basename(image_path)

        # Dimensions
        height, width = gray.shape
        aspect_ratio = round(width / height, 3)

        # File size
        file_size = os.path.getsize(image_path) / 1024  # KB

        # Intensity stats (from grayscale)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        # Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        shannon_entropy = entropy(hist.flatten() + 1e-9)

        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        # Texture
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Color stats (from BGR)
        mean_b, mean_g, mean_r = cv2.mean(img_bgr)[:3]

        return {
            "file_name": file_name,
            "class": class_label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3),
            "laplacian_var": round(laplacian_var, 3),
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2)
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "error": str(e)
        }

# --- UI for dataset path ---
dataset_root = st.text_input(
    "📂 Enter dataset root path:",
"C:/Users/dhara/OneDrive/Desktop/Infosys_ws/AI_TraceFinder/data-tifs-2016-maps/data-tifs-2016-maps")

if dataset_root and os.path.isdir(dataset_root):
    st.info("🔎 Scanning dataset...")
    records = []

    classes = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]

    if classes:
        st.success(f"Detected {len(classes)} classes: {classes}")
        for class_dir in classes:
            class_path = os.path.join(dataset_root, class_dir)
            files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
            st.write(f"Class '{class_dir}' → {len(files)} images")
            for fname in files:
                path = os.path.join(class_path, fname)
                rec = extract_features(path, class_dir)
                records.append(rec)
    else:
        st.warning("No subdirectories found. Assuming the root folder contains all images.")
        files = [f for f in os.listdir(dataset_root) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        st.write(f"Root folder → {len(files)} images")
        for fname in files:
            path = os.path.join(dataset_root, fname)
            rec = extract_features(path, "root")
            records.append(rec)

    if records:
        df = pd.DataFrame(records)
        st.subheader("📑 Features Extracted (Preview)")
        st.dataframe(df.head(20))

        # --- Save CSV in the same AI_TraceFinder folder ---
        output_folder = "C:/Users/dhara/OneDrive/Desktop/Infosys_ws/AI_TraceFinder"
        os.makedirs(output_folder, exist_ok=True)
        dataset_name = os.path.basename(dataset_root.rstrip("/\\"))
        save_path = os.path.join(output_folder, f"{dataset_name}_features.csv")
        df.to_csv(save_path, index=False)
        st.success(f"✅ Features saved to {save_path}")

        # --- Class distribution ---
        if "class" in df.columns:
            st.subheader("📊 Class Distribution")
            st.bar_chart(df["class"].value_counts())
    else:
        st.warning("No valid images found in the specified path.")

elif dataset_root:
    st.error("❌ Invalid dataset path. Please enter a valid folder.")