import os
import cv2
import numpy as np
from skimage.feature import (
    local_binary_pattern,
    hog,
    graycomatrix,
    graycoprops
)
import pandas as pd

DATASET_PATH = "dataset/train"
OUTPUT_FILE = "features.csv"


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Impossible de lire l'image")
    img = cv2.resize(img, (384, 384))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. HOG FEATURES (formes / contours)
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)

    # 2. LBP FEATURES (texture locale)
    lbp = local_binary_pattern(gray, P=8, R=1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # 3. RGB HISTOGRAMS
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
    color_hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
    color_hist /= (color_hist.sum() + 1e-6)

    # 4. HSV HISTOGRAMS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hsv_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hsv_hist /= (hsv_hist.sum() + 1e-6)

    # 5. GLCM FEATURES (texture avancée) utile pour la répétiton de motif
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    glcm_features = np.array([contrast, energy, homogeneity, correlation])

    # 6. HU MOMENTS (forme globale)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # transformation logarithmique
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # 7. CANNY FEATURES (densité de contours)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    canny_features = np.array([edge_density])

    # Concaténation de toutes les features
    features = np.concatenate([
        hog_features,
        lbp_hist,
        color_hist,
        hsv_hist,
        glcm_features,
        hu_moments,
        canny_features
    ])

    return features


X = []
y = []

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Traitement : {class_name}")

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        try:
            features = extract_features(img_path)

            X.append(features)
            y.append(class_name)

        except Exception as e:
            print(f"Erreur avec {img_path} : {e}")

X = np.array(X)
y = np.array(y)

print("Shape des features :", X.shape)

df = pd.DataFrame(X)

df["label"] = y

df.to_csv(OUTPUT_FILE, index=False)

print("Features extraites et sauvegardées dans", OUTPUT_FILE)