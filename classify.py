import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

DATASET_PATH = r"DATASETS/DATASET 1"
TRAIN_PATH = os.path.join(DATASET_PATH, "TRAIN 1")
TEST_PATH = os.path.join(DATASET_PATH, "TEST 1")
IMG_SIZE = (128, 128)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

def extract_features(image):
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 1. GLCM
    glcm = graycomatrix(gray, [1, 2], [0, np.pi/4], levels=256, symmetric=True, normed=True)
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    f_glcm = [np.mean(graycoprops(glcm, p)) for p in props]
    
    # 2. LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    f_lbp, _ = np.histogram(lbp.ravel(), bins=int(lbp.max()+1), density=True)
    
    # 3. HSV Color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    f_color = []
    for i in range(3):
        f_color.extend([np.mean(hsv[:,:,i]), np.std(hsv[:,:,i])])
        
    return np.concatenate([f_glcm, f_lbp, f_color])

def load_data(path):
    X, y = [], []
    for label, cls in enumerate(['BENIGN', 'MALIGNANT']):
        d = os.path.join(path, cls)
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            try:
                img = cv2.imread(os.path.join(d, f))
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(extract_features(img))
                    y.append(label)
            except: pass
    return np.array(X), np.array(y)

def main():
    print("Loading Data...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)
    
    print("Training RF...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['BENIGN', 'MALIGNANT'])
    
    print(f"Accuracy: {acc}")
    print(report)
    
    with open("classification_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\nReport:\n{report}")
        
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.savefig('confusion_matrix.png')

if __name__ == "__main__": main()