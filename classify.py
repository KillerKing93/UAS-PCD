import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_PATH = r"DATASETS/DATASET 1"
TRAIN_PATH = os.path.join(DATASET_PATH, "TRAIN 1")
TEST_PATH = os.path.join(DATASET_PATH, "TEST 1")
IMG_SIZE = (128, 128)

def extract_glcm_features(image):
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate GLCM
    # Distances: 1 pixel. Angles: 0, 45, 90, 135 degrees
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    # Extract properties
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        # Calculate property and flatten (average across angles or keep separate? Let's flatten/mean for simplicity and rotation invariance)
        val = graycoprops(glcm, prop)
        features.append(np.mean(val)) # Mean across all angles
        
    return np.array(features)

def load_data(folder_path):
    X = []
    y = []
    classes = ['BENIGN', 'MALIGNANT']
    
    print(f"Loading data from {folder_path}...")
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Preprocessing
                img = cv2.resize(img, IMG_SIZE)
                
                # Feature Extraction
                features = extract_glcm_features(img)
                
                X.append(features)
                y.append(label) # 0 for BENIGN, 1 for MALIGNANT
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(X), np.array(y)

def main():
    # 1. Load Data
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)
    
    print(f"Training Data: {X_train.shape[0]} samples")
    print(f"Testing Data: {X_test.shape[0]} samples")
    
    # 2. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train Classifier (SVM)
    print("Training SVM Classifier...")
    clf = SVC(kernel='rbf', C=1.0, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 4. Prediction
    y_pred = clf.predict(X_test_scaled)
    
    # 5. Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate Sensitivity (Recall) and Specificity
    # TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n--- Classification Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Save Classification Report
    report = classification_report(y_test, y_pred, target_names=['BENIGN', 'MALIGNANT'])
    print("\nClassification Report:\n", report)
    
    # Save Results to File
    with open("classification_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nReport:\n")
        f.write(report)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALIGNANT'], yticklabels=['BENIGN', 'MALIGNANT'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - SVM with GLCM')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    main()
