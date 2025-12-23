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
import sys

# Configuration
DATASET_PATH = r"DATASETS/DATASET 1"
TRAIN_PATH = os.path.join(DATASET_PATH, "TRAIN 1")
TEST_PATH = os.path.join(DATASET_PATH, "TEST 1")
IMG_SIZE = (128, 128)
LOG_DIR = "preprocessing_logs"
TRAIN_LOG_FILE = "training_log.txt"
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# Ensure log directory exists and is clean
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)

# Reset training log
with open(TRAIN_LOG_FILE, "w") as f:
    f.write("--- Training Log ---\n")

def log_print(message):
    """Prints to console and appends to training log file."""
    print(message)
    with open(TRAIN_LOG_FILE, "a") as f:
        f.write(message + "\n")

def save_preprocessing_steps(img_name, label, original, resized, gray, blurred, final):
    """Saves visualization of preprocessing steps for a few samples."""
    class_dir = os.path.join(LOG_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        
    if len(os.listdir(class_dir)) >= 5: 
        return

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ["Original", "Resized", "Grayscale", "Gaussian Blur", "CLAHE (Final)"]
    images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), 
              cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 
              gray, blurred, final]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(class_dir, f"{img_name}_steps.png"))
    plt.close()

def log_feature_extraction(img_name, label, feat_glcm, feat_lbp, feat_color):
    """Logs sample feature values to a text file in the log dir."""
    log_file = os.path.join(LOG_DIR, "feature_extraction_log.txt")
    with open(log_file, "a") as f:
        f.write(f"\n--- Image: {img_name} ({label}) ---\n")
        f.write(f"GLCM Features (Total {len(feat_glcm)}): {feat_glcm[:5]} ...\n")
        f.write(f"LBP Features (Total {len(feat_lbp)}): {feat_lbp[:5]} ...\n")
        f.write(f"Color Features (Total {len(feat_color)}): {feat_color}\n")

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        channel = hsv[:, :, i].ravel()
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(skew(channel))
        features.append(kurtosis(channel))
    return np.array(features)

def extract_lbp_features(gray_image):
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_glcm_features(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    distances = [1, 2, 3] 
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        val = graycoprops(glcm, prop)
        for i in range(len(distances)):
            row = val[i, :]
            features.append(np.mean(row))
            features.append(np.ptp(row))
    return np.array(features)

def load_data(folder_path, is_training=False):
    X = []
    y = []
    classes = ['BENIGN', 'MALIGNANT']
    log_print(f"Loading data from {folder_path}...")
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir):
            continue
        files = os.listdir(class_dir)
        log_print(f"  Processing {class_name}: {len(files)} images found.")
        for i, img_name in enumerate(files):
            img_path = os.path.join(class_dir, img_name)
            try:
                original = cv2.imread(img_path)
                if original is None: continue
                resized = cv2.resize(original, IMG_SIZE)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                final_img = clahe_obj.apply(blurred)
                
                feat_glcm = extract_glcm_features(final_img)
                feat_lbp = extract_lbp_features(final_img)
                feat_color = extract_color_features(resized)
                
                if is_training and i < 5: 
                     save_preprocessing_steps(img_name, class_name, original, resized, gray, blurred, final_img)
                     log_feature_extraction(img_name, class_name, feat_glcm, feat_lbp, feat_color)

                combined_features = np.concatenate([feat_glcm, feat_lbp, feat_color])
                X.append(combined_features)
                y.append(label)
            except Exception as e:
                log_print(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

def main():
    log_print("--- Starting Classification Pipeline ---")
    
    log_print("\n[Step 1] Loading and Preprocessing Data...")
    X_train, y_train = load_data(TRAIN_PATH, is_training=True)
    X_test, y_test = load_data(TEST_PATH)
    
    log_print(f"\nTraining Data Shape: {X_train.shape}")
    log_print(f"Testing Data Shape: {X_test.shape}")
    
    log_print("\n[Step 2] Feature Scaling skipped for Random Forest.")
    
    log_print("\n[Step 3] Training Random Forest Classifier with Grid Search...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # We can't easily capture GridSearchCV stdout to file without redirecting sys.stdout
    # So we will rely on logging the results after fitting.
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=1, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    log_print(f"\nBest Parameters found: {grid.best_params_}")
    log_print(f"Best Cross-Validation Score: {grid.best_score_:.4f}")
    
    clf = grid.best_estimator_
    
    # Feature Importance
    if hasattr(clf, 'feature_importances_'):
        log_print("\nTop 15 Feature Importances:")
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(min(15, X_train.shape[1])):
            log_print(f"{f+1}. Feature {indices[f]} ({importances[indices[f]]:.4f})")

    log_print("\n[Step 4] Predicting on Test Set...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    log_print("\n--- Classification Results ---")
    log_print(f"Accuracy: {acc:.4f}")
    log_print(f"Sensitivity (Recall): {sensitivity:.4f}")
    log_print(f"Specificity: {specificity:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=['BENIGN', 'MALIGNANT'])
    log_print("\nClassification Report:\n" + report)
    
    with open("classification_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Best Params: {grid.best_params_}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nReport:\n")
        f.write(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALIGNANT'], yticklabels=['BENIGN', 'MALIGNANT'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig('confusion_matrix.png')
    log_print("Confusion matrix saved to confusion_matrix.png")
    log_print(f"\nPreprocessing logs saved to {LOG_DIR}")
    log_print(f"Training log saved to {TRAIN_LOG_FILE}")

if __name__ == "__main__":
    main()