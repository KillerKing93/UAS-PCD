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

# Configuration
DATASET_PATH = r"DATASETS/DATASET 1"
TRAIN_PATH = os.path.join(DATASET_PATH, "TRAIN 1")
TEST_PATH = os.path.join(DATASET_PATH, "TEST 1")
IMG_SIZE = (128, 128)
LOG_DIR = "preprocessing_logs"
TRAIN_LOG_FILE = "training_log.txt"
FEATURE_LOG_FILE = "feature_extraction_log.txt"
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# Ensure log directory exists and is clean
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)

# Reset logs
with open(TRAIN_LOG_FILE, "w") as f:
    f.write("--- Training Log ---\n")
with open(FEATURE_LOG_FILE, "w") as f:
    f.write("--- Feature Extraction Log (Sample) ---\n")

def log_print(message):
    print(message)
    with open(TRAIN_LOG_FILE, "a") as f:
        f.write(message + "\n")

def log_features(img_name, label, glcm, lbp, color, hu):
    with open(FEATURE_LOG_FILE, "a") as f:
        f.write(f"\n[Image: {img_name} | Class: {label}]\n")
        f.write(f"  > GLCM (Texture): {glcm[:4]}... (Total {len(glcm)})\n")
        f.write(f"  > LBP (Micro-Texture): {lbp[:4]}... (Total {len(lbp)})\n")
        f.write(f"  > Color Hist (HSV): {color[:4]}... (Total {len(color)})\n")
        f.write(f"  > Hu Moments (Shape): {hu}\n")

def save_preprocessing_steps(img_name, label, original, resized, gray, blurred, final):
    class_dir = os.path.join(LOG_DIR, label)
    if not os.path.exists(class_dir): os.makedirs(class_dir)
    if len(os.listdir(class_dir)) >= 5: return
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ["Original", "Resized", "Grayscale", "Gaussian Blur", "CLAHE (Final)"]
    images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), gray, blurred, final]
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2: ax.imshow(img, cmap='gray')
        else: ax.imshow(img)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(class_dir, f"{img_name}_steps.png"))
    plt.close()

def augment_image(image):
    # Augmentation: Original + Flip H (2x Data) - Optimized for speed
    augmented = [image]
    augmented.append(cv2.flip(image, 1)) 
    return augmented

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    # Moments (Mean, Std, Skew, Kurtosis)
    for i in range(3):
        channel = hsv[:, :, i].ravel()
        features.extend([np.mean(channel), np.std(channel), skew(channel), kurtosis(channel)])
    # Histogram (8 bins per channel) - Normalized
    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    features.extend(h_hist / (np.sum(h_hist) + 1e-7))
    features.extend(s_hist / (np.sum(s_hist) + 1e-7))
    features.extend(v_hist / (np.sum(v_hist) + 1e-7))
    return np.array(features)

def extract_hu_moments(gray_image):
    # Otsu Thresholding for shape binary mask
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    # Log scale transform to handle small ranges
    for i in range(7):
        if hu[i] != 0: hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))
        else: hu[i] = 0
    return hu

def extract_lbp_features(gray_image):
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_glcm_features(image):
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    distances = [1, 2, 3] 
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        val = graycoprops(glcm, prop)
        for i in range(len(distances)):
            row = val[i, :]
            features.extend([np.mean(row), np.ptp(row)])
    return np.array(features)

def get_full_features(original_bgr, final_gray):
    feat_glcm = extract_glcm_features(final_gray)
    feat_lbp = extract_lbp_features(final_gray)
    feat_hu = extract_hu_moments(final_gray)
    feat_color = extract_color_features(original_bgr)
    
    combined = np.concatenate([feat_glcm, feat_lbp, feat_hu, feat_color])
    return combined, feat_glcm, feat_lbp, feat_color, feat_hu

def load_data(folder_path, is_training=False):
    X, y = [], []
    classes = ['BENIGN', 'MALIGNANT']
    log_print(f"Loading data from {folder_path}...")
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir): continue
        files = os.listdir(class_dir)
        log_print(f"  Processing {class_name}: {len(files)} images found.")
        for i, img_name in enumerate(files):
            img_path = os.path.join(class_dir, img_name)
            try:
                original = cv2.imread(img_path)
                if original is None: continue
                
                # Resize first
                resized = cv2.resize(original, IMG_SIZE)
                
                # Augmentation only for training
                imgs_to_process = augment_image(resized) if is_training else [resized]
                
                for j, img in enumerate(imgs_to_process):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blurred)
                    
                    full_feat, f_glcm, f_lbp, f_color, f_hu = get_full_features(img, clahe)
                    
                    # Logging
                    if is_training and i < 5 and j == 0:
                        save_preprocessing_steps(img_name, class_name, img, img, gray, blurred, clahe)
                        log_features(img_name, class_name, f_glcm, f_lbp, f_color, f_hu)
                    
                    X.append(full_feat)
                    y.append(label)
            except Exception as e: log_print(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

def main():
    log_print("--- Starting Enhanced Classification Pipeline (RF + Augmentation 4x) ---")
    
    # 1. Load Data
    X_train, y_train = load_data(TRAIN_PATH, is_training=True)
    X_test, y_test = load_data(TEST_PATH)
    
    log_print(f"Training Data Shape (Augmented): {X_train.shape}")
    log_print(f"Testing Data Shape: {X_test.shape}")
    
    # 2. Train Random Forest
    log_print("\n[Step 3] Training Random Forest Classifier...")
    # Optimized Grid to force generalization (prevent overfitting)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [None, 20],
        'min_samples_leaf': [2, 4], # Pruning
        'class_weight': ['balanced'], # Handle class imbalance
        'max_features': ['sqrt', 'log2'],
        'n_jobs': [-1] 
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=2, cv=3)
    grid.fit(X_train, y_train)
    
    log_print(f"Best Parameters: {grid.best_params_}")
    log_print(f"Best Cross-Validation Score: {grid.best_score_:.4f}")
    
    clf = grid.best_estimator_
    y_pred = clf.predict(X_test)
    
    # 3. Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    log_print(f"\n--- Results ---\nAccuracy: {acc:.4f}\nSensitivity: {sensitivity:.4f}\nSpecificity: {specificity:.4f}")
    report = classification_report(y_test, y_pred, target_names=['BENIGN', 'MALIGNANT'])
    log_print("\nReport:" + report)
    
    # Save Results
    with open("classification_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nSensitivity: {sensitivity:.4f}\nSpecificity: {specificity:.4f}\n")
        f.write(f"Best Params: {grid.best_params_}\n")
        f.write(f"Confusion Matrix:\n{cm}\n\nReport:\n{report}")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALIGNANT'], yticklabels=['BENIGN', 'MALIGNANT'])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix - Random Forest Enhanced')
    plt.savefig('confusion_matrix.png')
    log_print("Confusion matrix saved. Pipeline complete.")

if __name__ == "__main__": main()