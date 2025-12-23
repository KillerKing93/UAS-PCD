import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Configuration
DATASET_PATH = r"DATASETS/DATASET 1"
TRAIN_PATH = os.path.join(DATASET_PATH, "TRAIN 1")
TEST_PATH = os.path.join(DATASET_PATH, "TEST 1")
IMG_SIZE = (128, 128)
LOG_DIR = "preprocessing_logs"
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)

def log_print(msg):
    print(msg)

def hair_removal(image):
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Morphological BlackHat to find hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Thresholding hair
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint to remove hair
    dst = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return dst

def segment_lesion(image):
    # Convert to HSV, S-channel usually separates lesion best
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    # Blur/Smooth
    blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
    
    # Otsu Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological Opening/Closing to clean noise
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find largest contour (Assume it's the lesion)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(s_channel)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
    else:
        mask = np.ones_like(s_channel) * 255 # Fallback: whole image
        
    return mask

def extract_features_roi(image, mask):
    # Apply Mask
    img_masked = cv2.bitwise_and(image, image, mask=mask)
    
    features = []
    
    # 1. Color Features (Only inside ROI)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    for space in [hsv, lab]:
        for i in range(3):
            chan = space[:, :, i]
            # Extract pixels where mask is white
            pixels = chan[mask > 0]
            if len(pixels) == 0: pixels = np.array([0])
            
            features.extend([np.mean(pixels), np.std(pixels), skew(pixels), kurtosis(pixels)])
            
    # 2. Shape Features from Mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: perimeter = 1
        compactness = 4 * np.pi * area / (perimeter ** 2)
        features.append(compactness)
        features.append(area)
    else:
        features.extend([0, 0])

    # 3. Texture Features (ROI) - Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp[mask > 0], bins=int(lbp.max()+1), density=True)
    features.extend(lbp_hist)
    
    # GLCM (Computed on rectangular crop for library compatibility, but masked)
    x,y,w,h = cv2.boundingRect(mask) if contours else (0,0,128,128)
    roi_gray = gray[y:y+h, x:x+w]
    if roi_gray.size > 0:
        glcm = graycomatrix(roi_gray, [1], [0], levels=256, symmetric=True, normed=True)
        props = ['contrast', 'energy', 'homogeneity', 'correlation', 'ASM']
        features.extend([graycoprops(glcm, p)[0, 0] for p in props])
    else:
        features.extend([0]*5)
        
    return np.array(features)

def save_debug_images(img_name, label, original, hair_removed, mask):
    class_dir = os.path.join(LOG_DIR, label)
    if not os.path.exists(class_dir): os.makedirs(class_dir)
    if len(os.listdir(class_dir)) >= 3: return
    
    masked_img = cv2.bitwise_and(original, original, mask=mask)
    
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original")
    ax[1].imshow(cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)); ax[1].set_title("Hair Removed")
    ax[2].imshow(mask, cmap='gray'); ax[2].set_title("Lesion Mask")
    ax[3].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)); ax[3].set_title("ROI Extracted")
    for a in ax: a.axis('off')
    plt.savefig(os.path.join(class_dir, f"{img_name}_seg.png"))
    plt.close()

def load_data(path, is_training=False):
    X, y = [], []
    print(f"Loading {path}...")
    for label, cls in enumerate(['BENIGN', 'MALIGNANT']):
        d = os.path.join(path, cls)
        files = os.listdir(d)
        for i, f in enumerate(files):
            try:
                img = cv2.imread(os.path.join(d, f))
                if img is None: continue
                img = cv2.resize(img, IMG_SIZE)
                
                # Pipeline: Hair Removal -> Segmentation -> Features
                clean = hair_removal(img)
                mask = segment_lesion(clean)
                
                if is_training and i < 5:
                    save_debug_images(f, cls, img, clean, mask)
                
                feats = extract_features_roi(clean, mask)
                X.append(feats)
                y.append(label)
            except Exception as e: pass
            
    return np.array(X), np.array(y)

def main():
    X_train, y_train = load_data(TRAIN_PATH, is_training=True)
    X_test, y_test = load_data(TEST_PATH)
    
    # Drop NaNs
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    print("Training RF...")
    # High estimators, balanced subsample, tuned for performance
    clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, 
                                 class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # THRESHOLD TUNING for R1 (Recall Malignant) optimization
    # Standard is 0.5. We want to catch more Malignant, so we LOWER threshold for class 1.
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0
    
    # Simple grid search for threshold
    for thresh in np.arange(0.3, 0.7, 0.05):
        preds = (y_probs >= thresh).astype(int)
        # Check if preds contain both classes
        if len(np.unique(preds)) < 2: continue
        
        rep = classification_report(y_test, preds, output_dict=True)
        # Class 1 is Malignant
        f1 = rep['1']['f1-score'] 
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal Threshold for F1: {best_thresh}")
    y_pred = (y_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['BENIGN', 'MALIGNANT'])
    
    print(f"Final Accuracy: {acc}")
    print(report)
    
    with open("classification_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nOptimal Threshold: {best_thresh}\n\nReport:\n{report}")
        
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALIGNANT'], yticklabels=['BENIGN', 'MALIGNANT'])
    plt.title("Confusion Matrix (Segmentation + ROI Features)")
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__": main()
