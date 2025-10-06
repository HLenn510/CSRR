# CSRR V6_5 DSVM Training Script - Vereinfacht
# Für die 6 Klassen: aluminium, glas, mdf, ohne, polystyrol, s235jr

import glob, os, numpy as np
import skrf as rf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from joblib import dump
import time
import warnings
warnings.filterwarnings('ignore')

print(" CSRR V6_5 DSVM Training - Vereinfacht")
print(" Deep Support Vector Machine")
print("=" * 60)

# ---- Konfiguration --------------------------------------------------------
BASE_DIR = "/Users/helenabongartz/Documents/GitHub/CSRR_Learning/Training_V6_5"
V6_5_MATERIALS = ["aluminium", "glas", "mdf", "ohne", "polystyrol", "s235jr"]

VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Frequenzraster (ohne Interpolation da alle Dateien gleich sind)
fmin, fmax, npts = 0.1e9, 1.5e9, 101
f_target = np.linspace(fmin, fmax, npts)

print(f" Materialklassen: {len(V6_5_MATERIALS)}")

# ---- Vereinfachte Datenlader --------------------------------------------------------
def load_s2p_simple(path):
    """Vereinfachter S2P-Loader ohne Interpolation"""
    try:
        net = rf.Network(path)
    except Exception:
        # Handle european decimal format
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if "," in content:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.s2p', delete=False) as tmp:
                tmp.write(content.replace(",", "."))
                temp_path = tmp.name
            
            try:
                net = rf.Network(temp_path)
            finally:
                os.unlink(temp_path)
        else:
            raise
    
    return net

def label_from_path_v6_5(filepath):
    """Extrahiert Label aus dem Ordnerpfad"""
    path_parts = filepath.split(os.sep)
    for part in path_parts:
        if part.lower() in V6_5_MATERIALS:
            return part.lower()
    
    filename = os.path.basename(filepath).lower()
    for material in V6_5_MATERIALS:
        if material in filename:
            return material
    return "unbekannt"

def find_reference_simple(mat_file, ohne_files):
    """Einfache Referenz-Strategie"""
    import random
    random.seed(RANDOM_STATE)
    
    current_label = label_from_path_v6_5(mat_file)
    
    if current_label == "ohne":
        other_ohne = [f for f in ohne_files if f != mat_file]
        if other_ohne:
            return random.choice(other_ohne)
    else:
        if ohne_files:
            return random.choice(ohne_files)
    
    return None

def load_features_simple(mat_path, ref_path):
    """
    Vereinfachte DSVM Feature-Extraktion:
    - Magnitude: |S11|, |S21|, |S12|, |S22|
    - Phase: arg(S11), arg(S21), arg(S12), arg(S22)
    - Gradienten: erste Ableitung
    
    Total: 12 Arrays × 101 = 1212 Features
    """
    eps = 1e-12
    net_m = load_s2p_simple(mat_path)
    net_r = load_s2p_simple(ref_path) if ref_path else None

    # Magnitude Features
    s11_mag = np.abs(net_m.s[:,0,0])
    s21_mag = np.abs(net_m.s[:,1,0])
    s12_mag = np.abs(net_m.s[:,0,1])
    s22_mag = np.abs(net_m.s[:,1,1])

    # Phase Features
    s11_phase = np.angle(net_m.s[:,0,0])
    s21_phase = np.angle(net_m.s[:,1,0])
    s12_phase = np.angle(net_m.s[:,0,1])
    s22_phase = np.angle(net_m.s[:,1,1])

    # Gradient Features
    s11_grad = np.gradient(s11_mag)
    s21_grad = np.gradient(s21_mag)
    s12_grad = np.gradient(s12_mag)
    s22_grad = np.gradient(s22_mag)

    # Normalisierung
    s11_mag = s11_mag / (np.max(s11_mag) + eps)
    s21_mag = s21_mag / (np.max(s21_mag) + eps)
    s12_mag = s12_mag / (np.max(s12_mag) + eps)
    s22_mag = s22_mag / (np.max(s22_mag) + eps)
    
    s11_phase = s11_phase / np.pi
    s21_phase = s21_phase / np.pi
    s12_phase = s12_phase / np.pi
    s22_phase = s22_phase / np.pi
    
    s11_grad = s11_grad / (np.max(np.abs(s11_grad)) + eps)
    s21_grad = s21_grad / (np.max(np.abs(s21_grad)) + eps)
    s12_grad = s12_grad / (np.max(np.abs(s12_grad)) + eps)
    s22_grad = s22_grad / (np.max(np.abs(s22_grad)) + eps)

    # Kombiniere Features: 12 × 101 = 1212
    features = np.concatenate([
        s11_mag, s21_mag, s12_mag, s22_mag,           # 404 Features
        s11_phase, s21_phase, s12_phase, s22_phase,   # 404 Features
        s11_grad, s21_grad, s12_grad, s22_grad        # 404 Features
    ])
    
    return features

# ---- Daten sammeln --------------------------------------------------------
print(f"\n Sammle Trainingsdaten...")

all_files = []
ohne_files = []

for material in V6_5_MATERIALS:
    mat_dir = os.path.join(BASE_DIR, material)
    if os.path.exists(mat_dir):
        files = sorted(glob.glob(os.path.join(mat_dir, "*.s2p")))
        all_files.extend(files)
        print(f"   {material}: {len(files)} Dateien")
        
        if material == "ohne":
            ohne_files.extend(files)
    else:
        print(f"   {material}: Ordner nicht gefunden!")

total_files = len(all_files)
print(f"\n Gesamtdaten: {total_files} Dateien")

# ---- Feature-Extraktion --------------------------------------------------------
print(f"\n Extrahiere Features...")
start_time = time.time()
X, y = [], []

for i, f in enumerate(all_files):
    if i % 500 == 0 or i == len(all_files) - 1:
        elapsed = time.time() - start_time
        print(f"   {i+1}/{len(all_files)} ({100*(i+1)/len(all_files):.1f}%) - {elapsed:.1f}s")
    
    try:
        ref = find_reference_simple(f, ohne_files)
        feat = load_features_simple(f, ref)
        label = label_from_path_v6_5(f)
        
        if label != "unbekannt":
            X.append(feat)
            y.append(label)
        
    except Exception as e:
        print(f"Fehler bei {os.path.basename(f)}: {e}")
        continue

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n Features extrahiert: {len(X)} Samples mit {X.shape[1]} Features")

# ---- Datenaufteilung --------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=VALIDATION_SPLIT, 
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\n Datenaufteilung:")
print(f"   Training: {X_train.shape[0]} Samples")
print(f"   Validation: {X_val.shape[0]} Samples")

# ---- DSVM Pipeline --------------------------------------------------------
print(f"\n Trainiere Deep Support Vector Machine...")

# Pipeline: StandardScaler → PCA → SVM
dsvm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)),
    ('svc', SVC(kernel='rbf', 
                C=100, 
                gamma='scale',
                class_weight='balanced',
                random_state=RANDOM_STATE))
])

start_training = time.time()
dsvm_pipeline.fit(X_train, y_train)
training_time = time.time() - start_training

print(f" Training abgeschlossen in {training_time:.1f}s")

# ---- Evaluation --------------------------------------------------------
print(f"\n Modell-Evaluation:")

y_val_pred = dsvm_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall
precision, recall, _, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted')
print(f" Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f" Recall: {recall:.4f} ({recall*100:.2f}%)")

# Confusion Matrix
print(f"\n Confusion Matrix:")
conf_matrix = confusion_matrix(y_val, y_val_pred, labels=V6_5_MATERIALS)

print("Predicted\\Actual", end="")
for cls in V6_5_MATERIALS:
    print(f"\t{cls[:8]}", end="")
print()

for i, pred_cls in enumerate(V6_5_MATERIALS):
    print(f"{pred_cls[:12]}", end="")
    for j in range(len(V6_5_MATERIALS)):
        print(f"\t{conf_matrix[i][j]}", end="")
    print()

# Classification Report
print(f"\n Classification Report:")
print(classification_report(y_val, y_val_pred))

# ---- Modell-Information --------------------------------------------------------
print(f"\n Modell-Information:")
pca_components = dsvm_pipeline['pca'].n_components_
pca_variance = dsvm_pipeline['pca'].explained_variance_ratio_.sum()
print(f"   PCA Komponenten: {pca_components}")
print(f"   Erklärte Varianz: {pca_variance:.3f}")

# ---- Modell speichern --------------------------------------------------------
model_filename = "s2p_classifier_v6_5_dsvm_simple.joblib"

model_data = {
    "model": dsvm_pipeline,
    "f_target": f_target,
    "classes": sorted(list(set(y))),
    "materials": V6_5_MATERIALS,
    "training_info": {
        "total_samples": X.shape[0],
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "num_classes": len(V6_5_MATERIALS),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "features": X.shape[1],
        "pca_components": int(pca_components),
        "pca_variance": float(pca_variance),
        "training_time": float(training_time)
    }
}

dump(model_data, model_filename)

# ---- Zusammenfassung --------------------------------------------------------
print(f"\n" + "="*60)
print(f" V6_5 DSVM Training Abgeschlossen!")
print(f" Daten: {X_train.shape[0]} Training + {X_val.shape[0]} Validation")
print(f" Features: {X.shape[1]} (12 Arrays × 101)")
print(f" PCA: {pca_components} Komponenten ({pca_variance:.1%} Varianz)")
print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f" Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f" Training: {training_time:.1f}s")
print(f" Gespeichert: {model_filename}")
print("="*60)