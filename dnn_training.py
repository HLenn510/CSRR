# DNN 6 Klassen: aluminium, glas, mdf, ohne, polystyrol, s235jr

import glob, re, os, numpy as np
import skrf as rf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from joblib import dump
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/helenabongartz/Documents/GitHub/CSRR_Learning/Training_V6_5"
V6_5_MATERIALS = ["aluminium", "glas", "mdf", "ohne", "polystyrol", "s235jr"]

VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Frequenzraster
fmin, fmax, npts = 0.1e9, 1.5e9, 101
f_target = np.linspace(fmin, fmax, npts)

print(f"Materialklassen: {len(V6_5_MATERIALS)}")

# Daten laden
def load_s2p_simple(path):
    try:
        net = rf.Network(path)
    except Exception:
        # Komma zu Punkt konvertieren
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

#Label aus Dateiname extrahieren
def label_from_path_v6_5(filepath):
    path_parts = filepath.split(os.sep)
    for part in path_parts:
        if part.lower() in V6_5_MATERIALS:
            return part.lower()
    
    filename = os.path.basename(filepath).lower()
    for material in V6_5_MATERIALS:
        if material in filename:
            return material
    return "unbekannt"

# Referenzdatei finden
def find_reference_simple(mat_file, ohne_files):
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

# Features extrahieren
def load_features_simple(mat_path, ref_path):
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

    # Referenz-normierte Features
    if net_r is not None:
        s11_ref = np.abs(net_r.s[:,0,0])
        s21_ref = np.abs(net_r.s[:,1,0])
        s12_ref = np.abs(net_r.s[:,0,1])
        s22_ref = np.abs(net_r.s[:,1,1])
        
        s11_norm = s11_mag / (s11_ref + eps)
        s21_norm = s21_mag / (s21_ref + eps)
        s12_norm = s12_mag / (s12_ref + eps)
        s22_norm = s22_mag / (s22_ref + eps)
    else:
        s11_norm = s11_mag / (np.max(s11_mag) + eps)
        s21_norm = s21_mag / (np.max(s21_mag) + eps)
        s12_norm = s12_mag / (np.max(s12_mag) + eps)
        s22_norm = s22_mag / (np.max(s22_mag) + eps)

    # Gradient Features
    s11_grad = np.gradient(s11_mag)
    s21_grad = np.gradient(s21_mag)
    s12_grad = np.gradient(s12_mag)
    s22_grad = np.gradient(s22_mag)

    # Statistische Features (3 Frequenzbänder)
    band_size = len(s11_mag) // 3
    s11_band1 = np.mean(s11_mag[:band_size])
    s11_band2 = np.mean(s11_mag[band_size:2*band_size])
    s11_band3 = np.mean(s11_mag[2*band_size:])
    s21_band1 = np.mean(s21_mag[:band_size])
    s21_band2 = np.mean(s21_mag[band_size:2*band_size])
    s21_band3 = np.mean(s21_mag[2*band_size:])

    # Normalisierung
    s11_mag = s11_mag / (np.max(s11_mag) + eps)
    s21_mag = s21_mag / (np.max(s21_mag) + eps)
    s12_mag = s12_mag / (np.max(s12_mag) + eps)
    s22_mag = s22_mag / (np.max(s22_mag) + eps)
    
    s11_phase = s11_phase / np.pi
    s21_phase = s21_phase / np.pi
    s12_phase = s12_phase / np.pi
    s22_phase = s22_phase / np.pi
    
    s11_norm = s11_norm / (np.max(s11_norm) + eps)
    s21_norm = s21_norm / (np.max(s21_norm) + eps)
    s12_norm = s12_norm / (np.max(s12_norm) + eps)
    s22_norm = s22_norm / (np.max(s22_norm) + eps)
    
    s11_grad = s11_grad / (np.max(np.abs(s11_grad)) + eps)
    s21_grad = s21_grad / (np.max(np.abs(s21_grad)) + eps)
    s12_grad = s12_grad / (np.max(np.abs(s12_grad)) + eps)
    s22_grad = s22_grad / (np.max(np.abs(s22_grad)) + eps)

    # Kombiniere Features: 16 × 101 + 6 = 1622
    features = np.concatenate([
        s11_mag, s21_mag, s12_mag, s22_mag,           
        s11_phase, s21_phase, s12_phase, s22_phase,   
        s11_norm, s21_norm, s12_norm, s22_norm,       
        s11_grad, s21_grad, s12_grad, s22_grad,       
        [s11_band1, s11_band2, s11_band3, s21_band1, s21_band2, s21_band3]  
    ])
    
    return features

# Daten sammeln

all_files = []
ohne_files = []

for material in V6_5_MATERIALS:
    mat_dir = os.path.join(BASE_DIR, material)
    if os.path.exists(mat_dir):
        files = sorted(glob.glob(os.path.join(mat_dir, "*.s2p")))
        all_files.extend(files)
        print(f"  + {material}: {len(files)} Dateien")
        
        if material == "ohne":
            ohne_files.extend(files)
    else:
        print(f"  - {material}: Ordner nicht gefunden!")

total_files = len(all_files)
print(f"\nGesamtdaten: {total_files} Dateien")

# Features extrahieren
print(f"\n Features extrahieren")
start_time = time.time()
X, y = [], []

for i, f in enumerate(all_files):
    if i % 500 == 0 or i == len(all_files) - 1:
        elapsed = time.time() - start_time
        print(f"  ⏱️ {i+1}/{len(all_files)} ({100*(i+1)/len(all_files):.1f}%) - {elapsed:.1f}s")
    
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

print(f"\nFeatures extrahiert: {len(X)} Samples mit {X.shape[1]} Features")

# Daten aufteilen
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, 
    test_size=VALIDATION_SPLIT, 
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

print(f"\nDatenaufteilung:")
print(f"Training: {X_train.shape[0]} Samples")
print(f"Validation: {X_val.shape[0]} Samples")

# Skalierung der Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Modell erstellen und trainieren
print(f"\nTraining beginnt")

mlp = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
    verbose=False
)

start_training = time.time()
mlp.fit(X_train_scaled, y_train)
training_time = time.time() - start_training

print(f"Training abgeschlossen in {training_time:.1f}s")

# ---- Evaluation --------------------------------------------------------
print(f"\nModell-Evaluation:")

y_val_pred = mlp.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_val_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall
precision, recall, _, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted')
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")

# Confusion Matrix
print(f"\nConfusion Matrix:")
y_val_labels = label_encoder.inverse_transform(y_val)
y_pred_labels = label_encoder.inverse_transform(y_val_pred)
conf_matrix = confusion_matrix(y_val_labels, y_pred_labels, labels=V6_5_MATERIALS)

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
print(f"\nClassification Report:")
print(classification_report(y_val_labels, y_pred_labels))

# Modell speichern
model_filename = "csrr_v6_5_dnn_sklearn_simple.joblib"

model_data = {
    "model": mlp,
    "scaler": scaler,
    "label_encoder": label_encoder,
    "classes": V6_5_MATERIALS,
    "training_info": {
        "total_samples": X.shape[0],
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "num_classes": len(V6_5_MATERIALS),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "features": X.shape[1],
        "training_time": float(training_time)
    }
}

dump(model_data, model_filename)

# Zusammenfassung
print(f"\n" + "="*60)
print(f"DNN Training Abgeschlossen!")
print(f"Daten: {X_train.shape[0]} Training + {X_val.shape[0]} Validation")
print(f"Features: {X.shape[1]} (16 Arrays × 101 + 6 stat)")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"Training: {training_time:.1f}s")
print(f"Gespeichert: {model_filename}")
print("="*60)