# CSRR V6_5 DSVM Test Script
# Testet das trainierte DSVM Modell mit V6_5 Testdateien

import glob, os, numpy as np
import skrf as rf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from joblib import load
import time
import warnings
warnings.filterwarnings('ignore')

print(" CSRR V6_5 DSVM Model Test - V6_5 Testdateien")
print(" Testet das trainierte Deep SVM Modell")
print("=" * 60)

# ---- Konfiguration --------------------------------------------------------
TEST_DIR = "/Users/helenabongartz/Documents/GitHub/CSRR_Learning/V6_5_Testdateien"
TRAINING_DIR = "/Users/helenabongartz/Documents/GitHub/CSRR_Learning/Training_V6_5"
MODEL_FILE = "s2p_classifier_v6_5_dsvm_simple.joblib"
V6_5_MATERIALS = ["aluminium", "glas", "mdf", "ohne", "polystyrol", "s235jr"]

# ---- Lade trainiertes Modell --------------------------------------------------------
print(f"\n Lade trainiertes DSVM-Modell...")
if not os.path.exists(MODEL_FILE):
    print(f" Modell nicht gefunden: {MODEL_FILE}")
    print("   Führe zuerst das DSVM Training-Skript aus!")
    exit(1)

model_data = load(MODEL_FILE)
model = model_data["model"]

print(f" DSVM-Modell geladen: {MODEL_FILE}")
print(f" Klassen: {model_data['training_info']['num_classes']}")
print(f" Features: {model_data['training_info']['features']}")
print(f" Training Accuracy: {model_data['training_info']['accuracy']:.4f}")
print(f" PCA Komponenten: {model_data['training_info']['pca_components']}")

# ---- Vereinfachte Datenlader (identisch zum Training) --------------------------------------------------------
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

def find_reference_for_test(mat_file, ohne_files):
    """Referenz-Strategie für Test (aus Training-Daten)"""
    import random
    random.seed(42)  # Gleicher Seed wie Training
    
    current_label = label_from_path_v6_5(mat_file)
    
    # Verwende Training-Referenzen
    if current_label == "ohne":
        if ohne_files:
            return random.choice(ohne_files)
    else:
        if ohne_files:
            return random.choice(ohne_files)
    
    return None

def load_features_simple(mat_path, ref_path):
    """
    Identische Feature-Extraktion wie im DSVM Training:
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
        s11_mag, s21_mag, s12_mag, s22_mag,           
        s11_phase, s21_phase, s12_phase, s22_phase,   
        s11_grad, s21_grad, s12_grad, s22_grad        
    ])
    
    return features

# ---- Sammle Referenzdateien aus Training --------------------------------------------------------
print(f"\n Sammle Referenzdateien aus Training")
training_ohne_files = []
ohne_dir = os.path.join(TRAINING_DIR, "ohne")
if os.path.exists(ohne_dir):
    training_ohne_files = sorted(glob.glob(os.path.join(ohne_dir, "*.s2p")))
    print(f"✅ Training-Referenzen gefunden: {len(training_ohne_files)} 'ohne' Dateien")
else:
    print(f" Keine Training-Referenzen gefunden!")

# ---- Sammle Testdaten --------------------------------------------------------

test_files = []
test_counts = {}

for material in V6_5_MATERIALS:
    mat_dir = os.path.join(TEST_DIR, material)
    if os.path.exists(mat_dir):
        files = sorted(glob.glob(os.path.join(mat_dir, "*.s2p")))
        test_files.extend(files)
        test_counts[material] = len(files)
        print(f"   {material}: {len(files)} Testdateien")
    else:
        print(f"   {material}: Ordner nicht gefunden!")
        test_counts[material] = 0

total_test_files = len(test_files)
print(f"\n Testdaten gesamt: {total_test_files} Dateien")

# ---- Feature-Extraktion für Testdaten --------------------------------------------------------
print(f"\n Extrahiere Test-Features...")
start_time = time.time()
X_test, y_test = [], []

for i, f in enumerate(test_files):
    if i % 100 == 0 or i == len(test_files) - 1:
        elapsed = time.time() - start_time
        print(f" {i+1}/{len(test_files)} ({100*(i+1)/len(test_files):.1f}%) - {elapsed:.1f}s")
    
    try:
        ref = find_reference_for_test(f, training_ohne_files)
        feat = load_features_simple(f, ref)
        label = label_from_path_v6_5(f)
        
        if label != "unbekannt":
            X_test.append(feat)
            y_test.append(label)
        
    except Exception as e:
        print(f" Fehler bei {os.path.basename(f)}: {e}")
        continue

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

print(f"\n Test-Features extrahiert: {len(X_test)} Samples mit {X_test.shape[1]} Features")

# ---- Vorhersagen --------------------------------------------------------
print(f"\n Führe DSVM-Vorhersagen durch...")

# Gesamte Vorhersagezeit
start_pred = time.time()
y_pred = model.predict(X_test)
try:
    y_pred_proba = model.predict_proba(X_test)
    has_proba = True
except:
    y_pred_proba = None
    has_proba = False
pred_time = time.time() - start_pred

# Einzelne Datei Inferenzzeit messen
print(f"\n Inferenzzeit-Messung (einzelne Datei)...")
if len(X_test) > 0:
    # Nimm erste Datei für Einzelmessung
    single_sample = X_test[0:1]
    
    # Mehrfache Messungen für genaueren Durchschnitt
    inference_times = []
    for i in range(10):
        start_single = time.time()
        _ = model.predict(single_sample)
        end_single = time.time()
        inference_times.append(end_single - start_single)
    
    avg_inference = np.mean(inference_times)
    min_inference = np.min(inference_times)
    max_inference = np.max(inference_times)
    
    print(f"  Einzeldatei Inferenz (10x gemessen):")
    print(f"  Durchschnitt: {avg_inference*1000:.2f}ms")
    print(f"  Schnellste: {min_inference*1000:.2f}ms") 
    print(f"  Langsamste: {max_inference*1000:.2f}ms")

print(f"\n Vorhersagen abgeschlossen in {pred_time:.3f}s")

# ---- Evaluation (identisch zum Training) --------------------------------------------------------
print(f"\n DSVM-Evaluation auf Testdaten:")

# String Labels für Evaluation verwenden (wie im Training)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f" Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f" Recall: {recall:.4f} ({recall*100:.2f}%)")

# Confusion Matrix
print(f"\n Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred, labels=V6_5_MATERIALS)

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
print(classification_report(y_test, y_pred))

# ---- Per-Klassen Analyse --------------------------------------------------------
print(f"\n Per-Klassen Test-Performance:")
for material in V6_5_MATERIALS:
    if material in y_test:
        mask = (y_test == material)
        correct = np.sum(y_pred[mask] == material)
        total = np.sum(mask)
        accuracy_cls = correct / total if total > 0 else 0
        
        if has_proba and total > 0:
            avg_conf = np.mean(np.max(y_pred_proba[mask], axis=1))
            print(f"   {material.upper()}: {correct}/{total} ({accuracy_cls*100:.1f}%) - Ø Konfidenz: {avg_conf:.3f}")
        else:
            print(f"   {material.upper()}: {correct}/{total} ({accuracy_cls*100:.1f}%)")

# ---- DSVM-spezifische Information --------------------------------------------------------
print(f"\n🔬 DSVM-Modell Information:")
try:
    pca_components = model['pca'].n_components_
    pca_variance = model['pca'].explained_variance_ratio_.sum()
    svm_C = model['svc'].C
    svm_kernel = model['svc'].kernel
    
    print(f"  PCA Komponenten: {pca_components}")
    print(f"  Erklärte Varianz: {pca_variance:.3f}")
    print(f"  SVM C-Parameter: {svm_C}")
    print(f"  SVM Kernel: {svm_kernel}")
except Exception as e:
    print(f"  Modell-Info nicht verfügbar: {e}")

# ---- Zusammenfassung --------------------------------------------------------
print(f"\n" + "="*60)
print(f" V6_5 DSVM Test Abgeschlossen!")
print(f" Testdaten: {len(X_test)} Samples")
print(f" Features: {X_test.shape[1]} (identisch zum Training)")
print(f" Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" Test Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f" Test Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f" Batch-Vorhersagezeit: {pred_time:.3f}s ({pred_time*1000/len(X_test):.1f}ms/Sample)")
if len(X_test) > 0 and 'avg_inference' in locals():
    print(f" Einzeldatei-Inferenz: {avg_inference*1000:.2f}ms (Ø aus 10 Messungen)")
    samples_per_sec = 1.0 / avg_inference
    print(f" Durchsatz: {samples_per_sec:.1f} Dateien/Sekunde")

# Vergleich mit Training
train_acc = model_data['training_info']['accuracy']
diff = accuracy - train_acc
if abs(diff) < 0.02:
    print(f"Δ = {diff:+.4f}")
elif diff < -0.05:
    print(f"Δ = {diff:+.4f}")
else:
    print(f"(Δ = {diff:+.4f}")

print("="*60)