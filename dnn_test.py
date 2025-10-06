# CSRR V6_5 Deep Neural Network Test Script (Scikit-learn MLP)
# Testet das trainierte DNN-Modell mit unabhängigen Testdaten
# Modell: csrr_v6_5_dnn_sklearn.joblib

import glob, os, numpy as np
import skrf as rf
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

print(" CSRR V6_5 Deep Neural Network Test - Modell Validierung")
print(" Testet: csrr_v6_5_dnn_sklearn.joblib")
print("=" * 70)

# ---- Konfiguration --------------------------------------------------------
TEST_DIR = "/Users/helenabongartz/Documents/GitHub/CSRR_Learning/V6_5_Testdateien"
MODEL_FILE = "csrr_v6_5_dnn_sklearn.joblib"
V6_5_MATERIALS = ["aluminium", "glas", "mdf", "ohne", "polystyrol", "s235jr"]

print(f" Test-Verzeichnis: {TEST_DIR}")
print(f" Modell-Datei: {MODEL_FILE}")

# ---- Modell laden --------------------------------------------------------
print(f"\n Lade trainiertes DNN-Modell...")

try:
    model_data = load(MODEL_FILE)
    
    # Modell-Komponenten extrahieren
    mlp_model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoder = model_data["label_encoder"]
    f_target = model_data["f_target"]
    classes = model_data["classes"]
    feature_info = model_data["feature_info"]
    training_info = model_data["training_info"]
    
    print(f" Modell erfolgreich geladen!")
    print(f" Architektur: {training_info['architecture']}")
    print(f" Features: {feature_info['dimensions']}")
    print(f" Klassen: {len(classes)}")
    print(f" Training Accuracy: {training_info['val_accuracy']:.4f}")
    
except FileNotFoundError:
    print(f" Modell-Datei nicht gefunden: {MODEL_FILE}")
    print(f" Führe zuerst training_v6_5_dnn_sklearn.py aus!")
    exit(1)
except Exception as e:
    print(f" Fehler beim Laden des Modells: {e}")
    exit(1)

# ---- Feature-Extraktion (IDENTISCH zum Training) --------------------------------------------------------
def load_net_on_grid_safe(path):
    """Lädt s2p-Datei und interpoliert auf Ziel-Frequenzraster"""
    try:
        net = rf.Network(path)
    except Exception:
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
    
    return net.interpolate(f_target, kind='linear')

def label_from_path_v6_5(filepath):
    """Extrahiert Label aus dem Ordnerpfad für V6_5-Struktur"""
    path_parts = filepath.split(os.sep)
    for part in path_parts:
        if part.lower() in V6_5_MATERIALS:
            return part.lower()
    
    filename = os.path.basename(filepath).lower()
    for material in V6_5_MATERIALS:
        if material in filename:
            return material
    return "unbekannt"

def find_reference_for_v6_5(mat_file, all_files, ohne_files):
    """Erweiterte Referenz-Strategie für V6_5"""
    import random
    random.seed(42)  # Gleicher Seed wie im Training
    
    current_label = label_from_path_v6_5(mat_file)
    
    if current_label == "ohne":
        other_ohne = [f for f in ohne_files if f != mat_file]
        if other_ohne:
            return random.choice(other_ohne)
    else:
        if ohne_files:
            return random.choice(ohne_files)
    
    other_files = [f for f in all_files if f != mat_file]
    if other_files:
        return random.choice(other_files)
    return None

def load_features_v6_5_dnn(mat_path, ref_path):
    """
    IDENTISCHE Feature-Extraktion wie im Training!
    16 Arrays × 101 Frequenzpunkte + 6 statistische Features = 1622 Features
    """
    eps = 1e-12
    net_m = load_net_on_grid_safe(mat_path)
    net_r = load_net_on_grid_safe(ref_path) if ref_path else None

    # === MAGNITUDE FEATURES ===
    s11_mag = np.abs(net_m.s[:,0,0])  # |S11|
    s21_mag = np.abs(net_m.s[:,1,0])  # |S21|
    s12_mag = np.abs(net_m.s[:,0,1])  # |S12|  
    s22_mag = np.abs(net_m.s[:,1,1])  # |S22|

    # === PHASE FEATURES ===
    s11_phase = np.angle(net_m.s[:,0,0])  # arg(S11)
    s21_phase = np.angle(net_m.s[:,1,0])  # arg(S21)
    s12_phase = np.angle(net_m.s[:,0,1])  # arg(S12)
    s22_phase = np.angle(net_m.s[:,1,1])  # arg(S22)

    # === REFERENZ-NORMIERTE FEATURES ===
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

    # === FREQUENZ-GRADIENT FEATURES ===
    s11_grad = np.gradient(s11_mag)
    s21_grad = np.gradient(s21_mag)
    s12_grad = np.gradient(s12_mag)
    s22_grad = np.gradient(s22_mag)

    # === ZUSÄTZLICHE STATISTISCHE FEATURES ===
    band_size = len(s11_mag) // 3  # Teile in 3 Bänder
    
    s11_band1 = np.mean(s11_mag[:band_size])
    s11_band2 = np.mean(s11_mag[band_size:2*band_size])
    s11_band3 = np.mean(s11_mag[2*band_size:])
    
    s21_band1 = np.mean(s21_mag[:band_size])
    s21_band2 = np.mean(s21_mag[band_size:2*band_size])
    s21_band3 = np.mean(s21_mag[2*band_size:])

    # === NORMALISIERUNG ===
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

    # === KOMBINIERE ALLE FEATURES ===
    features = np.concatenate([
        s11_mag, s21_mag, s12_mag, s22_mag,           # 4 × 101 = 404
        s11_phase, s21_phase, s12_phase, s22_phase,   # 4 × 101 = 404  
        s11_norm, s21_norm, s12_norm, s22_norm,       # 4 × 101 = 404
        s11_grad, s21_grad, s12_grad, s22_grad,       # 4 × 101 = 404
        [s11_band1, s11_band2, s11_band3, s21_band1, s21_band2, s21_band3]  # 6 stat.
    ])
    
    return features

# ---- Testdaten sammeln --------------------------------------------------------
print(f"\n📂 Sammle Test-Dateien...")

all_test_files = []
ohne_test_files = []
test_material_counts = {}

for material in V6_5_MATERIALS:
    mat_dir = os.path.join(TEST_DIR, material)
    if os.path.exists(mat_dir):
        files = sorted(glob.glob(os.path.join(mat_dir, "*.s2p")))
        all_test_files.extend(files)
        test_material_counts[material] = len(files)
        print(f"  📊 {material}: {len(files)} Testdateien")
        
        if material == "ohne":
            ohne_test_files.extend(files)
    else:
        print(f"   {material}: Ordner nicht gefunden!")
        test_material_counts[material] = 0

total_test_files = len(all_test_files)
print(f"\n Test-Dateien gesamt: {total_test_files}")
print(f" Durchschnitt pro Klasse: {total_test_files/len(V6_5_MATERIALS):.1f}")

if total_test_files == 0:
    print(f" Keine Testdateien gefunden!")
    exit(1)

# ---- Feature-Extraktion für Test --------------------------------------------------------
print(f"\n Extrahiere Features für Test-Dateien...")
start_time = time.time()

X_test, y_test, test_files_processed = [], [], []

for i, f in enumerate(all_test_files):
    if i % 20 == 0 or i == len(all_test_files) - 1:
        elapsed = time.time() - start_time
        print(f"   Verarbeitet: {i+1}/{len(all_test_files)} ({100*(i+1)/len(all_test_files):.1f}%) - {elapsed:.1f}s")
    
    try:
        ref = find_reference_for_v6_5(f, all_test_files, ohne_test_files)
        feat = load_features_v6_5_dnn(f, ref)
        label = label_from_path_v6_5(f)
        
        if label == "unbekannt":
            print(f"⚠️ Unbekanntes Label für: {f}")
            continue
        
        X_test.append(feat)
        y_test.append(label)
        test_files_processed.append(f)
        
    except Exception as e:
        print(f" Fehler bei Test-Datei {f}: {e}")
        continue

total_time = time.time() - start_time
print(f"\n Test-Feature-Extraktion: {len(X_test)}/{len(all_test_files)} in {total_time:.1f}s")

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

print(f"\n Test-Daten Dimensionen:")
print(f"   X_test Shape: {X_test.shape}")
print(f"   Unique Labels: {len(set(y_test))}")
print(f"   Features: {X_test.shape[1]} (erwartet: {feature_info['dimensions']})")

# Prüfe Feature-Konsistenz
if X_test.shape[1] != feature_info['dimensions']:
    print(f"⚠️ WARNUNG: Feature-Dimensionen stimmen nicht überein!")
    print(f"  Erwartet: {feature_info['dimensions']}, Erhalten: {X_test.shape[1]}")

# ---- Preprocessing (IDENTISCH zum Training) --------------------------------------------------------
print(f"\n Wende Training-Preprocessing an...")

# Verwende den TRAINIERTEN Scaler (wichtig!)
X_test_scaled = scaler.transform(X_test)

print(f" Feature-Skalierung angewendet")

# ---- Vorhersage mit DNN-Modell --------------------------------------------------------
print(f"\n Führe DNN-Vorhersage aus...")

start_pred = time.time()

# Vorhersagen
y_pred = mlp_model.predict(X_test_scaled)
y_pred_proba = mlp_model.predict_proba(X_test_scaled)

pred_time = time.time() - start_pred

# Labels zurück dekodieren
y_pred_labels = label_encoder.inverse_transform(y_pred)

print(f" Vorhersage abgeschlossen in {pred_time:.3f}s")
print(f" Durchschnitt: {pred_time/len(X_test)*1000:.2f}ms pro Sample")

# ---- Ergebnisanalyse --------------------------------------------------------
print(f"\n === DNN MODEL TEST RESULTS ===")

# Gesamtgenauigkeit
overall_accuracy = accuracy_score(y_test, y_pred_labels)
print(f" Gesamt-Genauigkeit: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# Korrekte vs. Falsche Vorhersagen
correct_predictions = np.sum(y_test == y_pred_labels)
total_predictions = len(y_test)
print(f" Korrekt: {correct_predictions}/{total_predictions}")

# ---- Per-Klassen Analyse --------------------------------------------------------
print(f"\n Per-Klassen Performance:")
for material in V6_5_MATERIALS:
    if material in y_test:
        # Mask für diese Klasse
        mask = (y_test == material)
        
        # Statistiken
        correct = np.sum(y_pred_labels[mask] == material)
        total = np.sum(mask)
        accuracy = correct / total if total > 0 else 0
        
        # Durchschnittliche Konfidenz für diese Klasse
        material_idx = label_encoder.transform([material])[0]
        if total > 0:
            confidences = y_pred_proba[mask, material_idx]
            avg_confidence = np.mean(confidences)
            max_confidence = np.max(confidences)
            min_confidence = np.min(confidences)
        else:
            avg_confidence = max_confidence = min_confidence = 0
        
        print(f"   {material.upper()}: {correct}/{total} ({accuracy*100:.1f}%)")
        print(f"      Konfidenz: Ø{avg_confidence:.3f} | Max:{max_confidence:.3f} | Min:{min_confidence:.3f}")

# ---- Confusion Matrix --------------------------------------------------------
print(f"\n Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred_labels, labels=V6_5_MATERIALS)

print("Actual\\Predicted", end="")
for cls in V6_5_MATERIALS:
    print(f"\t{cls[:8]}", end="")
print()

for i, true_cls in enumerate(V6_5_MATERIALS):
    print(f"{true_cls[:12]}", end="")
    for j in range(len(V6_5_MATERIALS)):
        print(f"\t{conf_matrix[i][j]}", end="")
    print()

# ---- Classification Report --------------------------------------------------------
print(f"\n Detaillierter Classification Report:")
print(classification_report(y_test, y_pred_labels))

# ---- Falsch klassifizierte Dateien --------------------------------------------------------
print(f"\n Falsch klassifizierte Dateien:")
misclassified = []
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_labels)):
    if true_label != pred_label:
        confidence = np.max(y_pred_proba[i])
        misclassified.append({
            'file': os.path.basename(test_files_processed[i]),
            'true': true_label,
            'predicted': pred_label,
            'confidence': confidence
        })

if misclassified:
    print(f"Gefunden: {len(misclassified)} Fehler")
    for error in misclassified[:10]:  # Zeige max. 10 Fehler
        print(f"  {error['file']}: {error['true']} → {error['predicted']} (Konfidenz: {error['confidence']:.3f})")
    if len(misclassified) > 10:
        print(f"  ... und {len(misclassified)-10} weitere")
else:
    print("🎉 PERFEKT! Alle Dateien korrekt klassifiziert!")

# ---- Modell-Vergleich --------------------------------------------------------
training_acc = training_info['val_accuracy']
test_acc = overall_accuracy
generalization_gap = abs(training_acc - test_acc)

print(f"\n === MODELL GENERALISIERUNG ===")
print(f" Training Validation Accuracy: {training_acc:.4f} ({training_acc*100:.2f}%)")
print(f" Independent Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f" Generalisierungslücke: {generalization_gap:.4f} ({generalization_gap*100:.2f}%)")

if generalization_gap < 0.02:
    print(f" EXZELLENTE Generalisierung! Modell ist sehr robust.")
elif generalization_gap < 0.05:
    print(f" GUTE Generalisierung! Modell funktioniert gut.")
elif generalization_gap < 0.10:
    print(f" AKZEPTABLE Generalisierung.")
else:
    print(f" Mögliches Overfitting! Große Generalisierungslücke.")

# ---- Abschluss-Zusammenfassung --------------------------------------------------------
print(f"\n" + "="*80)
print(f" V6_5 Deep Neural Network Test Abgeschlossen!")
print(f" Modell: {MODEL_FILE}")
print(f" Test-Samples: {len(X_test)} aus {len(V6_5_MATERIALS)} Klassen")
print(f" DNN-Architektur: {training_info['architecture']}")
print(f" Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f" Korrekte Vorhersagen: {correct_predictions}/{total_predictions}")
print(f" Durchschnittszeit: {pred_time/len(X_test)*1000:.2f}ms pro Sample")

if overall_accuracy >= 0.99:
    print(f" PERFEKTE Performance! DNN-Modell ist produktionsbereit! 🚀")
elif overall_accuracy >= 0.95:
    print(f" EXZELLENTE Performance! Modell funktioniert hervorragend!")
elif overall_accuracy >= 0.90:
    print(f" SEHR GUTE Performance! Modell ist zuverlässig.")
else:
    print(f" Performance könnte verbessert werden.")

print("="*80)