import os
import io
import csv
import json
import sqlite3
import datetime
import numpy as np
import tensorflow as tf
import cv2

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "cropwise_secret_2025"

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = load_model("model/crop_disease_model.h5")
dummy = np.zeros((1, 224, 224, 3))
model.predict(dummy)

with open("model/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ─────────────────────────────────────────────
# Disease knowledge base (expanded)
# ─────────────────────────────────────────────
DISEASE_DB = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Pepper – Bacterial Spot",
        "crop": "Pepper",
        "chemical": "Copper-based bactericide (e.g., Kocide 3000)",
        "organic": "Neem oil spray (5 ml / L water)",
        "prevention": "Avoid overhead irrigation; remove infected debris",
        "spread": "Water-splashed bacteria; infected seeds",
        "season_risk": [6, 7, 8, 9],   # months of high risk
        "color": "#e76f51",
        "description": "Dark, water-soaked spots with yellow halos on leaves and fruit."
    },
    "Potato___Early_blight": {
        "display": "Potato – Early Blight",
        "crop": "Potato",
        "chemical": "Mancozeb 75WP @ 2 g/L water",
        "organic": "Compost tea / Trichoderma spray",
        "prevention": "Crop rotation; remove infected plant debris",
        "spread": "Wind-borne fungal spores; infected tubers",
        "season_risk": [7, 8, 9, 10],
        "color": "#f4a261",
        "description": "Concentric ring lesions ('target board' pattern) on older leaves."
    },
    "Potato___Late_blight": {
        "display": "Potato – Late Blight",
        "crop": "Potato",
        "chemical": "Metalaxyl + Mancozeb @ 2.5 g/L",
        "organic": "Copper hydroxide spray",
        "prevention": "Remove infected leaves immediately; improve air circulation",
        "spread": "Wind, rain; infected tubers; cool moist weather",
        "season_risk": [9, 10, 11, 12],
        "color": "#264653",
        "description": "Water-soaked lesions turning brown-black; white mold on undersides."
    },
    "Potato___healthy": {
        "display": "Potato – Healthy",
        "crop": "Potato",
        "chemical": "None required",
        "organic": "Maintain balanced fertilization",
        "prevention": "Regular monitoring; proper irrigation scheduling",
        "spread": "N/A",
        "season_risk": [],
        "color": "#2a9d8f",
        "description": "No disease detected. Plant appears healthy."
    },
    "Tomato_Early_blight": {
        "display": "Tomato – Early Blight",
        "crop": "Tomato",
        "chemical": "Mancozeb 75WP @ 2 g/L",
        "organic": "Neem oil (3%) + copper soap",
        "prevention": "Mulch; avoid wetting foliage; stake plants",
        "spread": "Rain splash; infected debris; warm humid conditions",
        "season_risk": [6, 7, 8],
        "color": "#e9c46a",
        "description": "Brown target-like lesions starting on lower leaves."
    },
    "Tomato_Late_blight": {
        "display": "Tomato – Late Blight",
        "crop": "Tomato",
        "chemical": "Copper fungicide (Blitox) @ 3 g/L",
        "organic": "Compost extract spray",
        "prevention": "Proper plant spacing; avoid overhead watering",
        "spread": "Wind-borne spores; cool wet weather (15–20°C)",
        "season_risk": [10, 11, 12, 1],
        "color": "#457b9d",
        "description": "Greasy dark lesions; white fluffy growth on leaf undersides."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mi": {
        "display": "Tomato – Spider Mites",
        "crop": "Tomato",
        "chemical": "Miticide (Abamectin) @ 0.5 ml/L",
        "organic": "Insecticidal soap / water jet spray",
        "prevention": "Maintain humidity; introduce predatory mites",
        "spread": "Wind; physical contact; hot dry conditions",
        "season_risk": [3, 4, 5, 6],
        "color": "#c77dff",
        "description": "Tiny yellow stippling on leaves; fine webbing on underside."
    },
    "Tomato__Target_Spot": {
        "display": "Tomato – Target Spot",
        "crop": "Tomato",
        "chemical": "Chlorothalonil @ 2 g/L",
        "organic": "Baking soda spray (5 g/L)",
        "prevention": "Avoid dense planting; remove infected leaves",
        "spread": "Airborne fungal spores; warm and wet weather",
        "season_risk": [7, 8, 9],
        "color": "#80b918",
        "description": "Circular lesions with concentric rings on leaves and fruits."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Tomato – Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "chemical": "No direct cure; use systemic insecticides on whiteflies",
        "organic": "Neem oil spray; reflective mulch",
        "prevention": "Control whitefly vectors; use virus-resistant varieties",
        "spread": "Whitefly (Bemisia tabaci); cannot spread by contact",
        "season_risk": [4, 5, 6, 7, 8],
        "color": "#d62828",
        "description": "Upward leaf curling; yellowing; stunted growth caused by whitefly."
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Tomato – Mosaic Virus",
        "crop": "Tomato",
        "chemical": "No chemical cure available",
        "organic": "Remove and destroy infected plants",
        "prevention": "Disinfect tools; wash hands; use resistant varieties",
        "spread": "Contact (tools, hands); seeds; infected transplants",
        "season_risk": [3, 4, 5, 6, 7, 8, 9, 10],
        "color": "#f77f00",
        "description": "Mosaic discoloration; leaf distortion; reduced fruit quality."
    }
}

# Seasonal risk labels
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────
# SQLite history database
# ─────────────────────────────────────────────
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            raw_class TEXT,
            disease TEXT,
            crop TEXT,
            confidence REAL,
            severity TEXT,
            top2_class TEXT,
            top2_conf REAL,
            top3_class TEXT,
            top3_conf REAL,
            farmer_note TEXT,
            heatmap_path TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (timestamp, filename, raw_class, disease, crop, confidence, severity,
         top2_class, top2_conf, top3_class, top3_conf, farmer_note, heatmap_path)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["timestamp"], data["filename"], data["raw_class"],
        data["disease"], data["crop"], data["confidence"], data["severity"],
        data["top2_class"], data["top2_conf"],
        data["top3_class"], data["top3_conf"],
        data.get("farmer_note", ""), data.get("heatmap_path", "")
    ))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

def get_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_prediction_by_id(pred_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE id=?", (pred_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def update_note(pred_id, note):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE predictions SET farmer_note=? WHERE id=?", (note, pred_id))
    conn.commit()
    conn.close()

def delete_prediction(pred_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────
def generate_gradcam(img_array):
    try:
        base_model = model.layers[0]
        last_conv = None
        for layer in reversed(base_model.layers):
            if "conv" in layer.name:
                last_conv = layer.name
                break
        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[base_model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print("GradCAM Error:", e)
        return None

# ─────────────────────────────────────────────
# Severity + season risk helpers
# ─────────────────────────────────────────────
def get_severity(conf):
    if conf < 50:   return "Low"
    if conf < 75:   return "Moderate"
    return "High"

def get_season_risk(raw_class):
    month_now = datetime.datetime.now().month
    risk_months = DISEASE_DB.get(raw_class, {}).get("season_risk", [])
    if not risk_months:
        return "Not applicable"
    return "HIGH – Peak season now!" if month_now in risk_months else "LOWER – Off-peak season"

# ─────────────────────────────────────────────
# Folders
# ─────────────────────────────────────────────
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/heatmaps", exist_ok=True)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    farmer_note = request.form.get("farmer_note", "")

    if not file:
        return redirect(url_for("home"))

    filename = file.filename
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    # ── preprocess ──
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ── predict ──
    preds = model.predict(img_array)[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    def cls_name(i):
        return class_names[i] if i < len(class_names) else "Unknown"

    raw1 = cls_name(top3_idx[0])
    raw2 = cls_name(top3_idx[1])
    raw3 = cls_name(top3_idx[2])

    conf1 = round(float(preds[top3_idx[0]]) * 100, 2)
    conf2 = round(float(preds[top3_idx[1]]) * 100, 2)
    conf3 = round(float(preds[top3_idx[2]]) * 100, 2)

    # ── Get disease info ──
    info = DISEASE_DB.get(raw1, {
        "display": raw1.replace("___", " – ").replace("__", " ").replace("_", " "),
        "crop": "Unknown",
        "chemical": "Not available",
        "organic": "Not available",
        "prevention": "Not available",
        "spread": "Not available",
        "season_risk": [],
        "color": "#888",
        "description": "No detailed info available."
    })

    # ✅ LOW CONFIDENCE FIX (VERY IMPORTANT)
    if conf1 < 50:
        info["display"] = "No clear disease detected"
        info["description"] = "Model is not confident. Please verify manually."
        info["chemical"] = "Not recommended"
        info["organic"] = "Not recommended"
        info["prevention"] = "Observe plant for a few days"

    info2 = DISEASE_DB.get(raw2, {"display": raw2.replace("_"," "), "color": "#aaa"})
    info3 = DISEASE_DB.get(raw3, {"display": raw3.replace("_"," "), "color": "#aaa"})

    severity = get_severity(conf1)
    season_risk = get_season_risk(raw1)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── GradCAM ──
    heatmap = generate_gradcam(img_array)
    heatmap_path = ""
    if heatmap is not None:
        img_cv = cv2.imread(filepath)
        hmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        hmap = np.uint8(255 * hmap)
        hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(img_cv, 0.6, hmap, 0.4, 0)
        heatmap_path = os.path.join("static/heatmaps", "heatmap_" + filename)
        cv2.imwrite(heatmap_path, result_img)

        # ── Save to DB ──
    pred_id = save_prediction({
        "timestamp": timestamp,
        "filename": filename,
        "raw_class": raw1,
        "disease": info["display"],
        "crop": info.get("crop", "Unknown"),
        "confidence": conf1,
        "severity": severity,
        "top2_class": info2.get("display", raw2),
        "top2_conf": conf2,
        "top3_class": info3.get("display", raw3),
        "top3_conf": conf3,
        "farmer_note": farmer_note,
        "heatmap_path": heatmap_path
    })

    # ✅ PRINT LINK IN TERMINAL
    report_url = f"http://127.0.0.1:5000/report/{pred_id}"
    print("\n===== PDF REPORT LINK =====")
    print(report_url)
    print("===========================\n")

    return render_template(
        "result.html",
        pred_id=pred_id,
        disease=info["display"],
        crop=info.get("crop", "Unknown"),
        confidence=conf1,
        severity=severity,
        season_risk=season_risk,
        chemical=info.get("chemical", "N/A"),
        organic=info.get("organic", "N/A"),
        prevention=info.get("prevention", "N/A"),
        spread=info.get("spread", "N/A"),
        description=info.get("description", ""),
        disease_color=info.get("color", "#888"),
        image_path=filepath,
        heatmap_path=heatmap_path,
        top2_disease=info2.get("display", raw2),
        top2_conf=conf2,
        top2_color=info2.get("color","#aaa"),
        top3_disease=info3.get("display", raw3),
        top3_conf=conf3,
        top3_color=info3.get("color","#aaa"),
        farmer_note=farmer_note,
        timestamp=timestamp,
        month_names=MONTH_NAMES,
        season_months=DISEASE_DB.get(raw1, {}).get("season_risk", [])
    )
# ── History dashboard ──
@app.route("/history")
def history():
    rows = get_all_predictions()
    # Stats
    total = len(rows)
    disease_counts = {}
    crop_counts = {}
    for r in rows:
        disease_counts[r["disease"]] = disease_counts.get(r["disease"], 0) + 1
        crop_counts[r["crop"]] = crop_counts.get(r["crop"], 0) + 1

    return render_template(
        "history.html",
        rows=rows,
        total=total,
        disease_counts=json.dumps(disease_counts),
        crop_counts=json.dumps(crop_counts)
    )

# ── Save farmer note (AJAX) ──
@app.route("/save_note", methods=["POST"])
def save_note():
    data = request.get_json()
    update_note(data["id"], data["note"])
    return jsonify({"status": "ok"})

# ── Delete prediction ──
@app.route("/delete/<int:pred_id>")
def delete(pred_id):
    delete_prediction(pred_id)
    return redirect(url_for("history"))

# ── Export CSV ──
@app.route("/export_csv")
def export_csv():
    rows = get_all_predictions()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id","timestamp","filename","disease","crop","confidence","severity",
        "top2_class","top2_conf","top3_class","top3_conf","farmer_note"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k,"") for k in writer.fieldnames})
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="cropwise_history.csv"
    )

# ── PDF Report ──
@app.route("/report/<int:pred_id>")
def generate_report(pred_id):
    r = get_prediction_by_id(pred_id)
    if not r:
        return "Prediction not found", 404

    pdf = FPDF()
    pdf.add_page()

    # ✅ FIX 1: Use built-in font (NO FILE NEEDED)
    # This avoids your FileNotFoundError
    pdf.set_font("Arial", "B", 16)

    # Header
    pdf.set_fill_color(34, 85, 34)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(0, 8)
    pdf.cell(210, 12, "CropWise - Disease Analysis Report", align="C")

    # Timestamp
    pdf.set_font("Arial", "", 9)
    pdf.set_xy(0, 21)
    pdf.cell(210, 6,
             f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Scan ID: #{pred_id}",
             align="C")
    pdf.ln(18)

    # ✅ CLEAN TEXT (important fix)
    def clean(text):
        return str(text).replace("–", "-")

    # Disease section
    pdf.set_text_color(34, 85, 34)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Detected: {clean(r['disease'])}", ln=True)

    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7,
             f"Crop: {clean(r['crop'])} | Confidence: {r['confidence']}% | Severity: {r['severity']}",
             ln=True)
    pdf.cell(0, 7, f"Scan Time: {r['timestamp']}", ln=True)
    pdf.ln(4)

    # Top-3
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(34, 85, 34)
    pdf.cell(0, 8, "Top-3 Predictions", ln=True)

    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"1st: {clean(r['disease'])} ({r['confidence']}%)", ln=True)
    pdf.cell(0, 6, f"2nd: {clean(r['top2_class'])} ({r['top2_conf']}%)", ln=True)
    pdf.cell(0, 6, f"3rd: {clean(r['top3_class'])} ({r['top3_conf']}%)", ln=True)
    pdf.ln(4)

    # Management
        # Management
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(34, 85, 34)
    pdf.cell(0, 8, "Management Recommendations", ln=True)

    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Arial", "", 10)

    info = DISEASE_DB.get(r["raw_class"], {})

    for label, key in [
        ("Chemical Treatment", "chemical"),
        ("Organic Alternative", "organic"),
        ("Prevention", "prevention"),
        ("Spread Mode", "spread")
    ]:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, f"{label}:", ln=True)

        pdf.set_font("Arial", "", 10)
        text = clean(info.get(key, r.get(key, "N/A"))) or "N/A"
        pdf.multi_cell(0, 6, text)

        pdf.ln(1)

    pdf.ln(4)

    # Farmer notes
    if r.get("farmer_note"):
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(34, 85, 34)
        pdf.cell(0, 8, "Farmer Notes", ln=True)

        pdf.set_text_color(60, 60, 60)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, clean(r["farmer_note"]))

    # Image
    if os.path.exists(r.get("heatmap_path", "")):
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "GradCAM Visualization", ln=True)
        try:
            pdf.image(r["heatmap_path"], w=100)
        except:
            pass

    # Footer
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10,
             "CropWise AI - For educational purposes only.",
             align="C")

    # Final output
    pdf_output = bytes(pdf.output(dest='S'))

    return send_file(
        io.BytesIO(pdf_output),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"cropwise_report_{pred_id}.pdf"
    )
if __name__ == "__main__":
    print("\n🚀 Starting Flask Server...")
    print("👉 Open: http://127.0.0.1:5000/\n")
    app.run(debug=True)