import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from huggingface_hub import hf_hub_download
from PIL import Image
import httpx
from contextlib import asynccontextmanager

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
)

# ================== CLASS DEFINITIONS ==================
CANONICAL_CLASSES = [
    "Acne",
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Chickenpox",
    "Dermato Fibroma",
    "Dyshidrotic Eczema",
    "Melanoma",
    "Nail Fungus",
    "Nevus",
    "Normal Skin",
    "Pigmented Benign Keratosis",
    "Ringworm",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion",
]

TEXT_CLASSES = [
    "Atopic Dermatitis",
    "Benign keratosis",
    "Melanocytic nevus",
    "Tinea Ringworm Candidiasis",
    "Melanoma",
    "Basal Cell Carcinoma",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
]

TEXT_TO_CANONICAL = {
    "Atopic Dermatitis": "Dyshidrotic Eczema",
    "Benign keratosis": "Pigmented Benign Keratosis",
    "Melanocytic nevus": "Nevus",
    "Tinea Ringworm Candidiasis": "Ringworm",
}

# ================== GLOBAL MODELS (loaded at startup) ==================
image_model = None
text_model = None

def download_models():
    global image_model, text_model
    image_path = hf_hub_download(
        "PrudhviManikanta/skin-disease-efficientnet-multiclass",
        "skin_disease_efficientnet_ft.keras",
    )
    text_path = hf_hub_download(
        "PrudhviManikanta/skin-disease-efficientnet-multiclass",
        "model.keras",
    )
    image_model = tf.keras.models.load_model(image_path)
    text_model = tf.keras.models.load_model(text_path)

# ================== FASTAPI LIFESPAN ==================
BOT_TOKEN = os.getenv("TOKEN")
if not BOT_TOKEN:
    raise ValueError("TOKEN environment variable is missing!")

app_bot = ApplicationBuilder().token(BOT_TOKEN).build()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    download_models()
    # Set webhook once server is up
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if not render_host:
        # Fallback for local testing
        webhook_url = f"https://{os.getenv('HOST', 'localhost')}/webhook"
    else:
        webhook_url = f"https://{render_host}/webhook"
    await app_bot.bot.set_webhook(webhook_url)
    print(f"✅ Webhook set to {webhook_url}")
    yield
    # Shutdown: remove webhook and stop the bot
    await app_bot.bot.delete_webhook()
    await app_bot.shutdown()

app = FastAPI(lifespan=lifespan)

# ================== PREPROCESS ==================
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def preprocess_text(text: str):
    return tf.constant([text], dtype=tf.string)

# ================== TEXT HANDLING ==================
def is_medical_text(text):
    keywords = [
        "rash", "itch", "pimple", "skin", "lesion",
        "red", "swelling", "infection", "spot", "patch",
    ]
    text = text.lower()
    return any(k in text for k in keywords)

def rule_based_prediction(text):
    text = text.lower()
    if "ring" in text or "circular" in text:
        return "Ringworm"
    if "pimple" in text or "acne" in text:
        return "Acne"
    if "dark mole" in text:
        return "Melanoma"
    return None

def is_text_reliable(text_pred):
    return np.max(text_pred[0]) < 0.9

# ================== MAPPING ==================
def map_text_to_canonical(text_probs):
    canonical_probs = np.zeros(len(CANONICAL_CLASSES))
    for i, text_class in enumerate(TEXT_CLASSES):
        canonical_name = TEXT_TO_CANONICAL.get(text_class, text_class)
        if canonical_name in CANONICAL_CLASSES:
            idx = CANONICAL_CLASSES.index(canonical_name)
            canonical_probs[idx] += text_probs[i]
    return canonical_probs

# ================== FUSION ==================
MIN_CONFIDENCE = 0.40

def fuse(img_pred=None, text_pred=None):
    if img_pred is not None and text_pred is not None:
        img = img_pred[0]
        text = map_text_to_canonical(text_pred[0])
        img = img / (np.sum(img) + 1e-8)
        text = text / (np.sum(text) + 1e-8)
        img_conf = np.max(img)
        text_conf = np.max(text)
        if img_conf > 0.85:
            return img
        elif text_conf > 0.85:
            return text
        else:
            return (0.7 * img) + (0.3 * text)
    if img_pred is not None:
        return img_pred[0]
    if text_pred is not None:
        return map_text_to_canonical(text_pred[0])
    return None

def get_confidence_label(conf):
    if conf > 0.8:
        return "High ✅"
    elif conf > 0.6:
        return "Moderate ⚠️"
    else:
        return "Low ❗"

# ================== TELEGRAM HANDLERS ==================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Skin Disease AI Bot!\n\n"
        "📸 Send an image\n"
        "📝 Or describe symptoms\n"
        "🤖 Or both for best accuracy!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    text = update.message.text
    photo = update.message.photo

    data, files = {}, {}
    if text:
        data["text"] = text

    try:
        if photo:
            # Download photo bytes directly (no temp file)
            file = await photo[-1].get_file()
            img_bytes = await file.download_as_bytearray()
            files["image"] = ("image.jpg", bytes(img_bytes), "image/jpeg")
            async with httpx.AsyncClient() as client:
                res = await client.post("http://localhost:8000/predict", data=data, files=files)
                result = res.json()
        else:
            async with httpx.AsyncClient() as client:
                res = await client.post("http://localhost:8000/predict", data=data)
                result = res.json()

        if "error" in result:
            await update.message.reply_text(
                f"❗ {result.get('message', result['error'])}\n"
                f"Confidence: {result.get('confidence', 0):.2f}"
            )
            return

        msg = "🧠 *Skin Disease Prediction*\n\n"
        for i, p in enumerate(result["top_predictions"], start=1):
            label = get_confidence_label(p["confidence"])
            msg += f"{i}. *{p['disease']}*\n   Confidence: {p['confidence']:.2f} ({label})\n\n"
        msg += "⚠️ _This is an AI-based prediction. Please consult a dermatologist._"
        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

app_bot.add_handler(CommandHandler("start", start_cmd))
app_bot.add_handler(MessageHandler(filters.ALL, handle_message))

# ================== API ROUTES ==================
@app.post("/predict")
async def predict(image: UploadFile = File(None), text: str = Form(None)):
    if not image and not text:
        raise HTTPException(400, "Please provide an image or text description.")

    img_pred, text_pred = None, None

    # Text precheck
    if text:
        if not is_medical_text(text):
            return {
                "error": "Irrelevant input",
                "message": "Please describe skin-related symptoms.",
            }
        rule_pred = rule_based_prediction(text)
        if rule_pred:
            return {"top_predictions": [{"disease": rule_pred, "confidence": 0.85}]}

    if image:
        # Read image bytes directly
        contents = await image.read()
        img = preprocess_image(contents)
        img_pred = image_model.predict(img)
        print("IMAGE PRED:", img_pred)

    if text:
        txt = preprocess_text(text)
        text_pred = text_model.predict(txt)
        print("TEXT PRED:", text_pred)
        if not is_text_reliable(text_pred):
            print("Ignoring unreliable text output")
            text_pred = None

    final_pred = fuse(img_pred, text_pred)
    final_pred = final_pred / (np.sum(final_pred) + 1e-8)
    confidence = float(np.max(final_pred))

    if confidence < MIN_CONFIDENCE:
        return {
            "error": "Low confidence",
            "message": "Please provide clearer input.",
            "confidence": confidence,
        }

    top3_idx = np.argsort(final_pred)[::-1][:3]
    top3 = [
        {
            "disease": CANONICAL_CLASSES[int(i)],
            "confidence": float(final_pred[int(i)]),
        }
        for i in top3_idx
    ]
    return {"top_predictions": top3}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, app_bot.bot)
    await app_bot.process_update(update)
    return {"ok": True}