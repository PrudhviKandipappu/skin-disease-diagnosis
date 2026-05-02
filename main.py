import os
import asyncio
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from huggingface_hub import hf_hub_download
from PIL import Image
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
    # ----- NEW CLASSES (from image dataset) -----
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Eczema",
    "Scabies",
    "Seborrheic Dermatitis",
    "Tinea Corporis",
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
    "Atopic Dermatitis": "Atopic Dermatitis",
    "Benign keratosis": "Pigmented Benign Keratosis",
    "Melanocytic nevus": "Nevus",
    "Tinea Ringworm Candidiasis": "Tinea Corporis",
}

# ================== GLOBAL MODELS ==================
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
    # 1. Download models
    download_models()

    # 2. Initialize the bot
    await app_bot.initialize()
    print("✅ Bot initialized")

    # 3. Set webhook
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if not render_host:
        webhook_url = f"https://{os.getenv('HOST', 'localhost')}/webhook"
    else:
        webhook_url = f"https://{render_host}/webhook"
    await app_bot.bot.set_webhook(webhook_url)
    print(f"✅ Webhook set to {webhook_url}")

    # 4. Pre‑warm the image model (one‑time JIT compilation)
    print("🔥 Warming up image model (this may take 1–2 minutes)...")
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    image_model.predict(dummy)
    print("✅ Image model warmed up")

    yield

    # Shutdown
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
    # Common ringworm clues
    if "ring" in text or "circular" in text or "round" in text or "tinea" in text:
        return "Tinea Corporis"
    if "pimple" in text or "acne" in text:
        return "Acne"
    if "dark mole" in text:
        return "Melanoma"
    # New disease keywords
    if "scabies" in text:
        return "Scabies"
    if "seborrheic dermatitis" in text or "seb derm" in text:
        return "Seborrheic Dermatitis"
    if "contact dermatitis" in text:
        return "Contact Dermatitis"
    if "eczema" in text and "dyshidrotic" not in text:
        return "Eczema"
    if "atopic dermatitis" in text or "atopic eczema" in text:
        return "Atopic Dermatitis"
    return None

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

# ================== NEVUS OVERCONFIDENCE GUARD ==================
NEVUS_OVERCONFIDENCE_THRESHOLD = 0.95
NEVUS_GAP_THRESHOLD = 0.80

def is_nevus_blind_guess(top3):
    if not top3:
        return False
    first = top3[0]
    if first["disease"] != "Nevus":
        return False
    if first["confidence"] < NEVUS_OVERCONFIDENCE_THRESHOLD:
        return False
    if len(top3) > 1:
        gap = first["confidence"] - top3[1]["confidence"]
        if gap < NEVUS_GAP_THRESHOLD:
            return False
    return True

# ================== SHARED PREDICTION LOGIC ==================
def run_prediction(image_bytes: bytes | None = None, text: str | None = None) -> dict:
    """Returns the prediction dictionary used by both API and Telegram."""
    if image_bytes is None and text is None:
        raise HTTPException(400, "Please provide an image or text description.")

    img_pred, text_pred = None, None

    # ------ Text precheck ------
    if text:
        if not is_medical_text(text):
            return {
                "error": "Irrelevant input",
                "message": "Please describe skin-related symptoms.",
            }
        rule_pred = rule_based_prediction(text)
        if rule_pred:
            return {"top_predictions": [{"disease": rule_pred, "confidence": 0.85}]}

    # ------ Image ------
    if image_bytes:
        img = preprocess_image(image_bytes)
        img_pred = image_model.predict(img)
        print("IMAGE PRED:", img_pred)

    # ------ Text model ------
    if text and not (text and rule_based_prediction(text)):  # already handled rule
        txt = preprocess_text(text)
        text_pred = text_model.predict(txt)
        print("TEXT PRED:", text_pred)

    final_pred = fuse(img_pred, text_pred)

    if final_pred is None:
        return {
            "error": "Low confidence",
            "message": "The model could not confidently determine a condition. Please provide a clearer description or an image.",
            "confidence": 0.0,
        }

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

    # ---------- Nevus over‑confidence guard ----------
    if is_nevus_blind_guess(top3):
        return {
            "error": "Low confidence",
            "message": "The model is uncertain. It detected a mole‑like pattern, but cannot confidently say it's Nevus. Please provide a clearer image or describe symptoms.",
            "confidence": top3[0]["confidence"],
        }

    return {"top_predictions": top3}

# ================== API ROUTES ==================
@app.get("/")
async def root():
    return {"status": "alive", "message": "Skin Disease Bot API is running"}

@app.post("/predict")
async def predict(image: UploadFile = File(None), text: str = Form(None)):
    img_bytes = await image.read() if image else None
    return run_prediction(img_bytes, text)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    # Log the update type so we know what Telegram sent
    msg = data.get("message", {})
    print(f"📬 Webhook received: text={msg.get('text','')}, photo_count={len(msg.get('photo',[]))}, chat_id={msg.get('chat', {}).get('id')}")
    update = Update.de_json(data, app_bot.bot)
    
    async def safe_process():
        try:
            await app_bot.process_update(update)
        except Exception as e:
            print(f"❌ process_update CRASHED: {e}")
    asyncio.create_task(safe_process())
    return {"ok": True}

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
    chat_id = update.effective_chat.id

    print(f"🔥 HANDLER CALLED: text={text}, has_photo={bool(photo)}")

    # Show typing indicator while processing
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        img_bytes = None
        if photo:
            print("📥 Downloading image…")
            file = await photo[-1].get_file()
            # Timeout on download (30 seconds)
            try:
                raw_bytes = await asyncio.wait_for(
                    file.download_as_bytearray(),
                    timeout=30.0
                )
                img_bytes = bytes(raw_bytes)
                print(f"📸 Downloaded {len(img_bytes)} bytes")
            except asyncio.TimeoutError:
                await update.message.reply_text("❌ Image download timed out. Please try again.")
                return

        # Run prediction in a thread with a generous timeout (120 seconds)
        print("⏳ Running prediction...")
        result = await asyncio.wait_for(
            asyncio.to_thread(run_prediction, image_bytes=img_bytes, text=text),
            timeout=120.0
        )
        print(f"✅ Prediction result: {result}")

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

    except asyncio.TimeoutError:
        await update.message.reply_text("❌ Prediction timed out. Please try again with a smaller image or clearer description.")
    except Exception as e:
        print(f"❌ HANDLER ERROR: {e}")
        try:
            await update.message.reply_text("❌ Internal error. Please try again.")
        except:
            pass

# ---------- REGISTER HANDLERS (MANDATORY) ----------
app_bot.add_handler(CommandHandler("start", start_cmd))
app_bot.add_handler(MessageHandler(filters.ALL, handle_message))