# DRISHTI-XAI
**Explainable AI Diabetic Retinopathy Screening for Rural India**

> ⚠️ **Prototype disclaimer:** This is an AI-assisted screening **prototype** built for
> a hackathon (SIH 2026) context. It is **not a certified medical device** and has
> **not undergone clinical validation**. Do not use it to diagnose or treat real
> patients. See "Demo / Mock Mode" below.

---

## What it does

DRISHTI-XAI walks a rural health worker through a full diabetic retinopathy (DR)
screening workflow:

```
Login → Patient Registration → Fundus Image Upload → Image Quality Check →
DR Prediction → Grad-CAM Explanation → Risk Assessment → Smart Referral →
Doctor Dashboard → Follow-up Tracking
```

Key features:

- **Login** for two roles: Health Worker and Doctor
- **Patient registration** (ID, age, diabetes duration, previous screening history)
- **Fundus image upload** (JPG/PNG) with live preview, or use built-in demo samples
- **Image quality gating** (OpenCV): rejects blurry, too-dark, too-bright, or
  too-low-resolution images and asks for a recapture
- **DR severity classification**: No DR / Mild DR / Moderate DR / Severe DR,
  with a confidence score
- **Explainable AI**: a Grad-CAM-style heatmap overlay showing which regions of
  the retina most influenced the prediction
- **Rule-based risk engine**: Low / Medium / High / Critical, combining the AI
  output with diabetes duration and screening history — logic is transparent
  and shown to the user (not another black box)
- **Smart referral recommendation** derived from the risk level
- **Doctor dashboard**: all patients, AI results, confidence, risk, referral
  status, charts
- **Follow-up tracking**: Pending / Referred / Completed, editable by doctors
- **Multi-language UI**: English, Telugu (తెలుగు), Hindi (हिन्दी)

---

## Demo / Mock Mode

No clinically validated DR model ships with this prototype (real fundus datasets
are license-restricted and this repo has no trained weights). Instead:

- `core/model.py` implements a **mock predictor** that derives a DR severity
  estimate from simple, transparent OpenCV image statistics (dark/red
  lesion-like blob density + edge density) — clearly **not a diagnostic
  algorithm**, but structured so a real model drops in cleanly.
- `core/gradcam.py` implements a matching **mock saliency heatmap** in demo
  mode, and a **real Grad-CAM** implementation (Selvaraju et al.) that
  activates automatically once a real Keras model is present.
- `assets/sample_images/` contains **procedurally generated, synthetic**
  fundus-like images (one per severity class) so the app is runnable
  end-to-end without any external dataset. These are clearly not real
  patient photos.

The sidebar shows a "🧪 demo/mock mode" notice whenever no real model is loaded.

### Plugging in a real trained model later

1. Train a 4-class (No DR / Mild / Moderate / Severe) classifier and save it as:
   - Keras: `models/dr_model.h5`, **or**
   - PyTorch: `models/dr_model.pt`
2. On next launch, `core/model.py`'s `DRModel` class auto-detects the file and
   switches from `mode="mock"` to `mode="real"` — no other code changes needed.
3. `core/gradcam.py` will automatically use real Grad-CAM (via gradients from
   the last conv layer) instead of the mock saliency map for TensorFlow models.
4. Replace `assets/sample_images/` with a properly licensed, ethics-approved
   fundus image dataset before any real-world pilot.

---

## Project structure

```
drishti_xai/
├── app.py                     # Streamlit entry point — pages, routing, workflow
├── config.py                  # Constants: classes, colors, thresholds, demo users
├── requirements.txt
├── README.md
├── core/
│   ├── image_quality.py       # OpenCV blur / brightness / resolution checks
│   ├── model.py                # DRModel: real-model loader + mock predictor
│   ├── gradcam.py              # Real Grad-CAM + mock saliency heatmap
│   └── risk_engine.py          # Rule-based risk level + referral logic
├── db/
│   └── database.py             # SQLite schema, auth, CRUD helpers
├── i18n/
│   └── translations.py         # English / Telugu / Hindi UI strings
├── utils/
│   ├── demo_data.py            # Synthetic fundus image generator + patient seeding
│   └── helpers.py              # PIL/OpenCV conversion, badges, image saving
├── assets/
│   └── sample_images/          # Auto-generated synthetic demo fundus images
└── data/                        # Created at runtime: SQLite DB, uploads, heatmaps
```

---

## Setup & run

### 1. Requirements

- Python 3.9+

### 2. Install dependencies

```bash
cd drishti_xai
pip install -r requirements.txt
```

(Optional) If you have a trained model to plug in, also install the matching
framework — `tensorflow` or `torch` — and uncomment it in `requirements.txt`.

### 3. Run

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open it
in your browser.

### 4. Log in with a demo account

| Role          | Username        | Password     |
|---------------|-----------------|--------------|
| Health Worker | `healthworker1` | `worker123`  |
| Doctor        | `doctor1`       | `doctor123`  |

On first launch the app automatically:
- creates the SQLite database (`data/drishti_xai.db`),
- seeds the two demo accounts above,
- generates 4 synthetic demo fundus images (one per DR severity),
- seeds 4 demo patients.

### 5. Try the workflow

1. Log in as `healthworker1`.
2. Go to **Patient Registration** and add a patient (or use a seeded one).
3. Go to **Image Upload & Screening**, select the patient, and either upload
   your own fundus image or click one of the **"Use: <severity> sample"**
   buttons to try a demo image.
4. Run the **Quality Check**, then **Run AI Screening** to see the DR class,
   confidence, Grad-CAM heatmap, risk level, and referral recommendation.
5. Click **Save Result**.
6. Log out and log in as `doctor1` to view the **Doctor Dashboard** and update
   **Follow-up Tracking** status.

---

## Notes on scope & limitations

- This is a **single-process prototype** (SQLite, in-process Streamlit) —
  not designed for concurrent multi-clinic production load.
- Authentication uses a simple SHA-256 hash over two hardcoded demo accounts —
  **replace with proper auth (bcrypt/argon2, real user management) before any
  real deployment.**
- The mock DR classifier and mock Grad-CAM are **heuristic placeholders**, not
  trained AI — they exist to make the full pipeline runnable and demonstrable
  end-to-end. Do not interpret their output as medically meaningful.
- The risk engine's escalation rules (diabetes duration, screening history)
  are illustrative examples of a rule-based referral policy, not a validated
  clinical guideline — a real deployment should encode locally approved
  clinical protocols, reviewed by an ophthalmologist.
