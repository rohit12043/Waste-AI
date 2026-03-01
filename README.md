# WASTE·AI — AI-Based Waste Segregation & Carbon Footprint Analyser
---

## Overview

WASTE·AI is an end-to-end computer vision system that classifies waste items into **Recyclable**, **Biodegradable**, and **Hazardous** categories in real time. The system combines a fine-tuned MobileNetV3 classifier with OpenCV-based rule overrides, a Gemini Vision gatekeeper, and a sustainability analytics dashboard that estimates CO2 savings and scrap value per scan.

**Goal:** Reduce per-capita landfill waste by 40% through AI-assisted smart segregation at source.

---

## Problem Context

| Statistic | Value |
|---|---|
| Waste incorrectly segregated globally | 90% |
| Municipal solid waste generated yearly | 2.01 billion tonnes |
| CO2 equivalent from mismanaged waste | 1.6 billion tonnes |

Manual sorting is error-prone, citizens lack real-time disposal guidance, and e-waste mixed with general waste causes toxic contamination. WASTE·AI addresses all three failure points.

---

## Waste Categories

| Bin | Classes |
|---|---|
| Recyclable | Plastic, Metal, Glass, Paper, Cardboard, Textile |
| Biodegradable | Biological matter, General Waste |
| Hazardous | E-Waste |

---

## System Components

### 1. Gemini Vision Barrier
Screens every incoming image before classification. Uses `gemini-2.5-flash` to determine whether the image contains a waste item. Non-waste inputs (people, animals, furniture, screens) are rejected at this stage with a reason string, preventing misclassification by the downstream model.

### 2. MobileNetV3 Waste Classifier
PyTorch fine-tuned model trained on 8 waste categories.

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3 — 18 Inverted Residual blocks |
| Input | 224×224 RGB → Resize + ToTensor |
| Head | Dropout(0.2) → Linear(1280 → 8 classes) |
| Optimizer | AdamW, lr=3e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=10 |
| Training hardware | Kaggle T4 GPU |
| Validation accuracy | **94.78%** |

Per-class F1 scores:

| Class | F1 |
|---|---|
| Cardboard | 0.95 |
| E-Waste | 0.94 |
| General Waste | 0.95 |
| Glass | 0.95 |
| Metal | 0.93 |
| Paper | 0.94 |
| Plastic | 0.91 |
| Textile | 0.99 |

### 3. OpenCV CV Rule Engine
Post-inference overrides based on pixel-level analysis:

- **Black plastic detection** — HSV brightness analysis flags NIR-invisible dark plastics and redirects to General Waste
- **Grease detection** — Adaptive thresholding on cardboard; grease area > 9% triggers General Waste override
- **Contamination rules** — Text hint parsing for scenarios including wet cardboard, laminated paper, aerosol cans, and thermal receipts

### 4. Uncertainty & OOD Filter
- Rejects predictions where `confidence < 0.50` and `entropy > 2.1` bits (out-of-distribution)
- Issues mixed-material alert when top-two class probability margin is below 0.22

### 5. Explainable AI (XAI) Module
Step-by-step reasoning panel surfaced per classification:
- Visual cues detected
- Confidence tier rationale (high / moderate / low)
- CV flag descriptions
- Override trigger and reason
- Bin assignment rationale

### 6. Groq LLM Reasoning
For low-confidence, overridden, or OOD predictions, a secondary explanation is generated via `llama-3.1-8b-instant` on Groq. Covers disposal correctness, environmental impact, and one actionable user tip.

### 7. Carbon Impact & Analytics Dashboard
Per-scan outputs:

| Output | Description |
|---|---|
| Estimated weight (kg) | Derived from object pixel area and material density lookup |
| CO2 saved (kg) | `(landfill_emission - recycled_emission) × est_weight` |
| Scrap value (INR) | Material rate (INR/kg) × estimated weight |
| Nearest facility | Local recycling centre lookup by waste type |

Session-level tracking: item count, cumulative CO2 savings, visual milestone indicator (Seedling → Bush → Tree), and scan history chip log.

CO2 savings reference values:

| Waste Type | CO2 Saved/kg | Scrap Rate |
|---|---|---|
| E-Waste | 20.0 kg | INR 40/kg |
| Metal | 9.5 kg | INR 18/kg |
| Plastic | 6.0 kg | INR 18/kg |
| Textile | 5.0 kg | INR 18/kg |
| Paper | 3.5 kg | INR 18/kg |

### 8. FastAPI Backend
REST API serving the full inference pipeline. Exposed publicly via ngrok tunnel for client access.

- `GET /info` — Returns class list, bin categories, and gatekeeper model name
- `POST /predict` — Accepts base64-encoded image + optional hint string; returns full classification result

### 9. HTML/JS Frontend
- Live camera stream with snap-to-classify
- Drag-and-drop image upload
- Optional text hint field (e.g., "greasy pizza box", "black plastic")
- SVG confidence ring (green above 68%, amber below)
- Collapsible XAI panel
- Session impact tracker

---

## Pipeline Flow

```
Capture Image
     |
     v
Gemini Vision Gatekeeper  -->  [Non-Waste] Return rejection
     |
     v
infer_transform → GPU Forward Pass → Softmax → 8-class probability vector
     |
     v
CV Rule Engine (HSV / grease / hint overrides)
     |
     v
Uncertainty Filter (OOD + margin check)
     |
     v
XAI Builder + CO2/Scrap Calculator
     |
     v
Render: Bin assignment · Confidence · CO2 · Scrap value · Facility · XAI steps
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, JavaScript, Camera API |
| Backend | FastAPI, Uvicorn, ngrok |
| CV & Preprocessing | OpenCV, PIL |
| Deep Learning | PyTorch, MobileNetV3 |
| Vision LLM | Google Gemini 2.5 Flash |
| Text LLM | Groq — LLaMA 3.1 8B Instant |
| Deployment | Kaggle T4 (free GPU tier) |

---

## Dataset

The model was trained on a combination of three publicly available waste classification datasets.

| Dataset | Source | Link |
|---|---|---|
| Garbage Classification (12 classes) | Kaggle — mostafaabla | [View Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) |
| TrashNet | Hugging Face — garythung | [View Dataset](https://huggingface.co/datasets/garythung/trashnet) |
| TrashBox | Kaggle — minhle13 | [View Dataset](https://www.kaggle.com/datasets/minhle13/trashbox) |

**Reference notebook:** [AI-Based Waste Detection — rahul12043](https://www.kaggle.com/code/rahul12043/ai-based-waste-detection)

Classes used in training: `Plastic`, `Metal`, `Glass`, `Paper`, `Cardboard`, `Textile`, `E-Waste`, `Biological / General Waste`

---

## Methodology

1. **Data preparation** — Images resized to 224×224, normalised with ImageNet mean/std. Standard augmentations applied during training (random horizontal flip, colour jitter).
2. **Transfer learning** — MobileNetV3-Large pretrained on ImageNet; final classification head replaced with `Dropout(0.2) → Linear(1280 → 8)`.
3. **Training** — 10 epochs, AdamW optimiser with cosine annealing LR schedule. Trained on Kaggle T4 GPU.
4. **Post-processing** — Softmax probabilities fed into CV rule engine and contamination lookup before final bin assignment.
5. **Uncertainty quantification** — Shannon entropy over the probability vector used alongside top-1 confidence to flag OOD and ambiguous inputs.
6. **Explainability** — Visual cue taxonomy per class mapped to confidence tier logic and bin rationale strings, assembled into an ordered XAI step list per prediction.

---
