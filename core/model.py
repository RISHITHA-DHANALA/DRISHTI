
"""
core/model.py
--------------
DR classification model wrapper.

Design intent
-------------
This module exposes a single class, `DRModel`, with a stable interface:

    model = DRModel()
    dr_class, confidence, probs = model.predict(image_bgr)

so that a validated, clinically-tested model can be swapped in later
WITHOUT changing any UI or workflow code. To plug in a real model:

  1. Place a trained Keras model at `models/dr_model.h5` (4-class
     softmax output matching config.DR_CLASSES order), or a PyTorch
     model at `models/dr_model.pt`.
  2. Set `MODE = "real"` will be selected automatically if a model
     file is found and the corresponding framework is installed.
  3. Update `_predict_real()` if your architecture / preprocessing
     differs from the assumed 224x224 RGB, ImageNet-style normalization.

If no trained model file is present, the app runs in DEMO/MOCK mode:
predictions are derived from simple, transparent image statistics
(NOT a diagnostic algorithm). This is clearly surfaced in the UI via
`config.IS_PROTOTYPE` and the "demo_mode_notice" translation string.
"""

import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DR_CLASSES, BASE_DIR  # noqa: E402

MODEL_DIR = os.path.join(BASE_DIR, "models")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "dr_model.h5")
TORCH_MODEL_PATH = os.path.join(MODEL_DIR, "dr_model.pt")
INPUT_SIZE = (224, 224)


class DRModel:
    """Diabetic retinopathy severity classifier with a real/mock dual mode."""

    def __init__(self):
        self.mode = "mock"
        self.framework = None
        self.model = None

        if os.path.exists(KERAS_MODEL_PATH):
            try:
                import tensorflow as tf  # noqa: F401
                self.model = tf.keras.models.load_model(KERAS_MODEL_PATH)
                self.mode = "real"
                self.framework = "tensorflow"
            except Exception:
                self.mode = "mock"
        elif os.path.exists(TORCH_MODEL_PATH):
            try:
                import torch  # noqa: F401
                self.model = torch.load(TORCH_MODEL_PATH, map_location="cpu")
                self.model.eval()
                self.mode = "real"
                self.framework = "pytorch"
            except Exception:
                self.mode = "mock"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, image_bgr: np.ndarray):
        """
        Returns:
            dr_class (str): one of config.DR_CLASSES
            confidence (float): 0-100
            probs (dict): class -> probability (0-1), sums to ~1
        """
        if self.mode == "real":
            return self._predict_real(image_bgr)
        return self._predict_mock(image_bgr)

    # ------------------------------------------------------------------
    # Real model path (used automatically once a trained model exists)
    # ------------------------------------------------------------------
    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_bgr, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    def _predict_real(self, image_bgr: np.ndarray):
        batch = self._preprocess(image_bgr)
        if self.framework == "tensorflow":
            probs = self.model.predict(batch, verbose=0)[0]
        else:  # pytorch
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2))
                probs = torch.softmax(self.model(tensor), dim=1).numpy()[0]

        idx = int(np.argmax(probs))
        dr_class = DR_CLASSES[idx]
        confidence = float(probs[idx]) * 100
        prob_dict = {cls: float(p) for cls, p in zip(DR_CLASSES, probs)}
        return dr_class, round(confidence, 1), prob_dict

    # ------------------------------------------------------------------
    # Mock / demo path — heuristic, NOT a diagnostic algorithm.
    # Uses simple, explainable image statistics (lesion-like dark/red
    # blob density, contrast, vessel-edge density) so behaviour is
    # transparent to reviewers, and produces a plausible 4-class
    # softmax-style distribution for demo purposes.
    # ------------------------------------------------------------------
    def _predict_mock(self, image_bgr: np.ndarray):
        img = cv2.resize(image_bgr, INPUT_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Restrict analysis to the retina disc itself, excluding the
        # black background/vignette outside a fundus photo — otherwise
        # the background gets counted as one giant "dark blob" and
        # swallows the real lesion-like spots into a single contour.
        retina_mask = (gray > 20).astype("uint8") * 255

        # 1. Lesion-like dark/red blob density (hemorrhages/microaneurysms
        #    tend to appear as small dark or reddish spots in a fundus image).
        _, dark_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_thresh, retina_mask)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        small_blobs = [c for c in contours if 3 < cv2.contourArea(c) < 150]
        blob_density = min(len(small_blobs) / 12.0, 1.0)  # normalize 0-1

        # 2. Local contrast / texture irregularity (edge density),
        #    again restricted to inside the retina disc. Weighted lower
        #    than blob density since it varies less across severities
        #    in practice — lesion count is the stronger signal.
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, retina_mask)
        edge_density = float(np.sum(edges > 0)) / float(np.sum(retina_mask > 0) + 1e-6)

        # 3. Overall severity score combining both signals, deterministic
        #    given the same input image (reproducible for demos).
        severity_score = 0.85 * blob_density + 0.15 * min(edge_density * 5, 1.0)
        severity_score = float(np.clip(severity_score, 0.0, 1.0))

        # Map continuous severity score -> 4-class probability distribution
        # via soft binning around the score, so nearby classes retain
        # some probability mass (mimics a real softmax output).
        centers = np.array([0.10, 0.35, 0.62, 0.88])  # class "centers" on 0-1 scale
        distances = np.abs(centers - severity_score)
        raw = np.exp(-distances * 6.0)  # sharper peak = more "confident" model
        probs = raw / raw.sum()

        idx = int(np.argmax(probs))
        dr_class = DR_CLASSES[idx]
        confidence = float(probs[idx]) * 100
        prob_dict = {cls: float(p) for cls, p in zip(DR_CLASSES, probs)}
        return dr_class, round(confidence, 1), prob_dict
