
"""
core/gradcam.py
-----------------
Explainability layer.

- If a real Keras CNN is loaded (DRModel.mode == "real", framework ==
  "tensorflow"), this computes a genuine Grad-CAM heatmap from the last
  convolutional layer's activations and gradients (Selvaraju et al., 2017).

- In DEMO/MOCK mode, we generate a *saliency-style* heatmap from local
  image statistics (dark/red blob regions + edge density — the same
  signals the mock classifier uses) so the visual explanation stays
  consistent with what actually drove the mock prediction. This is
  clearly NOT a trained-model Grad-CAM and is labeled as such in the UI.

Both paths return an RGB heatmap overlay ready to display next to the
original image.
"""

import cv2
import numpy as np


def _overlay_heatmap(original_bgr: np.ndarray, heatmap_gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize heatmap to original size, colorize, and blend with original image."""
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap_gray, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)
    return overlay


def generate_mock_saliency(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a saliency-style heatmap from dark-blob density and edge
    strength, blurred for a smooth "attention" look. Used only when
    no real trained model is available.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Restrict attention to inside the retina disc, excluding the black
    # background outside a fundus photo (keeps this consistent with the
    # mock classifier's own logic in core/model.py).
    retina_mask = (gray > 20).astype("uint8") * 255

    # Dark/red lesion-like regions -> higher "attention"
    _, dark_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.bitwise_and(dark_thresh, retina_mask).astype("float32") / 255.0

    # Edge / texture irregularity -> also contributes to attention
    edges = cv2.bitwise_and(cv2.Canny(gray, 50, 150), retina_mask).astype("float32") / 255.0

    raw_saliency = 0.7 * dark_mask + 0.3 * edges
    saliency = cv2.GaussianBlur(raw_saliency, (31, 31), 0)

    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return _overlay_heatmap(image_bgr, saliency)


def generate_real_gradcam(model, image_bgr: np.ndarray, preprocess_fn, last_conv_layer_name: str = None):
    """
    Real Grad-CAM for a Keras model. Only invoked when DRModel is in
    'real' + 'tensorflow' mode.

    Args:
        model: loaded tf.keras.Model
        image_bgr: original BGR image (as read by OpenCV)
        preprocess_fn: function(image_bgr) -> preprocessed batch (1,H,W,3)
        last_conv_layer_name: name of last conv layer; auto-detected if None
    """
    import tensorflow as tf

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    batch = preprocess_fn(image_bgr)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return _overlay_heatmap(image_bgr, heatmap)


def generate_explanation(dr_model, image_bgr: np.ndarray) -> np.ndarray:
    """
    Convenience entry point used by the app: picks real Grad-CAM or the
    mock saliency fallback depending on the loaded model's mode.
    """
    if dr_model.mode == "real" and dr_model.framework == "tensorflow":
        try:
            return generate_real_gradcam(dr_model.model, image_bgr, dr_model._preprocess)
        except Exception:
            # Fall back gracefully if anything about the architecture
            # doesn't match assumptions (e.g., no 4D conv layer found).
            return generate_mock_saliency(image_bgr)
    return generate_mock_saliency(image_bgr)
