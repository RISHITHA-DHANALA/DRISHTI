
"""
utils/demo_data.py
--------------------
Generates synthetic, clearly-artificial "fundus-like" sample images so
the app is runnable end-to-end without requiring a real (and
license-restricted) retinal image dataset.

IMPORTANT: These are procedurally drawn placeholder images for UI/demo
purposes only — they are NOT real patient data and NOT suitable for
any clinical use. Replace `assets/sample_images/` with a properly
licensed, IRB/ethics-approved fundus dataset before any real
deployment or model training.
"""

import os
import random
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_IMAGES_DIR  # noqa: E402
from db.database import add_patient, get_patients  # noqa: E402

SEVERITY_LESION_COUNT = {
    "No DR": 0,
    "Mild DR": 4,
    "Moderate DR": 12,
    "Severe DR": 25,
}


def _draw_vessels(draw, size, rng):
    """Draw simple branching lines radiating from the optic disc to mimic retinal vessels."""
    cx, cy = size // 2, size // 2
    for _ in range(10):
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(size * 0.25, size * 0.45)
        x2 = cx + length * math.cos(angle)
        y2 = cy + length * math.sin(angle)
        width = rng.randint(1, 3)
        draw.line([(cx, cy), (x2, y2)], fill=(120, 40, 40), width=width)
        # a couple of branch offshoots
        for _ in range(2):
            branch_len = length * rng.uniform(0.3, 0.6)
            branch_angle = angle + rng.uniform(-0.6, 0.6)
            bx = x2 + branch_len * math.cos(branch_angle)
            by = y2 + branch_len * math.sin(branch_angle)
            draw.line([(x2, y2), (bx, by)], fill=(120, 40, 40), width=max(1, width - 1))


def generate_fundus_image(severity: str, size: int = 512, seed: int = None) -> Image.Image:
    """
    Procedurally generate a circular, fundus-photo-like image whose
    number of dark/red "lesion" spots scales with `severity`. This is a
    synthetic stand-in for a real retinal photograph.
    """
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (10, 10, 10))
    draw = ImageDraw.Draw(img)

    # Base retina disc: warm orange/red gradient look.
    base_color = (180, 70, 40)
    draw.ellipse([10, 10, size - 10, size - 10], fill=base_color)

    # Optic disc (bright, slightly off-center circle).
    disc_r = size * 0.09
    disc_cx = size * 0.5 + size * 0.08
    disc_cy = size * 0.5
    draw.ellipse(
        [disc_cx - disc_r, disc_cy - disc_r, disc_cx + disc_r, disc_cy + disc_r],
        fill=(230, 200, 150),
    )

    _draw_vessels(draw, size, rng)

    # Macula (subtle darker patch).
    macula_r = size * 0.07
    macula_cx = size * 0.5 - size * 0.12
    macula_cy = size * 0.5
    draw.ellipse(
        [macula_cx - macula_r, macula_cy - macula_r, macula_cx + macula_r, macula_cy + macula_r],
        fill=(140, 50, 35),
    )

    # Lesion-like spots (microaneurysm/hemorrhage stand-ins) — count
    # scales with severity to make demo predictions feel coherent.
    n_lesions = SEVERITY_LESION_COUNT.get(severity, 0)
    for _ in range(n_lesions):
        lx = rng.uniform(size * 0.15, size * 0.85)
        ly = rng.uniform(size * 0.15, size * 0.85)
        lr = rng.uniform(2, 7 if severity != "Severe DR" else 10)
        color = rng.choice([(60, 15, 10), (100, 10, 10), (200, 180, 60)])  # dark / red / yellow exudate
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=color)

    # Circular vignette mask so the image looks like a fundus photo,
    # not a square.
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([5, 5, size - 5, size - 5], fill=255)
    black_bg = Image.new("RGB", (size, size), (0, 0, 0))
    img = Image.composite(img, black_bg, mask)

    # Add a touch of fine texture/noise so the image has realistic
    # high-frequency content (a real fundus photo is not perfectly
    # smooth) — this keeps the blur-detection quality check meaningful
    # even for these synthetic placeholders.
    np_img = np.array(img).astype("int16")
    noise = rng_np.integers(-12, 12, size=np_img.shape, endpoint=True)
    np_img = np.clip(np_img + noise, 0, 255).astype("uint8")
    img = Image.fromarray(np_img)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def generate_all_samples(force: bool = False):
    """Create one demo sample image per DR severity class, if not already present."""
    from config import DR_CLASSES
    paths = {}
    for i, severity in enumerate(DR_CLASSES):
        filename = f"sample_{severity.replace(' ', '_').lower()}.png"
        filepath = os.path.join(SAMPLE_IMAGES_DIR, filename)
        if force or not os.path.exists(filepath):
            img = generate_fundus_image(severity, size=512, seed=100 + i)
            img.save(filepath)
        paths[severity] = filepath
    return paths


def seed_demo_patients():
    """Insert a handful of demo patients if the patients table is empty."""
    existing = get_patients()
    if existing:
        return

    demo_patients = [
        ("PT-1001", "Lakshmi Narayanan", 58, 12, "Yes"),
        ("PT-1002", "Ramesh Yadav", 47, 6, "No"),
        ("PT-1003", "Sunita Kumari", 63, 18, "Yes"),
        ("PT-1004", "Venkata Rao", 39, 3, "No"),
    ]
    for code, name, age, duration, prev in demo_patients:
        add_patient(code, name, age, duration, prev, registered_by="system_seed")
