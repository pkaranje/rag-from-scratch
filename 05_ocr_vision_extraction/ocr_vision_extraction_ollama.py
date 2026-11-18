import base64
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytesseract
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# override=True so re-running this cell picks up edits to .env without restarting the kernel
load_dotenv(override=True)

DATA_DIR = (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "data"
DATA_DIR.mkdir(exist_ok=True)
print(f"Data directory: {DATA_DIR.resolve()}")

OUT_PATH = DATA_DIR / "extracted_chunks.json"
print(f"Output path: {OUT_PATH.resolve()}")

TRANSCRIBE_PROMPT = (
    "Transcribe all text visible in this image exactly as written. "
    "If it looks like handwriting or a whiteboard, still transcribe your best "
    "reading. Output only the transcribed text, nothing else."
)

VISION_MODEL = "gemma3:4b"  # small, fast, and free for local use; see https://ollama.com/models
VISION_LLM = ChatOllama(model=VISION_MODEL, temperature=0, reasoning=False)


# Generate the sample images


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # try real system fonts first — PIL's bitmap default font looks nothing like real text
    # and would make OCR/vision comparisons meaningless
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _draw_lines(
    img: Image.Image,
    lines: list[str],
    *,
    x: int,
    y: int,
    spacing: int,
    size: int,
    fill: str | int | tuple[int, int, int],
    x_jitter: Callable[[], int] = lambda: 0,
) -> None:
    draw = ImageDraw.Draw(img)
    font = _font(size)  # load once per image, not once per line
    for i, line in enumerate(lines):
        draw.text((x + x_jitter(), y + i * spacing), line, fill=fill, font=font)


def _save(img: Image.Image, name: str) -> None:
    img.convert("RGB").save(DATA_DIR / name)
    print(f"wrote {name}")


def make_clean_screenshot() -> None:
    img = Image.new("RGB", (700, 300), "white")
    lines = [
        "def calculate_total(items):",
        '    """Sum item prices after a 7% tax."""',
        "    subtotal = sum(item.price for item in items)",
        "    return round(subtotal * 1.07, 2)",
    ]
    _draw_lines(img, lines, x=20, y=20, spacing=30, size=18, fill="black")
    _save(img, "clean_screenshot.png")


def make_scanned_page() -> None:
    img = Image.new("L", (700, 500), 245)
    lines = [
        "Warehouse Safety Bulletin",
        "",
        "All forklift operators must complete recertification",
        "training by the end of this quarter. Sessions are held",
        "every Tuesday at 9 AM in the east training room.",
    ]
    _draw_lines(img, lines, x=40, y=40, spacing=35, size=22, fill=20)
    # simulate scan noise + slight skew
    img = img.rotate(1.5, expand=True, fillcolor=245)
    noise = Image.effect_noise(img.size, 18)
    # blend at a low alpha so the noise is subtle grain, not enough to obscure the text
    img = Image.blend(img, noise, alpha=0.05)
    _save(img, "scanned_page.png")


def make_receipt() -> None:
    img = Image.new("L", (300, 500), 250)
    lines = [
        "CORNER MARKET",
        "12 Maple St",
        "------------------------",
        "Bread          3.49",
        "Milk 1gal      4.19",
        "Eggs dozen     3.99",
        "Coffee beans   9.50",
        "------------------------",
        "TOTAL         21.17",
    ]
    _draw_lines(img, lines, x=15, y=20, spacing=28, size=16, fill=10)
    # thermal-receipt paper is rarely perfectly sharp — a light blur mimics that
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    _save(img, "receipt.png")


def make_whiteboard() -> None:
    img = Image.new("RGB", (600, 400), (250, 250, 245))
    lines = ["Sprint Goals:", "- ship OCR demo", "- fix flaky test", "- review PR #42"]
    # local seeded RNG: deterministic jitter so the whiteboard looks the same on every run,
    # no matter what else used the global random state first
    rng = random.Random(42)
    # per-line horizontal jitter mimics the unevenness of real handwriting/marker strokes
    _draw_lines(
        img,
        lines,
        x=60,
        y=40,
        spacing=60,
        size=28,
        fill=(30, 60, 140),
        x_jitter=lambda: rng.randint(-3, 3),
    )
    img = img.rotate(-2, expand=True, fillcolor=(250, 250, 245))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    _save(img, "whiteboard.png")


def make_low_res_scan() -> None:
    img = Image.new("L", (700, 300), 240)
    _draw_lines(img, ["Return Authorization Form"], x=30, y=30, spacing=0, size=24, fill=15)
    _draw_lines(
        img,
        ["Order #: 88213", "Reason: item damaged in transit"],
        x=30,
        y=80,
        spacing=40,
        size=20,
        fill=15,
    )
    # downsample then upsample to simulate a low-res scan, then blur
    low_res = img.resize((175, 75)).resize((700, 300))
    low_res = low_res.filter(ImageFilter.GaussianBlur(1.2))
    _save(low_res, "low_res_scan.png")


# Extract text: Tesseract OCR and a vision LLM


def ocr_raw(image_path: Path) -> str:
    """Run Tesseract directly on the image, no preprocessing."""
    with Image.open(image_path) as img:
        return str(pytesseract.image_to_string(img)).strip()


def ocr_preprocessed(image_path: Path) -> str:
    """Grayscale + binarize (simple fixed threshold) before OCR — this is
    the single highest-leverage preprocessing step for scanned documents."""
    with Image.open(image_path) as src:
        img = src.convert("L")  # grayscale
    # stretch the contrast range before thresholding, so faint scans don't
    # collapse entirely to one color once we binarize
    img = ImageOps.autocontrast(img)
    threshold = 150
    # hard black/white cutoff — removes anti-aliasing/noise that confuses Tesseract's
    # character segmentation, at the cost of being sensitive to the threshold value
    img = img.point([255 if value > threshold else 0 for value in range(256)])
    return str(pytesseract.image_to_string(img)).strip()


def vision_transcribe(image_path: Path) -> str:
    # vision models take images as inline base64 data URIs, not file paths
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    # multimodal message: a text part (the instruction) plus an image part in one HumanMessage
    message = HumanMessage(
        content=[
            {"type": "text", "text": TRANSCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )
    try:
        return VISION_LLM.invoke([message]).text.strip()
    except ResponseError as err:
        # e.g. "token repeat limit reached" on the receipt's "-----" lines: skip, don't crash
        print(f"  vision LLM failed on {image_path.name}: {err}")
        return ""


def _sample_images() -> list[Path]:
    return sorted(DATA_DIR.glob("*.png"))


def ocr_extract() -> None:
    for image_path in _sample_images():
        print(f"\n--- {image_path.name} ---")
        print("raw:          ", ocr_raw(image_path)[:80].replace("\n", " "))
        print("preprocessed: ", ocr_preprocessed(image_path)[:80].replace("\n", " "))


def vision_llm_extract() -> None:
    for image_path in _sample_images():
        print(f"\n--- {image_path.name} ---")
        print(vision_transcribe(image_path))


# Compare methods and save the best result as chunks

METHODS: dict[str, Callable[[Path], str]] = {
    "ocr_raw": ocr_raw,
    "ocr_preprocessed": ocr_preprocessed,
    "vision_llm": vision_transcribe,
}


def compare_one(image_path: Path) -> dict[str, Any]:
    # wall-clock timing for each method so we can weigh accuracy against latency/cost,
    # not just compare the transcribed text
    result: dict[str, Any] = {"image": image_path.name}
    for name, extract in METHODS.items():
        t0 = time.perf_counter()
        text = extract(image_path)
        result[name] = {"text": text, "seconds": round(time.perf_counter() - t0, 2)}
    return result


def compare_tesseract_vs_vision_llm() -> None:
    results = []
    for image_path in _sample_images():
        print(f"\n{'=' * 70}\n{image_path.name}\n{'=' * 70}")
        result = compare_one(image_path)
        results.append(result)
        for method in METHODS:
            r = result[method]
            snippet = r["text"][:100].replace("\n", " ")
            print(f"  {method:16s} ({r['seconds']:.2f}s): {snippet}")

    # Store the vision-LLM transcription as the "best result" chunk — for
    # noisy/handwritten sources it consistently beats raw OCR, at higher cost
    chunks = [
        {
            "text": r["vision_llm"]["text"],
            "metadata": {
                "image_path": r["image"],
                "extraction_method": "vision_llm",
            },
        }
        for r in results
        if r["vision_llm"]["text"]  # skip images the vision LLM failed on
    ]
    if not chunks:
        # don't clobber a previous good file with "[]" when every vision call failed
        raise RuntimeError(
            f"Vision LLM ({VISION_MODEL}) failed on every image; see errors above. "
            f"{OUT_PATH.name} left unchanged."
        )
    OUT_PATH.write_text(json.dumps(chunks, indent=2))
    print(f"\nWrote {len(chunks)} chunks -> {OUT_PATH}")


def main() -> None:
    make_clean_screenshot()
    make_scanned_page()
    make_receipt()
    make_whiteboard()
    make_low_res_scan()
    print("\nDone. These 5 images are gitignored — rerun this cell any time.")
    # compare_* already runs both OCR variants and the vision LLM on every image;
    # ocr_extract()/vision_llm_extract() stay available as standalone steps but would
    # repeat that work (and double the slow vision calls) if run here too
    compare_tesseract_vs_vision_llm()


if __name__ == "__main__":
    main()
