import cv2
import pytesseract
from pathlib import Path


# Windows only:
# Change this if Tesseract is installed somewhere else.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text(
    image_path: str,
    min_confidence: float = 40.0
) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale small text
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    # Reduce small noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Automatically separate text/background
    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    # OCR with confidence information
    data = pytesseract.image_to_data(
        processed,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words = []

    for text, confidence in zip(
        data["text"],
        data["conf"],
    ):
        text = text.strip()

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        if text and confidence >= min_confidence:
            words.append(text)

    return " ".join(words)


def save_text(text: str, output_path: str) -> None:
    Path(output_path).write_text(
        text,
        encoding="utf-8",
    )


if __name__ == "__main__":
    image_path = r"C:\Users\sacs7\Downloads\image.png"

    text = extract_text(image_path)

    print("\nExtracted text:")
    print("----------------")
    print(text)

    save_text(text, "extracted_text.txt")

    print("\nSaved to extracted_text.txt")
