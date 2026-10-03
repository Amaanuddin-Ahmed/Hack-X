import os
import time
from functools import lru_cache
from io import BytesIO

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError

from .schemas import FoodAnalysis

load_dotenv()

PROMPT = """
You are the food-label extraction component of a health-assessment prototype.
Analyze ONLY what is visible in the supplied image.

Tasks:
1. Decide whether the image shows food or a packaged food product.
2. Decide whether a nutrition/ingredient label is visible.
3. Extract visible nutrition values and ingredients.
4. Never invent or estimate a nutrition value that is not clearly visible.
5. Use null for unknown numeric/string nutrition fields.
6. Put unavailable field names in missing_fields.
7. If text is blurry, cropped, obscured, or ambiguous, lower confidence and explain briefly in notes.
8. Do NOT provide medical advice in this smoke test.
9. Keep ingredients faithful to visible text; do not add ingredients from general knowledge.

Return data matching the provided schema.
"""


class GemmaConfigurationError(RuntimeError):
    pass


class GemmaUnavailableError(RuntimeError):
    pass


class ImageValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def validate_image(payload: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageValidationError("Upload JPG, PNG, or WEBP only.")
            return source.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc


def analyze_food_image(image_bytes: bytes) -> FoodAnalysis:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMMA_MODEL")

    if not api_key:
        raise GemmaConfigurationError("GEMINI_API_KEY is not configured")
    if not model or model.startswith("replace_with_"):
        raise GemmaConfigurationError("GEMMA_MODEL is not configured. Use the exact Gemma 4 model ID confirmed by the organizers.")

    image = validate_image(image_bytes)
    client = _client(api_key)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[PROMPT, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FoodAnalysis,
                    temperature=0.1,
                ),
            )
            return FoodAnalysis.model_validate_json(response.text)
        except Exception as exc:
            transient = any(code in str(exc) for code in ("429", "500", "502", "503", "504", "UNAVAILABLE"))
            if not transient or attempt == 2:
                if transient:
                    raise GemmaUnavailableError("Gemma is temporarily unavailable. Please retry shortly.") from exc
                raise
            time.sleep(2**attempt)
