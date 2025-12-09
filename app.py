"""
Hugging Face Space entry point for the Image Captioning app.

This file is SELF-CONTAINED:
- Loads API keys from environment variables
- Defines OpenAI vision captioning
- Defines Mistral caption polishing
- Builds a Gradio interface and exposes it as `demo`
"""

import base64
import io
import os

from dotenv import load_dotenv
from openai import OpenAI
from mistralai import Mistral
from PIL import Image
import gradio as gr

# ==============================
# 1. Load environment variables
# ==============================
# On Hugging Face, these come from "Secrets" in the Space settings.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
MISTRAL_TEXT_MODEL = os.getenv("MISTRAL_TEXT_MODEL", "mistral-small-latest")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it as a secret in the Space.")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is missing. Set it as a secret in the Space.")

# Create clients for both providers
openai_client = OpenAI(api_key=OPENAI_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)


# =========================================
# 2. Helper: convert PIL image -> base64
# =========================================
def pil_to_base64_str(image: Image.Image) -> str:
    """
    Take a PIL Image and return a base64-encoded string.
    This is how we send local images to the OpenAI vision model.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


# =========================================
# 3. OpenAI: generate a basic caption
# =========================================
def caption_with_openai(image: Image.Image) -> str:
    """
    Ask OpenAI's vision model to describe the image in one short sentence.
    """
    img_b64 = pil_to_base64_str(image)

    response = openai_client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in one short, clear sentence "
                            "for a non-technical person."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=80,
    )

    caption = response.choices[0].message.content

    # In the new SDK, content can be a string or a list; handle both
    if isinstance(caption, str):
        return caption.strip()
    else:
        text_parts = []
        for part in caption:
            if hasattr(part, "text"):
                text_parts.append(part.text)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return " ".join(text_parts).strip()


# =========================================
# 4. Mistral: polish / enhance the caption
# =========================================
def polish_with_mistral(basic_caption: str, style: str = "descriptive") -> str:
    """
    Use Mistral (text-only) to rewrite or enhance the caption.
    style can be 'descriptive', 'short', or 'fun'.
    """
    if style == "descriptive":
        instruction = (
            "Rewrite this caption to be a bit more descriptive, "
            "but still under 25 words, and easy to understand:\n"
        )
    elif style == "short":
        instruction = "Rewrite this caption to be very short (max 10 words) and clear:\n"
    elif style == "fun":
        instruction = (
            "Rewrite this caption in a friendly, playful tone (max 20 words):\n"
        )
    else:
        instruction = "Improve this caption while keeping the meaning the same:\n"

    prompt = instruction + basic_caption

    response = mistral_client.chat.complete(
        model=MISTRAL_TEXT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=80,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# =========================================
# 5. Main pipeline used by Gradio
# =========================================
def generate_caption(image: Image.Image, provider_mode: str, style: str) -> str:
    """
    This is what Gradio calls.
    1) Always get a basic caption from OpenAI.
    2) Optionally send it to Mistral to polish/transform it.
    """
    if image is None:
        return "Please upload an image first."

    try:
        basic_caption = caption_with_openai(image)
    except Exception as e:
        return f"Error from OpenAI: {e}"

    if provider_mode == "OpenAI only":
        return basic_caption

    if provider_mode == "OpenAI + Mistral (polish)":
        try:
            polished = polish_with_mistral(basic_caption, style=style)
            return (
                f"Basic caption (OpenAI): {basic_caption}\n\n"
                f"Polished (Mistral): {polished}"
            )
        except Exception as e:
            return f"OpenAI caption: {basic_caption}\n\nError from Mistral: {e}"

    # Fallback
    return basic_caption


# =========================================
# 6. Build Gradio interface and expose `demo`
# =========================================
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🖼️ Image Captioning with OpenAI + Mistral")
        gr.Markdown(
            "1. Upload an image\n"
            "2. Choose how you want to generate the caption\n"
            "3. Click **Generate Caption**"
        )

        with gr.Row():
            image_input = gr.Image(
                label="Upload an image",
                type="pil",
            )

            with gr.Column():
                provider_mode = gr.Radio(
                    label="Provider mode",
                    choices=[
                        "OpenAI only",
                        "OpenAI + Mistral (polish)",
                    ],
                    value="OpenAI only",
                )

                style = gr.Radio(
                    label="Mistral style (used only when polishing)",
                    choices=["descriptive", "short", "fun"],
                    value="descriptive",
                )

                generate_btn = gr.Button("Generate Caption", variant="primary")
                output_box = gr.Textbox(
                    label="Caption",
                    placeholder="Your image caption will appear here...",
                    lines=4,
                )

        generate_btn.click(
            fn=generate_caption,
            inputs=[image_input, provider_mode, style],
            outputs=output_box,
        )

    return demo


# Hugging Face Spaces looks for a variable named `demo`
demo = build_interface()

# Optional: allow local run with `python app.py`
if __name__ == "__main__":
    demo.launch()
