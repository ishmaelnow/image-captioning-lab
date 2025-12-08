"""
Hugging Face Space entry point.

This file imports the Gradio interface from app_image_captioning.py
and exposes it as `demo`, which Spaces will run automatically.
"""

from app_image_captioning import build_interface

# Hugging Face Spaces looks for a variable named `demo`
demo = build_interface()
