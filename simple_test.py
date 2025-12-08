from PIL import Image
from app_image_captioning import caption_with_openai, polish_with_mistral

# ✅ Use a local file in THIS folder
img_path = "test_image.jpg"   # <-- IMPORTANT: no /mnt/data here

img = Image.open(img_path)

basic = caption_with_openai(img)
print("OpenAI caption:", basic)

polished = polish_with_mistral(basic, style="descriptive")
print("Mistral polished:", polished)
