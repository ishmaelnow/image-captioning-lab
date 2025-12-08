from PIL import Image, ImageDraw

# Create a simple 400x300 light-blue image
img = Image.new("RGB", (400, 300), "lightblue")
draw = ImageDraw.Draw(img)
draw.text((120, 140), "Test image", fill="black")

# Save it in this same folder
img.save("test_image.jpg")
print("Saved test_image.jpg in current folder")
