from PIL import Image
from pathlib import Path

tex_path = Path("./red.png")

Image.new("RGB", (16, 16), (193, 46, 31)).save(tex_path)