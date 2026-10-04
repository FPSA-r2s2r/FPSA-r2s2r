from PIL import Image
from pathlib import Path

tex_path = Path("./blue.png")

Image.new("RGB", (16, 16), (10, 41, 137)).save(tex_path)