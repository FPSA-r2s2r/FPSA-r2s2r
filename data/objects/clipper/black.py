from PIL import Image
from pathlib import Path

tex_path = Path("./black.png")

Image.new("RGB", (16, 16), (40, 40, 40)).save(tex_path)