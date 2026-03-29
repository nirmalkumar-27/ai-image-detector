from PIL import Image
import io

def read_image(file):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image