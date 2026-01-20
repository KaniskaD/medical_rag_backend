import torch
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer

_clip_model = SentenceTransformer("clip-ViT-B-32")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_image_embedding(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return _clip_model.encode([img], convert_to_numpy=True)[0].astype("float32")


def generate_image_caption(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    emb = _clip_model.encode([img], convert_to_numpy=True)

    return "Medical image uploaded (X-ray, scan, chart, or document). Extracted visual features available."
