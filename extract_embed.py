# src/extract_embed.py
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle

from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega modelo FaceNet pré-treinado
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def extract_embeddings(img_dir):
    embeddings = []
    labels = []

    for person in os.listdir(img_dir):
        person_dir = os.path.join(img_dir, person)
        if not os.path.isdir(person_dir): continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(img_tensor).cpu().numpy()
                embeddings.append(embedding[0])
                labels.append(person)
            except Exception as e:
                print(f"Erro com {img_path}: {e}")

    return np.array(embeddings), np.array(labels)

if __name__ == "__main__":
    img_dir = "rostos_cortados"
    embeddings, labels = extract_embeddings(img_dir)

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/embeddings.pkl", "wb") as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    print(f"[OK] Salvo: {len(labels)} embeddings.")
