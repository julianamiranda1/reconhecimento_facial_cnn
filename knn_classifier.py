# src/knn_classifier.py
import torch
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import sys
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_embeddings(path="embeddings/embeddings.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["labels"]

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def recognize_face(img_path, embeddings, labels, threshold=0.6):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embed = model(img_tensor).cpu().numpy()[0]

    sims = [cosine_similarity(embed, e) for e in embeddings]
    max_sim = max(sims)
    max_idx = np.argmax(sims)

    if max_sim > threshold:
        name = labels[max_idx]
    else:
        name = "Desconhecido"

    # Desenha na imagem
    draw = ImageDraw.Draw(img)

    try:
        # Tenta carregar fonte do sistema, senão usa padrão
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    text = f"{name} ({max_sim:.2f})"

    # Define posição do texto (topo da imagem)
    text_pos = (10, 10)

    # Desenha retângulo (contorno da imagem inteira, pois já está cortada)
    width, height = img.size
    draw.rectangle([0, 0, width-1, height-1], outline="green", width=3)
    draw.text(text_pos, text, fill="green", font=font)

    return img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python knn_classifier.py caminho/para/rosto_cortado.jpg")
        sys.exit(1)

    test_img = sys.argv[1]
    embeddings, labels = load_embeddings()
    img_annotated = recognize_face(test_img, embeddings, labels)

    plt.figure(figsize=(6,6))
    plt.imshow(img_annotated)
    plt.axis('off')
    plt.title("Reconhecimento Facial")
    plt.show()
