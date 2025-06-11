# src/multi_face_recognition.py
import torch
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelo de reconhecimento facial (FaceNet)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Modelo de detecção facial (MTCNN)
mtcnn = MTCNN(keep_all=True, device=device)

# Carregamento de embeddings
def load_embeddings(path="embeddings/embeddings.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["labels"]

# Similaridade de cosseno
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# Reconhecimento
def recognize_faces(img_path, embeddings, labels, threshold=0.6):
    img = Image.open(img_path).convert('RGB')
    boxes, faces = mtcnn.detect(img, landmarks=False)
    results = []

    if faces is None:
        print("Nenhum rosto detectado.")
        return []

    faces = mtcnn(img)  # retorna os rostos alinhados e cortados

    for face, box in zip(faces, boxes):
        if face is None:
            continue
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embed = facenet(face).cpu().numpy()[0]

        sims = [cosine_similarity(embed, e) for e in embeddings]
        max_sim = max(sims)
        idx = np.argmax(sims)

        name = labels[idx] if max_sim > threshold else "Desconhecido"
        results.append((name, max_sim, box))

    return results, img

# Visualização
def draw_boxes(img, results):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=40)  # fonte maior, precisa estar disponível no sistema

    for name, sim, box in results:
        box = [int(b) for b in box]
        draw.rectangle(box, outline='green', width=3)
        draw.text((box[0], box[1] - 25), f"{name} ({sim:.2f})", fill='green', font=font)
    return img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python multi_face_recognition.py C:/Users/Dell/Downloads/IMG_3032.jpg")
        sys.exit(1)

    embeddings, labels = load_embeddings()
    results, img = recognize_faces(sys.argv[1], embeddings, labels)

    if results:
        img_annotated = draw_boxes(img, results)
        plt.figure(figsize=(12, 12))  # ajusta tamanho da janela de exibição
        plt.imshow(img_annotated)
        plt.axis('off')
        plt.title("Rostos reconhecidos")
        plt.show()
