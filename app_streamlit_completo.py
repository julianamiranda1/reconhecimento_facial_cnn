import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
import tempfile
import os
import cv2
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)
DB_PATH = "embeddings/embeddings.pkl"

def load_database(path=DB_PATH):
    if not os.path.exists(path):
        return [], []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["labels"]

def save_database(embeddings, labels, path=DB_PATH):
    data = {"embeddings": embeddings, "labels": labels}
    with open(path, "wb") as f:
        pickle.dump(data, f)

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def extract_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    if isinstance(face, list) or len(face.shape) == 3:
        face = face.unsqueeze(0)
    face = face.to(device)
    with torch.no_grad():
        emb = facenet(face).cpu().numpy()[0]
    return emb

def add_new_user(name, img_paths, embeddings, labels):
    new_embs = []
    for img_path in img_paths:
        emb = extract_embedding(img_path)
        if emb is not None:
            new_embs.append(emb)
    if not new_embs:
        return embeddings, labels, False
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    embeddings.extend(new_embs)
    labels.extend([name] * len(new_embs))
    return embeddings, labels, True

def recognize_faces(img_pil, embeddings, labels, threshold=0.6):
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None:
        return [], img_pil
    faces = mtcnn(img_pil)
    results = []
    for face, box in zip(faces, boxes):
        if face is None:
            continue
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = facenet(face).cpu().numpy()[0]
        sims = [cosine_similarity(emb, e) for e in embeddings]
        max_sim = max(sims)
        idx = np.argmax(sims)
        name = labels[idx] if max_sim > threshold else "Desconhecido"
        results.append((name, max_sim, box))
    return results, img_pil

def draw_boxes(img, results):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except:
        font = ImageFont.load_default()
    for name, sim, box in results:
        box = [int(b) for b in box]
        draw.rectangle(box, outline='green', width=3)
        draw.text((box[0], box[1]-25), f"{name} ({sim:.2f})", fill='green', font=font)
    return img

# --- STREAMLIT UI ---
st.set_page_config(page_title="Reconhecimento Facial", layout="wide")
st.title("💻 Sistema de Reconhecimento Facial")

tab = st.tabs(["Cadastro de Usuário", "Reconhecimento Facial"])

with tab[0]:
    st.header("Cadastro Incremental de Usuários")
    nome = st.text_input("Nome do novo usuário")
    imagens = st.file_uploader("Upload de uma ou mais imagens", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if st.button("Cadastrar"):
        if not nome:
            st.error("Insira um nome.")
        elif not imagens:
            st.error("Envie pelo menos uma imagem.")
        else:
            temp_paths = []
            for img in imagens:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img.read())
                    temp_paths.append(tmp.name)
            embs, labs = load_database()
            embs, labs, ok = add_new_user(nome, temp_paths, embs, labs)
            if ok:
                save_database(embs, labs)
                st.success(f"Usuário {nome} cadastrado com {len(temp_paths)} imagens.")
            else:
                st.warning("Nenhum rosto detectado.")
            for p in temp_paths:
                os.remove(p)

with tab[1]:
    st.header("Reconhecimento Facial")
    embs, labs = load_database()
    if not embs:
        st.warning("Cadastre usuários primeiro.")
    else:
        modo = st.radio("Modo:", ["Upload de Imagem", "Webcam"])
        if modo == "Upload de Imagem":
            img_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
            if img_file:
                img = Image.open(img_file).convert('RGB')
                resultados, img = recognize_faces(img, embs, labs)
                img_annot = draw_boxes(img, resultados)
                st.image(img_annot, caption="Resultado", use_column_width=True)
        else:
            start = st.button("Iniciar Webcam")
            stop = st.button("Parar Webcam")
            frame_slot = st.empty()

            if start:
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Erro ao acessar webcam.")
                        break
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    resultados, _ = recognize_faces(img_pil, embs, labs)
                    for nome, sim, box in resultados:
                        box = [int(b) for b in box]
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                        cv2.putText(frame, f"{nome} ({sim:.2f})", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    frame_slot.image(frame, channels="BGR")
                    time.sleep(0.1)
                    if stop:
                        break
                cap.release()
                frame_slot.empty()