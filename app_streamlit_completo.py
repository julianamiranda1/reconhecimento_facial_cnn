import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
import tempfile
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelos
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

DB_PATH = "embeddings/embeddings.pkl"

# -------- Banco de Dados --------
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

# -------- Função de Similaridade --------
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# -------- Processamento Embeddings --------
def extract_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    # face pode ser um tensor com múltiplas faces, mas como usamos keep_all=True, ele retorna lista
    # Vamos garantir que é um tensor 4D para o facenet
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
    
    # embeddings vem como lista ou array numpy, converte para lista
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    embeddings.extend(new_embs)
    
    # labels é lista normal
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
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    for name, sim, box in results:
        box = [int(b) for b in box]
        draw.rectangle(box, outline='green', width=3)
        draw.text((box[0], box[1]-25), f"{name} ({sim:.2f})", fill='green', font=font)
    return img

# -------- Reconhecimento Webcam --------
def recognize_faces_cv2(frame, embeddings, labels, threshold=0.6):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results, _ = recognize_faces(img_pil, embeddings, labels, threshold)
    for name, sim, box in results:
        box = [int(b) for b in box]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({sim:.2f})", (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return frame

# -------- Streamlit UI --------
st.title("💻 Sistema de Reconhecimento Facial - MVP Completo")

tab = st.tabs(["Cadastro de Usuário", "Reconhecimento Facial"])

with tab[0]:
    st.header("Cadastro Incremental de Usuários")
    nome = st.text_input("Nome do novo usuário")
    imagens = st.file_uploader("Faça upload de uma ou mais imagens", type=['jpg','jpeg','png'], accept_multiple_files=True)
    if st.button("Cadastrar Usuário"):
        if not nome:
            st.error("Por favor, insira o nome do usuário.")
        elif not imagens:
            st.error("Por favor, faça upload de ao menos uma imagem.")
        else:
            temp_paths = []
            for img in imagens:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img.read())
                    temp_paths.append(tmp.name)

            embs, labs = load_database()
            embs, labs, success = add_new_user(nome, temp_paths, embs, labs)
            if success:
                save_database(embs, labs)
                st.success(f"Usuário '{nome}' cadastrado com sucesso com {len(temp_paths)} imagens.")
            else:
                st.warning("Nenhuma face detectada nas imagens. Tente outras imagens.")

            for path in temp_paths:
                os.remove(path)

with tab[1]:
    st.header("Reconhecimento Facial")
    embs, labs = load_database()

    if not embs:
        st.warning("Banco de dados vazio. Cadastre usuários primeiro.")
    else:
        option = st.radio("Escolha o método:", ["Upload de Imagem", "Webcam"])

        if option == "Upload de Imagem":
            uploaded_file = st.file_uploader("Faça upload da imagem para reconhecimento", type=['jpg','jpeg','png'])
            if uploaded_file:
                img = Image.open(uploaded_file).convert('RGB')
                results, img = recognize_faces(img, embs, labs)
                if results:
                    img_annotated = draw_boxes(img, results)
                    st.image(img_annotated, caption="Imagem com rostos reconhecidos", use_column_width=True)
                else:
                    st.warning("Nenhum rosto detectado ou reconhecido na imagem.")

        else:  # Webcam
            # Inicializa o estado da webcam
            if 'webcam_running' not in st.session_state:
                st.session_state['webcam_running'] = False

            if st.session_state['webcam_running']:
                if st.button("Parar Webcam"):
                    st.session_state['webcam_running'] = False
            else:
                if st.button("Iniciar Webcam"):
                    st.session_state['webcam_running'] = True

            stframe = st.empty()

            if st.session_state['webcam_running']:
                cap = cv2.VideoCapture(0)
                while st.session_state['webcam_running']:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Não foi possível acessar a webcam.")
                        break

                    frame = recognize_faces_cv2(frame, embs, labs)
                    stframe.image(frame, channels="BGR")

                    # Pequena pausa para o Streamlit atualizar interface e responder a cliques
                    if not st.session_state['webcam_running']:
                        break

                cap.release()
                stframe.empty()