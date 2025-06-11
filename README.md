# Sistema de Reconhecimento Facial - MVP Completo

## Descrição

Este projeto é um sistema básico de reconhecimento facial implementado em Python, utilizando modelos pré-treinados do Facenet (InceptionResnetV1) para extrair embeddings faciais e MTCNN para detecção de faces. A interface é feita com Streamlit, permitindo:

- Cadastro incremental de usuários com múltiplas imagens.
- Reconhecimento facial em imagens enviadas pelo usuário.
- Reconhecimento facial em tempo real via webcam.
- Salvamento persistente dos embeddings e labels usando pickle.

---

## Funcionalidades

- **Cadastro de novos usuários**: associe um nome a múltiplas imagens para gerar embeddings faciais.
- **Reconhecimento por imagem**: upload de imagens para identificar rostos conhecidos.
- **Reconhecimento por webcam**: captura em tempo real e identificação dos rostos detectados.
- **Persistência**: armazenamento e carregamento dos dados de embeddings e nomes via arquivo `.pkl`.

---

## Requisitos

- Python 3.11+
- Biblioteca Streamlit
- PyTorch (com suporte CUDA opcional)
- facenet-pytorch
- OpenCV (`cv2`)
- Pillow (`PIL`)
- numpy

---


