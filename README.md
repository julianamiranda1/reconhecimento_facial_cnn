# Sistema de Reconhecimento Facial com PyTorch, FaceNet e Streamlit

Este projeto é um sistema completo de reconhecimento facial baseado em redes neurais convolucionais, utilizando a biblioteca `facenet-pytorch` e uma interface interativa via `Streamlit`. Permite o **cadastro incremental de usuários** e o **reconhecimento facial em tempo real** ou via upload de imagens.

---

## 📁 Estrutura do Projeto

```
reconhecimento_facial/
├── app_streamlit_completo.py       # Código principal com interface Streamlit
├── extract_embed.py                # Código utilizado para extrair os embeddings
├── rostos_cortados/                # Pasta com o rosto dos participantes recortado
│   └── henrique              
│   └── juliana 
│   └── rebecca 
├── rostos_participantes/           # Pasta com o rosto dos participantes
│   └── henrique              
│   └── juliana 
│   └── rebecca 
├── embeddings/
│   └── embeddings.pkl              # Arquivo de base de dados com embeddings e labels
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação do projeto
```

---

## 🚀 Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/reconhecimento-facial-cnn.git
cd reconhecimento-facial-cnn
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o aplicativo

```bash
python -m streamlit run app_streamlit_completo.py
```

---

## 🧠 Tecnologias Utilizadas

- **PyTorch** — Treinamento e inferência de modelos neurais
- **FaceNet (via facenet-pytorch)** — Geração de embeddings faciais
- **MTCNN** — Detecção facial
- **Streamlit** — Interface web para cadastro e reconhecimento
- **OpenCV** — Visualização da webcam e manipulação de vídeo

---

## ✨ Funcionalidades

### Cadastro de Usuário

- Envie múltiplas imagens do mesmo rosto
- Faces são detectadas com MTCNN e transformadas em vetores (embeddings)
- Nome do usuário é associado a cada vetor
- Dados são salvos incrementalmente no arquivo `embeddings.pkl`

### Reconhecimento Facial

- **Upload de Imagem**: detecta rostos e mostra nome/similaridade
- **Webcam**: reconhecimento em tempo real usando câmera local

### 🔁 Atualização

- O script `extract_embed.py` usado para pré-processamento inicial pode ser removido, pois a funcionalidade de cadastro incremental já está implementada no app principal (`app.py`).