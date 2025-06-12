## 📊 Relatório Técnico (Resumo)

### Objetivo

Desenvolver um MVP funcional de um sistema de reconhecimento facial com interface interativa, possibilitando cadastro incremental de usuários e reconhecimento em tempo real ou via upload.

### Descrição do Pipeline

1. Upload ou captura de imagem (webcam)
2. Detecção facial com MTCNN
3. Extração de embedding com InceptionResnetV1 (FaceNet)
4. Cálculo da similaridade cosseno entre embedding atual e base de dados
5. Classificação com base em limiar de similaridade
6. Exibição do resultado com anotações visuais

### Arquitetura da Rede Utilizada

- **InceptionResnetV1** (FaceNet)
- Pré-treinada no dataset VGGFace2 para extração de características faciais
- Usada apenas como extrator de embeddings (modelo congelado)

### Estratégia de Pré-processamento e Treino

- As imagens são redimensionadas para 160x160
- Normalização com média e desvio padrão padrão [0.5]
- Não há treino supervisionado: usamos **aprendizado por similaridade** (comparando embeddings)
- Embeddings de novas imagens são salvos incrementalmente com o nome do usuário

### Resultados

- Sistema funcional com identificação correta de rostos conhecidos
- Limiar de similaridade (default = 0.6) pode ser ajustado para sensibilidade/precisão
- Acurácia qualitativa: bons resultados com imagens bem iluminadas e rostos frontais

> ⚠️ Como não há um conjunto de teste rotulado formal, métricas como **acurácia** e **matriz de confusão** não foram computadas quantitativamente. Elas podem ser incluídas em futuras iterações com dataset anotado.

### Dificuldades e Aprendizados

- Controle da webcam no Streamlit requer cuidados especiais para evitar múltiplas instâncias e travamentos
- Gerenciamento incremental dos dados com `pickle`
- Detecção falha em rostos parcialmente visíveis ou com iluminação ruim
- Importância da padronização das imagens para melhor performance

---

## 📝 Melhorias Futuras

- Reconhecimento de expressões faciais ou emoções
- Persistência em banco de dados real (SQLite ou MongoDB)
- Dashboard com histórico de acessos
- Otimizações de performance para webcam
- Sistema de avaliação quantitativa com base em dataset rotulado

---

## 👤 Autores

- Henrique Araujos Felix de Lima
- Juliana Soares de Miranda
- Rebecca Campos Machado
- Projeto desenvolvido como parte de estudo/prática com redes neurais e aplicações de visão computacional
