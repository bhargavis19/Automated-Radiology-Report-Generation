# Automated-Radiology-Report-Generation

## 📌 Project Overview
Radiologists interpret a large number of chest X-rays daily, and writing detailed reports is time-consuming and prone to inter-observer variability.  
This project focuses on **automatically generating clinically meaningful radiology reports from chest X-ray images** using deep learning–based **vision–language models**.

We implement and compare:
- A **baseline CNN–RNN architecture**
- An **attention-enhanced baseline**
- **Advanced transformer-based multimodal pipelines**
- **Retrieval-Augmented Generation (RAG)**
- **LoRA fine-tuning for domain adaptation**

The system aims to assist radiologists by improving reporting speed, consistency, and clinical reliability.

---

## 🧠 Key Objectives
- Generate **structured radiology reports** (Findings + Impression) from chest X-rays
- Improve **visual–text alignment** using attention mechanisms
- Reduce **hallucinations** using retrieval-augmented generation
- Enhance **clinical accuracy** using domain-specific vision–language models
- Compare baseline and advanced architectures using comprehensive evaluation metrics

---

## 🏗️ Architectures Implemented

### 1️⃣ Baseline Model: CNN + LSTM
- **Encoder**: CheXNet (DenseNet-121)
- **Decoder**: LSTM with GloVe embeddings
- **Decoding**: Greedy Search, Beam Search (Beam=2, 5)
- Generates reports word-by-word from image features

---

### 2️⃣ Baseline + Attention
- Retains **spatial feature maps** from CheXNet
- Uses **GRU decoder with additive attention**
- Allows the model to focus on clinically relevant regions of the X-ray
- Produces more coherent and clinically meaningful reports

---

### 3️⃣ Advanced Model 1: ViT + GPT-2 + RAG
- **Encoder**: Vision Transformer (ViT-Base)
- **Decoder**: DistilGPT-2
- **Retrieval**: Cosine similarity search over image embeddings
- Retrieved similar cases condition the report generation
- Reduces hallucinations and improves contextual grounding

---

### 4️⃣ Advanced Model 2: MedCLIP + T5 + RAG
- **Encoder**: BioMedCLIP (medical vision–language model)
- **Decoder**: Flan-T5
- **Retrieval**: FAISS-based similarity search
- Combines visual understanding with historical case reasoning

---

### 5️⃣ Advanced Model 3 (Proposed): MedCLIP + T5 + RAG + LoRA
- Adds **LoRA (Low-Rank Adaptation)** fine-tuning
- Parameter-efficient domain adaptation
- Improves factual correctness and report fluency
- Best-performing model across most evaluation metrics

---

## 📂 Dataset
**Indiana University Chest X-ray Dataset (IU-CXR / Open-I)**

- ~8,000 chest X-ray images
- Paired with expert-authored radiology reports
- Multiple views (PA, lateral)
- Reports stored in structured XML format

### Preprocessing
- Image resizing, normalization
- XML parsing for Findings and Impression
- Text cleaning and normalization
- Patient-level train/validation/test split to avoid data leakage

---

## ⚙️ Training Details
- **Frameworks**: PyTorch, TensorFlow, HuggingFace
- **Embeddings**: GloVe (baseline), Transformer embeddings (advanced)
- **Optimizers**: Adam 
- **Training Epochs**: 20
- **Hardware**: GPU (Kaggle / Colab compatible)

---

## 📊 Evaluation Metrics
The models are evaluated using both **linguistic** and **semantic** metrics:

| Metric | Purpose |
|------|--------|
| BLEU-1/2/3/4 | N-gram overlap |
| ROUGE-1/2/L | Recall-oriented similarity |
| METEOR | Semantic matching |
| BERTScore (P/R/F1) | Contextual semantic similarity |
| Perplexity | Language fluency |

---

## 🏆 Results Summary
- Attention improves baseline CNN–LSTM performance
- Transformer-based models outperform RNN-based models
- Retrieval-Augmented Generation significantly reduces hallucinations
- **MedCLIP + T5 + RAG + LoRA** achieves the best balance of:
  - Clinical correctness
  - Linguistic fluency
  - Semantic alignment

---

## 🚀 Applications
- Clinical decision support
- Radiologist workload reduction
- Reporting consistency improvement
- AI-assisted diagnosis in resource-constrained settings

---

## 🔮 Future Scope
- Extend to CT, MRI, and Ultrasound modalities
- Incorporate clinical metadata (age, history)
- Human-in-the-loop evaluation
- Real-world PACS/RIS integration
- Clinical validation studies

---
