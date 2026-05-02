# 💊 MedSimplify — AI-Powered Medication Instructions Simplifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-green)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/GPT--4o--mini-LangChain-orange?logo=openai)](https://openai.com)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Gradio-brightgreen)](https://e6ffb09b9a7e358077.gradio.live)

> **MedSimplify** transforms complex pharmaceutical drug information into clear, patient-friendly medication instructions using a production-grade agentic AI pipeline with hybrid RAG, multi-layer guardrails, and automated evaluation.

---

## 🎯 Problem

Prescription labels in the United States are written at a **9th-grade reading level** on average — far above the 6th-grade literacy of most adults. This gap leads to:
- **50%** of patients misunderstanding their medication instructions
- **$500B+** in annual preventable healthcare costs from non-adherence
- Disproportionate harm to elderly, immigrant, and low-literacy populations

**MedSimplify solves this** by using Generative AI + RAG to instantly translate any drug name into structured, plain-language patient instructions.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────┐
│ Input Guard │────▶│ Hybrid RAG   │────▶│   Generate    │────▶│ Output Guard │────▶│ Evaluate │
│  (Node 1)   │     │  (Node 2)    │     │  GPT-4o-mini  │     │  (Node 4)    │     │ (Node 5) │
│             │     │              │     │   (Node 3)    │     │              │     │          │
│ • Injection │     │ • FAISS Dense│     │ • Clinical    │     │ • Hallucin.  │     │ • ROUGE  │
│   detection │     │ • BM25 Sparse│     │   pharmacist  │     │   detection  │     │ • FK     │
│ • Sanitize  │     │ • Cross-Enc. │     │   prompt      │     │ • Disclaimer │     │   Grade  │
│             │     │   Reranking  │     │ • 6 sections  │     │   injection  │     │ • Complet│
└─────────────┘     └──────────────┘     └───────────────┘     └──────────────┘     └──────────┘
                           │
                    ┌──────┴──────┐
                    │  248K Drug  │
                    │  Knowledge  │
                    │    Base     │
                    └─────────────┘
```

### Key Technologies

| Component | Technology |
|-----------|-----------|
| Agentic Orchestration | LangGraph (StateGraph) |
| Dense Retrieval | FAISS + Sentence-BERT (all-MiniLM-L6-v2) |
| Sparse Retrieval | BM25 Okapi |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Language Model | GPT-4o-mini via LangChain |
| Safety | Custom GuardrailsEngine (4 layers) |
| Evaluation | ROUGE-1/2/L, Flesch-Kincaid, Faithfulness |
| UI | Gradio Blocks |
| TTS | gTTS |
| Translation | Google Translate API |
| PDF Export | ReportLab |

---

## ✨ Features

- 🤖 **5-Node LangGraph Pipeline** — input guard → hybrid RAG → generate → output guard → evaluate
- 🔍 **Hybrid RAG** — FAISS dense + BM25 sparse + cross-encoder reranking for maximum precision
- 🛡️ **4-Layer Guardrails** — prompt injection detection, faithfulness scoring, dosage override blocking, mandatory disclaimers
- 📊 **Automated Evaluation** — ROUGE scores, Flesch-Kincaid Grade, section completeness, faithfulness
- 🌐 **10 Languages** — EN, ES, HI, FR, PT, AR, ZH, DE, JA, TL
- 🔊 **Text-to-Speech** — voice responses in native language via gTTS
- 📄 **PDF Export** — downloadable medication summary reports
- 💬 **Multi-turn Memory** — conversation history preserved across turns
- ⚡ **Streaming Output** — word-by-word response streaming

---

## 📊 Evaluation Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Flesch-Kincaid Grade Level | 7.8 | ≤ 8.0 | ✅ Pass |
| Flesch Reading Ease | 58.6 | ≥ 55.0 | ✅ Pass |
| ROUGE-1 F1 | 0.41 | — | — |
| ROUGE-L F1 | 0.38 | — | — |
| Section Completeness | 96% | ≥ 80% | ✅ Pass |
| Faithfulness Score | 0.21 | > 0.05 | ✅ Pass |
| Guardrail Pass Rate | 100% | 100% | ✅ Pass |

*Evaluated on 20 representative medication queries across multiple drug classes.*

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/MedSimplify.git
cd MedSimplify
```

### 2. Install Dependencies
```bash
pip install pandas numpy faiss-cpu sentence-transformers rank-bm25
pip install langchain langchain-openai langgraph openai langchain-community
pip install gradio gtts deep-translator langdetect
pip install reportlab rouge-score textstat langchain-text-splitters nltk
```

### 3. Set Your OpenAI API Key
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 4. Upload the Dataset
Download the [Medicine Dataset from Kaggle](https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details) and upload it to your Google Colab session or place it in the project root.

### 5. Run the Notebook
Open `MedSimplify.ipynb` in Google Colab and run all cells. The Gradio interface will launch with a public shareable link.

---

## 📁 Project Structure

```
MedSimplify/
├── MedSimplify.ipynb              # Main notebook (run in Google Colab)
├── README.md                      # This file
├── requirements.txt               # All dependencies
├── data/
│   └── medicine_dataset.csv       # Kaggle medicine dataset (download separately)
└── outputs/
    └── medication_summary.pdf     # Sample exported PDF report
```

---

## 🗃️ Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle – 11000 Medicine Details](https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details) |
| Total Records | 248,218 |
| Columns | 58 |
| Side Effect Columns | 42 |
| Use Case Columns | 5 |
| Indexing Sample | 15,000 (configurable) |

---

## 🧪 Sample Output

**Query:** *"What is augmentin 625 duo tablet used for?"*

```
💊 MEDICATION OVERVIEW
Augmentin 625 is an antibiotic used to treat bacterial infections.
It helps your body fight off germs that can make you sick.

📋 HOW TO USE THIS MEDICATION
Take as prescribed by your doctor, usually every 8–12 hours with food.
Always complete the full course even if you feel better.

⚠️ POSSIBLE SIDE EFFECTS
• Nausea  • Vomiting  • Diarrhea  • Skin rash

🚫 IMPORTANT WARNINGS
Do not take if allergic to penicillin. Inform your doctor of all
current medications before starting this treatment.

🔄 ALTERNATIVE OPTIONS
Moxikind-CV 625, Clavam 625, Augpen 625mg Tablet

🧠 SIMPLE EXPLANATION
Augmentin 625 is a two-medicine antibiotic combination...

⚕️ Medical Disclaimer: For educational purposes only. Always consult
a licensed healthcare provider before making any medication decisions.
```

---

## 🛡️ Guardrails System

```
Layer 1 – Input Validation
  └── Length & format checks
  └── Regex prompt injection detection (6 patterns)
  └── High-risk keyword flagging

Layer 2 – Context Grounding
  └── RAG forces responses grounded in retrieved drug records
  └── Low-overlap hallucination flagging

Layer 3 – Output Safety
  └── Dosage override pattern detection
  └── Context faithfulness scoring (threshold > 0.05)

Layer 4 – Disclaimer Injection
  └── Mandatory medical disclaimer on EVERY response
```

---

## 📚 Research Paper

This project is accompanied by a research paper submitted to the **International Journal of Artificial Intelligence in Healthcare (IJAIH), Inderscience Publishers**:

> Jeelakarra, S.S.P. (2026). *MedSimplify: An Agentic AI System for Automated Generation of Patient-Friendly Medication Instructions Using Hybrid Retrieval-Augmented Generation and Multi-Layer Guardrails.* Int. J. Artificial Intelligence in Healthcare.

---

## 🔮 Future Work

- [ ] Expand knowledge base to full DrugBank + FDA drug labels
- [ ] Implement BERTScore and FactScore for rigorous faithfulness evaluation
- [ ] Human evaluation study measuring patient comprehension improvement
- [ ] Fine-tuned open-source clinical LLM (BioMistral, Llama-3-Med)
- [ ] Drug-drug interaction warnings in guardrails
- [ ] Clinical pilot study with patient volunteers

---

## 📖 Key References

- Lewis et al. (2020). Retrieval-Augmented Generation for NLP tasks. *NeurIPS 2020*.
- Karpukhin et al. (2020). Dense Passage Retrieval. *EMNLP 2020*.
- Nogueira & Cho (2019). Passage Re-ranking with BERT. *arXiv*.
- Wolf et al. (2010). Improving prescription drug labels. *JAMA Internal Medicine*.
- Robertson & Zaragoza (2009). BM25. *Foundations and Trends in IR*.

---

## ⚠️ Disclaimer

MedSimplify is an **educational and research prototype**. It is **not a medical device** and should not be used for clinical decision-making. All generated content is for informational purposes only. Always consult a licensed healthcare provider before making any medication decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Shanmukha Sai Prakash Jeelakarra**  
Department of Health Informatics, School of Healthcare Professions  
Rutgers University  
📧 sj1398@shp.rutgers.edu  
🔗 [LinkedIn](www.linkedin.com/in/shanmukhasaiprakashjeelakarra)

---

*Built as the final project for BINF5550 – Generative AI for Healthcare, Rutgers University, April 2026.*
