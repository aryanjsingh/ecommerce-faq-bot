# 🛒 E-Commerce FAQ Bot

An intelligent, fully **offline** e-commerce customer support chatbot built with **LangGraph**, **ChromaDB**, and **Ollama (Qwen2.5:3b)**. No API keys required — runs entirely on your local machine using Apple Silicon GPU acceleration.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph Agent                │
│                                             │
│  memory → router ──► retrieve → answer      │
│                  ├──► skip    → answer      │
│                  └──► tool   → answer      │
│                               └──► eval    │
│                                    └──► save│
└─────────────────────────────────────────────┘
    │
    ▼
Streamlit Chat UI
```

**8-node state machine:**
| Node | Role |
|------|------|
| `memory` | Maintains conversation history (sliding window) |
| `router` | Classifies intent: retrieve / skip / tool |
| `retrieve` | Vector search over ChromaDB knowledge base |
| `skip` | Handles greetings / off-topic with no KB lookup |
| `tool` | Live date/time queries |
| `answer` | Generates response grounded in retrieved context |
| `eval` | Faithfulness check on the answer |
| `save` | Persists turn to memory |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-faq-bot.git
cd ecommerce-faq-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the local model
```bash
ollama pull qwen2.5:3b
```

### 4. Run the Streamlit app
```bash
streamlit run capstone_streamlit.py
```

Open `http://localhost:8501` in your browser. The bot is ready!

---

## 📁 Project Structure

```
├── agent.py                  # Core LangGraph agent, KB, LLM setup
├── capstone_streamlit.py     # Streamlit chat UI
├── day13_capstone.ipynb      # Dev/test notebook with multi-turn tests
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🧠 Knowledge Base Topics (25 chunks)

- Returns, Refunds & Exchanges
- Standard, Expedited, Overnight & Same-Day Shipping
- International Shipping
- Order Tracking, Cancellation & Modification
- Missing / Damaged / Wrong Items
- Warranty & Extended Warranty
- Payment Methods & Security
- Promo Codes & Gift Cards
- Loyalty Rewards Program
- Account Management
- Customer Support Hours
- Bulk & Business Orders

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen2.5:3b via Ollama (local, offline) |
| Agent Framework | LangGraph |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| UI | Streamlit |
| Memory | LangGraph MemorySaver |
