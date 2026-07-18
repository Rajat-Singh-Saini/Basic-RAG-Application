# Academics Support Chatbot

## Project Description

A RAG-based (Retrieval-Augmented Generation) chatbot that answers student questions grounded strictly in course documents. The system ingests PDF course materials, stores them in a FAISS vector database, and serves a Streamlit chat interface powered by OpenAI's `gpt-4o-mini`.

**How it works:**

1. **Ingest** — `Ingest.py` loads all PDFs from the `documents/` folder, splits them into overlapping chunks, embeds them using `text-embedding-3-small`, and saves the index to `Course_faiss_index/`.
2. **Chat** — `app.py` loads the saved index, retrieves the top-4 relevant chunks for each student question, and passes them as context to the LLM to generate a grounded answer.

The chatbot only answers from the provided course content. If the answer is not in the documents, it explicitly says so.

---

## Setup

### Prerequisites

- Python 3.9+
- An OpenAI API key (or a compatible API endpoint)

### 1. Clone / download the project

```bash
cd "Week 15_Graded Mini Project_Rajat Singh Saini"
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r Requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

> If you are using a custom OpenAI-compatible endpoint, replace `OPENAI_BASE_URL` accordingly.

### 5. Add course documents

Place the PDF course materials inside the `documents/` folder. Any `.pdf` file dropped there will be picked up during ingestion.

---

## Run Instructions

### Step 1 — Build the knowledge base (run once, or after adding new documents)

```bash
python Ingest.py
```

Expected output:
```
Loaded X documents.
Split documents into Y chunks.
Course knowledge base created and saved to disk.
```

This creates / overwrites the `Course_faiss_index/` directory.

### Step 2 — Launch the chatbot

```bash
streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`. Type a course-related question in the chat input and the bot will respond using only the ingested documents.

---

## Data Sources

The following documents are used as the knowledge base for this chatbot. Place the corresponding PDFs in the `documents/` folder before running ingestion.

| # | Document | Source Link |
|---|----------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |

> Add rows as needed. Links can point to course portals, public URLs, or internal drives where the source material lives.

---

## Project Structure

```
.
├── documents/              # Place course PDF files here
├── Course_faiss_index/     # Auto-generated FAISS vector index
├── Ingest.py               # One-time ingestion script
├── app.py                  # Streamlit chatbot application
├── Requirements.txt        # Python dependencies
└── .env                    # API keys (not committed to version control)
```
