## Streamlit RAG (Chroma) – Saudi Arabia Legal Assistant

Minimal Streamlit app to chat over your PDFs using a Chroma vector DB and LangChain. The agent is a cautious legal assistant specialized in KSA (Saudi Arabia) laws, providing general information only and citing sources where possible.

Inspired by the LangGraph + FastAPI starter: [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit).

### Setup

1) Create a virtual environment and install dependencies (uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

2) Configure environment:

```bash
cp .env.example .env
# edit .env to set OPENAI_API_KEY and optional settings
```

3) Add your PDFs under:

```
rag/knowledge_base/
```

4) Build the Chroma index:

```bash
python -m rag.build_index
```

5) Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### Agent behavior (prompt)
- Provides general KSA legal information; not legal advice.
- Answers are strictly based on retrieved context; if unclear, it says it doesn’t know and suggests consulting a qualified Saudi lawyer or authority.
- Cites sources (e.g., filenames) when possible and avoids definitive statements unless explicitly supported by the context.

### Notes
- Requires an OpenAI API key in `.env` or environment (`OPENAI_API_KEY`).
- Adjust chunking with `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`.
- The Chroma DB persists in `rag/chroma_db/`.

### Troubleshooting
- If you see import warnings, ensure you’ve activated the venv and installed deps with uv.
- If no answers return, verify you’ve added PDFs and rebuilt the index.

