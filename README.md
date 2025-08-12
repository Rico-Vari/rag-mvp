## Streamlit RAG (Chroma) Sample

Minimal Streamlit app to chat over your PDFs using a Chroma vector DB and LangChain.

Inspired by the LangGraph + FastAPI starter in the excellent template: [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit).

### Setup

1. Install dependencies (using uv):

   ```bash
   uv pip install -r requirements.txt
   ```

2. Configure environment:

   ```bash
   cp .env.example .env
   # edit .env to set OPENAI_API_KEY and optional settings
   ```

3. Add your PDFs under:

   ```
   rag/knowledge_base/
   ```

4. Build the Chroma index:

   ```bash
   python -m rag.build_index
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

### Notes

- Requires an OpenAI API key in `.env` or environment (`OPENAI_API_KEY`).
- You can tweak chunking via `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`.
- The Chroma DB is stored in `rag/chroma_db/`.

