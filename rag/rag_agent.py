from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


RAG_FOLDER = Path(__file__).parent
PERSIST_DIR = RAG_FOLDER / "chroma_db"


def get_llm() -> ChatOpenAI:
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    return ChatOpenAI(model=model, temperature=temperature)


def get_retriever():
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        embedding_function=embeddings, persist_directory=str(PERSIST_DIR)
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def build_rag_chain() -> Runnable:
    llm = get_llm()
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.",
            ),
            (
                "system",
                "Use the provided context to answer the user's question.\n"
                "If the answer is not in the context, say you don't know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(f"Source: {d.metadata.get('source','')}\n{d.page_content}" for d in docs)

    chain: Runnable = {
        "context": retriever | format_docs,
        "question": lambda x: x["question"],
    } | prompt | llm

    return chain


def generate_answer(question: str) -> str:
    chain = build_rag_chain()
    result = chain.invoke({"question": question})
    return result.content if hasattr(result, "content") else str(result)


def stream_answer(question: str) -> Iterable[str]:
    chain = build_rag_chain()
    for chunk in chain.stream({"question": question}):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def retrieve_sources(question: str) -> List[str]:
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(question)
    return [str(Path(d.metadata.get("source", "")).name) for d in docs]

