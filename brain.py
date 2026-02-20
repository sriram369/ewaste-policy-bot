import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def _ensure_groq_key() -> None:
    """
    Ensures GROQ_API_KEY is in os.environ.
    - When running via Streamlit: app.py already sets it from st.secrets.
    - When running directly (testing, evaluate.py): loads from .streamlit/secrets.toml.
    """
    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        import tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomllib.load(f)
        key = secrets.get("GROQ_API_KEY", "")
        if key:
            os.environ["GROQ_API_KEY"] = key
    except Exception:
        pass

_ensure_groq_key()

KNOWLEDGE_BASE_DIR = "./knowledge_base"
VECTOR_DB_DIR = "./faiss_db"

# Embedding model — runs on CPU locally and on Streamlit Cloud, no API key needed
EMBED_MODEL = "all-MiniLM-L6-v2"

# Groq model names — production models (Feb 2026)
GROQ_MODEL_STANDARD = "llama-3.3-70b-versatile"  # standard policy queries
GROQ_MODEL_COMPLEX  = "openai/gpt-oss-120b"      # complex legal reasoning (largest production model)

# Keywords that strongly suggest a policy/compliance question
_POLICY_KEYWORDS = {
    "hb", "sb", "bill", "section", "article", "law", "regulation", "require",
    "mandate", "comply", "compliance", "epa", "nfpa", "dpw", "baltimore",
    "battery", "lithium", "ewaste", "e-waste", "waste", "recycl", "producer",
    "manufacturer", "register", "registration", "fee", "penalty", "deadline",
    "october", "protocol", "universal", "hazard", "storage", "collection",
    "takeback", "take-back", "dispose", "disposal", "handler", "facility",
}

# Prompt for casual / conversational messages
CONVERSATIONAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a friendly, professional Maryland E-Waste Policy Assistant. \
You help municipal workers, compliance officers, and producers understand Maryland \
e-waste law (HB 992, SB 655, SB 553), EPA Universal Waste rules, \
Baltimore DPW protocols, and NFPA 855 safety standards.

Respond naturally to the following message. If it is a greeting or small talk, \
reply warmly and briefly explain what you can help with. \
If it is a general question about e-waste topics (not requiring document lookup), \
answer from your knowledge but remind the user you can also cite specific legislation. \
Keep your tone helpful and concise.

Message: {question}

Response:"""
)

# Prompt that forces the model to cite specific sections and never hallucinate
CITATION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise legal policy assistant specializing in Maryland e-waste legislation, \
EPA federal regulations, Baltimore DPW operational protocols, and NFPA safety standards.

Use ONLY the following retrieved document sections to answer the question.
Do NOT use any knowledge outside of the provided context.

Rules:
- Answer the question DIRECTLY and specifically — do not give general background unless asked.
- Always cite the specific Bill number, Section, or Article (e.g. "HB 992, Section 4-101" or "SB 553, Article 2").
- If the answer spans multiple documents, cite each one.
- Keep your answer focused on exactly what was asked. Do not add unrequested context.
- If the answer cannot be found in the provided context, respond with exactly: \
"This information is not available in the current knowledge base."
- Be direct and actionable — this assistant is used by municipal compliance officers.

Context:
{context}

Question: {question}

Direct answer (with citations):"""
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def _get_llm(complex_mode: bool = False):
    # Reads GROQ_API_KEY from environment automatically
    model = GROQ_MODEL_COMPLEX if complex_mode else GROQ_MODEL_STANDARD
    return ChatGroq(model=model, temperature=0)


def _format_docs(docs) -> str:
    """Concatenates retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def init_rag_pipeline() -> FAISS:
    """
    Initializes the RAG pipeline.

    Fast path: if faiss_db already exists on disk, loads and returns it instantly.
    First-time path: reads all PDFs recursively from knowledge_base/, chunks them,
    embeds with all-MiniLM-L6-v2, and persists to disk as a FAISS index.

    Returns the FAISS vectorstore object.
    """
    embeddings = _get_embeddings()

    # Fast path — vector store already built, load from disk
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        vectorstore = FAISS.load_local(
            VECTOR_DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    # First-time build — ingest all PDFs from subfolders
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)

    loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE_DIR, recursive=True)
    docs = loader.load()

    if not docs:
        raise ValueError(
            f"No PDFs found in '{KNOWLEDGE_BASE_DIR}'. "
            "Add PDF documents to the knowledge_base/ subfolders before initializing."
        )

    # Tag each chunk with its source category (subfolder name)
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        relative = source_path.replace(os.path.abspath(KNOWLEDGE_BASE_DIR), "").strip("/")
        parts = relative.split("/")
        doc.metadata["category"] = parts[0] if len(parts) > 1 else "Uncategorized"

    text_splitter = _get_text_splitter()
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)

    return vectorstore


def query_rag(query_text: str, use_deepseek: bool = False) -> str:
    """
    Queries the RAG pipeline with a user question.

    - Loads FAISS index from disk.
    - Retrieves top 8 most relevant chunks.
    - Routes to llama-3.3-70b (standard) or deepseek-r1 (complex reasoning).
    - Returns a cited, grounded answer.
    """
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        return (
            "Knowledge base has not been initialized yet. "
            "Please wait for the initial document ingestion to complete."
        )

    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    llm = _get_llm(complex_mode=use_deepseek)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | CITATION_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query_text)


def _is_policy_question(text: str) -> bool:
    """Returns True if the message looks like a compliance/policy question."""
    text_lower = text.lower()
    words = text_lower.split()

    if len(words) <= 3:
        return False

    for keyword in _POLICY_KEYWORDS:
        if keyword in text_lower:
            return True

    return False


def respond(query_text: str, use_deepseek: bool = False) -> str:
    """
    Main entry point called by app.py for every user message.
    Routes casual messages to conversational LLM, policy questions through RAG.
    """
    if _is_policy_question(query_text):
        return query_rag(query_text, use_deepseek=use_deepseek)

    llm = _get_llm(complex_mode=False)
    chain = CONVERSATIONAL_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": query_text})


def ingest_single_pdf(file_path: str) -> int:
    """
    Adds a single PDF into the existing FAISS vector store.
    Used by app.py Tab 3 for live document uploads.
    Returns number of new chunks added.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")

    embeddings = _get_embeddings()

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = _get_text_splitter()
    splits = text_splitter.split_documents(docs)

    new_index = FAISS.from_documents(documents=splits, embedding=embeddings)
    existing_index = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    existing_index.merge_from(new_index)
    existing_index.save_local(VECTOR_DB_DIR)

    return len(splits)
