import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

KNOWLEDGE_BASE_DIR = "./knowledge_base"
VECTOR_DB_DIR = "./faiss_db"
EMBED_MODEL = "nomic-embed-text"

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


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def _format_docs(docs) -> str:
    """Concatenates retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def init_rag_pipeline() -> FAISS:
    """
    Initializes the RAG pipeline.

    Fast path: if faiss_db already exists on disk, loads and returns it instantly.
    First-time path: reads all PDFs recursively from knowledge_base/, chunks them,
    embeds with nomic-embed-text, and persists to disk as a FAISS index.

    Returns the FAISS vectorstore object.
    """
    embeddings = _get_embeddings()

    # Fast path — vector store already built, just load from disk
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

    # recursive=True walks all subfolders so it finds PDFs in nested category folders
    loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE_DIR, recursive=True)
    docs = loader.load()

    if not docs:
        raise ValueError(
            f"No PDFs found in '{KNOWLEDGE_BASE_DIR}'. "
            "Add PDF documents to the knowledge_base/ folder before initializing."
        )

    # Tag each chunk with its source category (derived from the subfolder name)
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

    - Connects to the existing FAISS index on disk (no re-ingestion).
    - Retrieves the 6 most relevant chunks from the legal documents.
    - Routes to qwen2.5:7b for standard queries.
    - Routes to deepseek-r1 for complex legal reasoning (when use_deepseek=True).
    - Returns an answer that always cites the specific law section.

    Args:
        query_text:   The user's compliance question.
        use_deepseek: If True, uses DeepSeek-R1 for chain-of-thought legal reasoning.

    Returns:
        Answer string with section/article citations.
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
        search_kwargs={"k": 8}  # increased from 6 — wider net catches edge-case sections
    )

    model_name = "deepseek-r1" if use_deepseek else "qwen2.5:7b"
    # temperature=0 for deterministic, grounded legal answers — no creative guessing
    llm = OllamaLLM(model=model_name, temperature=0)

    # Modern LCEL chain
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | CITATION_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query_text)


def _is_policy_question(text: str) -> bool:
    """
    Returns True if the message looks like a compliance/policy question.
    Uses keyword matching — fast, no extra LLM call needed.
    Short greetings and casual messages return False.
    """
    text_lower = text.lower()
    words = text_lower.split()

    # Very short messages are almost always casual
    if len(words) <= 3:
        return False

    # Check for any policy-related keyword in the message
    for keyword in _POLICY_KEYWORDS:
        if keyword in text_lower:
            return True

    return False


def respond(query_text: str, use_deepseek: bool = False) -> str:
    """
    Main entry point called by app.py for every user message.

    Routes the message to the right handler:
    - Casual / greetings / general questions → conversational LLM response
    - Policy / compliance / legislation questions → full RAG pipeline with citations

    Args:
        query_text:   The user's message.
        use_deepseek: If True, uses DeepSeek-R1 for policy questions.

    Returns:
        Response string — either conversational or a cited policy answer.
    """
    if _is_policy_question(query_text):
        return query_rag(query_text, use_deepseek=use_deepseek)

    # Casual conversation — respond naturally without touching the vector store
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0.7)
    chain = CONVERSATIONAL_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": query_text})


def ingest_single_pdf(file_path: str) -> int:
    """
    Adds a single PDF into the existing FAISS vector store.

    Used by app.py Tab 3 when a user uploads a new document through the UI.
    Merges new chunks into the existing index and saves back to disk.
    Safe to call while the app is running.

    Args:
        file_path: Absolute or relative path to the PDF file to ingest.

    Returns:
        Number of new chunks added to the vector store.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")

    embeddings = _get_embeddings()

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = _get_text_splitter()
    splits = text_splitter.split_documents(docs)

    # Build a small index for just the new doc, then merge into the existing one
    new_index = FAISS.from_documents(documents=splits, embedding=embeddings)

    existing_index = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    existing_index.merge_from(new_index)
    existing_index.save_local(VECTOR_DB_DIR)

    return len(splits)
