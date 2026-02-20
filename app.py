import os
import streamlit as st
from metal_calculator import calculate_recovery_value

# Inject Groq API key from Streamlit secrets into the environment
# so brain.py can access it without importing streamlit directly
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from brain import init_rag_pipeline, respond, ingest_single_pdf

UPLOADS_DIR = "./knowledge_base/uploads"

st.set_page_config(
    page_title="E-Waste Policy Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply basic dark mode aesthetic styling adjustments
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1e1e1e;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background: #2b2b2b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("♻️ E-Waste Policy Assistant (Maryland 2026)")
st.markdown("Actionable operating procedures from Maryland HB 992/SB 655 & SB 553.")

# Initialize the RAG pipeline once per session.
# First run: builds FAISS index from all PDFs.
# Subsequent runs: loads from disk instantly (fast path).
if "rag_ready" not in st.session_state:
    with st.spinner("Loading knowledge base... (first run will take a minute)"):
        try:
            init_rag_pipeline()
            st.session_state["rag_ready"] = True
        except Exception as e:
            st.error(f"Failed to load knowledge base: {e}")
            st.stop()

tab1, tab2, tab3 = st.tabs(["Policy Chat", "Metal Calculator", "Document Upload"])

with tab1:
    # DeepSeek toggle lives above the chat history
    use_deepseek = st.toggle("Use DeepSeek-R1 for complex legal reasoning", value=False)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello. I'm your Maryland E-Waste Policy Assistant. Ask me anything about HB 992, SB 655, SB 553, EPA Universal Waste rules, Baltimore DPW protocols, or NFPA 855 standards. I will always cite the specific section."
            }
        ]

    # Render full chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input at the bottom — works like a real chatbot
    if prompt := st.chat_input("Ask about e-waste compliance (e.g. What does HB 992 require for producers?)"):
        # Show user message immediately
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = respond(prompt, use_deepseek=use_deepseek)
                except Exception as e:
                    if "AuthenticationError" in type(e).__name__ or "auth" in str(e).lower() or "api_key" in str(e).lower():
                        response = "⚠️ Groq API key error. Please check that **GROQ_API_KEY** is correctly set in your Streamlit secrets panel (Manage app → Secrets)."
                    else:
                        response = f"⚠️ An error occurred: {e}"
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})
                
with tab2:
    st.subheader("Scrap Metal Recovery Calculator — Feb 2026 Rates")
    st.caption("Based on mixed municipal e-waste (phones, laptops, TVs, appliances). Rates as of Feb 19, 2026.")

    tonnage = st.number_input("Estimated E-Waste Tonnage:", min_value=0.0, value=1.0, step=0.5)

    if st.button("Calculate Recovery Value"):
        result = calculate_recovery_value(tonnage)

        st.success(f"**Total Estimated Recovery Value: ${result['total']:,.2f}**")
        st.markdown("---")
        st.markdown("#### Breakdown by Metal")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Gold",
                value=f"${result['gold']['value']:,.2f}",
                delta=result['gold']['yield_units']
            )
            st.caption("@ $5,006 / troy oz")
        with col2:
            st.metric(
                label="Palladium",
                value=f"${result['palladium']['value']:,.2f}",
                delta=result['palladium']['yield_units']
            )
            st.caption("@ $1,752 / troy oz")
        with col3:
            st.metric(
                label="Copper",
                value=f"${result['copper']['value']:,.2f}",
                delta=result['copper']['yield_units']
            )
            st.caption("@ $4.50 / lb scrap")

with tab3:
    st.subheader("Document Upload")
    st.markdown("Upload a new PDF (e.g. a local ordinance or amended regulation) and it will be immediately searchable in the Policy Chat.")
    uploaded_file = st.file_uploader("Upload a PDF into the Knowledge Base", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            save_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Ingesting '{uploaded_file.name}' into the knowledge base..."):
                try:
                    chunk_count = ingest_single_pdf(save_path)
                    st.success(f"Done. Added {chunk_count} chunks from '{uploaded_file.name}' — now searchable in Policy Chat.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
