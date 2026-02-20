# E-Waste Policy Assistant: 10-Day Roadmap & File Structure

## File Structure
```
/Users/sriram/Downloads/Policy BOT/
├── requirements.txt         # Project dependencies
├── app.py                   # Streamlit frontend application
├── brain.py                 # Core RAG pipeline logic (Document ingestion, ChromaDB, LLMs)
├── metal_calculator.py      # Business logic for scrap metal recovery calculation
├── evaluate.py              # local RAGAS evaluation script
├── knowledge_base/          # Directory containing all source PDFs
│   ├── MD_HB992_2026.pdf
│   ├── MD_SB655_2026.pdf
│   ├── MD_SB553.pdf
│   ├── EPA_Feb_2026_Universal_Waste.pdf
│   ├── Baltimore_DPW_Ewaste_Protocols.pdf
│   └── NFPA_855_2026.pdf
└── .streamlit/
    └── config.toml          # Dark mode styling and theming for Streamlit
```

## 10-Day Roadmap

**Day 1-2: Environment Setup & Knowledge Lake Ingestion**
*   Initialize Python environment and install `requirements.txt`.
*   Pull required local Ollama models (`qwen2.5:7b`, `deepseek-r1`, `nomic-embed-text`).
*   Retrieve all required PDF documents (State, Federal, Local, Technical) and place them in `/knowledge_base`.
*   Implement basic ingestion logic in `brain.py` to chunk and embed documents into a local ChromaDB instance.
*   Verify successful vector storage using exploratory queries.

**Day 3-4: RAG Core Development (`brain.py`)**
*   Implement LangChain retrieval chains.
*   Configure the retrieval mechanism to heavily prioritize exact Section/Article citations.
*   Develop routing logic to send standard queries to `Qwen 2.5 7B` and complex legal interpretation queries to `DeepSeek-R1`.
*   Test and refine prompts to address the "2026 Compliance Gap" specifically.

**Day 5: Business Logic (`metal_calculator.py`)**
*   Implement `metal_calculator.py` with 2026 projected or current scrap market rates for Gold, Palladium, and Copper.
*   Build functions to estimate recovery value based on expected e-waste tonnage inputs.
*   Write unit tests to verify calculation accuracy.

**Day 6-7: Dashboard UI Development (`app.py`)**
*   Scaffold the Streamlit application with a dark-mode, professional GovTech aesthetic (via `.streamlit/config.toml`).
*   Integrate the `metal_calculator.py` into a dedicated UI tab.
*   Integrate the RAG pipeline from `brain.py` into a chat interface tab.
*   Implement dynamic document upload functionality, allowing users to add PDFs directly through the UI, triggering asynchronous insertion into ChromaDB.

**Day 8-9: Evaluation & Refinement (`evaluate.py`)**
*   Develop a test suite using RAGAS in `evaluate.py`.
*   Create a dataset of expected municipal questions and ground truth answers based on the legislation.
*   Run the evaluation script to score "Faithfulness" (are answers supported by the text?) and "Answer Relevancy" (does it answer the specific question?).
*   Tune chunk sizes, overlap, and prompts based on evaluation results.

**Day 10: Final Review & Polish**
*   End-to-end testing of the entire application.
*   Formatting pass on Streamlit UI to ensure a seamless user experience.
*   Document final deployment steps for municipal IT staff (assuming local deployment).
