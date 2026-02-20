"""
RAGAS Evaluation Script — E-Waste Policy Assistant (Maryland 2026)

Metrics:
  - Faithfulness:      Is the answer actually supported by the retrieved document chunks?
                       Score of 1.0 = every claim in the answer is backed by the context.
  - Answer Relevancy: Does the answer actually address the question asked?
                       Score of 1.0 = answer is fully on-topic.

Models used for evaluation:
  - LLM:        qwen2.5:7b (local via Ollama)
  - Embeddings: nomic-embed-text (local via Ollama)

Run with:
    python3 -W ignore evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ── Local model configuration ──────────────────────────────────────────────────
EVAL_LLM = LangchainLLMWrapper(OllamaLLM(model="qwen2.5:7b", temperature=0))
EVAL_EMBEDDINGS = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

VECTOR_DB_DIR = "./faiss_db"

# ── Ground truth test dataset ───────────────────────────────────────────────────
# 12 questions drawn from the actual legislation in the knowledge base.
# Each has a reference answer grounded in the specific bill/section.
TEST_DATASET = [
    {
        "question": "What is the initial registration fee for a manufacturer selling at least 1,000 covered electronic devices in Maryland under HB 992?",
        "reference": "The initial registration fee is $10,000 for manufacturers selling at least 1,000 covered electronic devices in the State in the prior year, as specified in HB 992, Section 4(c)(2).",
    },
    {
        "question": "What is the registration fee for manufacturers selling between 100 and 999 covered electronic devices under HB 992?",
        "reference": "The initial registration fee is $5,000 for manufacturers selling between 100 and 999 covered electronic devices in the State in the prior year, as specified in HB 992, Section 4(c)(2).",
    },
    {
        "question": "How many days does a manufacturer have to revise an insufficient registration under HB 992?",
        "reference": "A manufacturer must revise an insufficient registration within 60 days of receiving notice from the Department, as required by HB 992, Section 4(d)(2).",
    },
    {
        "question": "Where does the Department publish the list of registered manufacturers under HB 992?",
        "reference": "The Department is required to maintain and publish a list of registered manufacturers on its website, as stated in HB 992, Section 4(e).",
    },
    {
        "question": "What commission does SB 553 establish?",
        "reference": "SB 553 establishes a Lithium-Ion Battery Safety Commission to study and make recommendations on lithium-ion battery safety, collection, storage, and recycling.",
    },
    {
        "question": "What does SB 553 Section 4 require regarding first responder training?",
        "reference": "SB 553, Section 4 mandates the Commission to consider training, education, and other information to better inform the public and first responders regarding lithium-ion battery safety.",
    },
    {
        "question": "What is EPA proposing for lithium batteries under its Universal Waste rulemaking?",
        "reference": "EPA is proposing to add lithium batteries as a designated category of universal waste under RCRA's Universal Waste regulations, which would streamline their collection and recycling by reducing regulatory burden on handlers.",
    },
    {
        "question": "What types of energy storage systems does NFPA 855 cover?",
        "reference": "NFPA 855 covers the installation of stationary energy storage systems, including lithium-ion batteries, and establishes requirements for fire safety, hazard mitigation analysis, and safe siting of such systems.",
    },
    {
        "question": "What does HB 992 require producers to do to ensure covered entities can access takeback services?",
        "reference": "HB 992, Section 4 requires producers to conduct public awareness activities using a statewide promotion system so that covered entities can easily identify, understand, and access services provided by all electronic device producer responsibility programs.",
    },
    {
        "question": "What is the purpose of the Electronic Device Producer Responsibility Program established by HB 992?",
        "reference": "HB 992 establishes a separate covered electronic device producer responsibility program to ensure manufacturers fund and operate takeback programs for end-of-life electronics, reducing e-waste going to landfills in Maryland.",
    },
    {
        "question": "What does SB 655 require of electronic device producers in Maryland?",
        "reference": "SB 655 establishes a producer responsibility program requiring electronic device manufacturers to register, pay fees, and implement or join an approved takeback program for covered electronic devices sold in Maryland.",
    },
    {
        "question": "What fund is established under HB 992 for managing electronic device recycling fees?",
        "reference": "HB 992, Section 1 establishes a separate covered electronic device producer responsibility program plan and annual report, and registration and review fees account within the State Recycling Trust Fund.",
    },
]


def _get_answer_and_contexts(question: str) -> tuple[str, list[str]]:
    """
    Runs a question through the RAG pipeline and returns both the answer
    and the raw retrieved document chunks (needed by RAGAS for faithfulness scoring).
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Get retrieved docs
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    # Get answer from brain.py's query_rag (uses the same FAISS + prompt)
    from brain import query_rag
    answer = query_rag(question, use_deepseek=False)

    return answer, contexts


def run_evaluation():
    """
    Runs the full RAGAS evaluation over the test dataset.
    Prints per-question results and overall scores.
    """
    print("=" * 60)
    print("E-Waste Policy Assistant — RAGAS Evaluation")
    print("Metrics: Faithfulness, Answer Relevancy")
    print(f"Questions: {len(TEST_DATASET)}")
    print("LLM: qwen2.5:7b (local) | Embeddings: nomic-embed-text")
    print("=" * 60)
    print()

    samples = []
    for i, item in enumerate(TEST_DATASET, 1):
        print(f"[{i}/{len(TEST_DATASET)}] Running: {item['question'][:70]}...")
        answer, contexts = _get_answer_and_contexts(item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=answer,
                retrieved_contexts=contexts,
                reference=item["reference"],
            )
        )
        print(f"        Answer: {answer[:80]}...")
        print()

    print("Running RAGAS scoring (this may take a few minutes)...")
    print()

    dataset = EvaluationDataset(samples=samples)

    faithfulness_metric = Faithfulness(llm=EVAL_LLM)
    answer_relevancy_metric = AnswerRelevancy(llm=EVAL_LLM, embeddings=EVAL_EMBEDDINGS)

    # max_workers=1 prevents parallel timeout failures with local Ollama LLMs
    result = evaluate(
        dataset,
        metrics=[faithfulness_metric, answer_relevancy_metric],
        run_config=RunConfig(max_workers=1, timeout=300),
    )

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(result)
    print()

    # Save results to a file for reference
    with open("evaluation_results.txt", "w") as f:
        f.write("E-Waste Policy Assistant — RAGAS Evaluation Results\n")
        f.write(f"Questions evaluated: {len(TEST_DATASET)}\n\n")
        f.write(str(result))
    print("Results saved to evaluation_results.txt")
    print()
    print("Score guide:")
    print("  Faithfulness   — 1.0 = every claim in answers backed by retrieved docs")
    print("  AnswerRelevancy — 1.0 = answers are fully on-topic to the questions asked")


if __name__ == "__main__":
    run_evaluation()