"""RAG quality evaluation with RAGAS.

Builds the same retrieval chain used by the app over the sources in
eval_dataset.json, generates answers + retrieved contexts for each question,
and scores them with RAGAS metrics:

  - faithfulness          : is the answer grounded in the retrieved context?
  - answer_relevancy      : does the answer address the question?
  - context_precision     : are the retrieved chunks relevant (ranked well)?
  - context_recall        : did retrieval surface the info needed for the truth?

Usage:
    pip install -r eval/requirements-eval.txt
    export OPENAI_API_KEY=sk-...        # (Windows: set OPENAI_API_KEY=...)
    python eval/eval.py

Note: RAGAS APIs evolve quickly; this targets ragas>=0.2,<0.3.
"""
from __future__ import annotations

import json
import os
import sys

# Make the parent package (rag.py) importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_chain_for_sources(sources):
    docs, errors = rag.load_documents(sources)
    for src, msg in errors:
        print(f"[warn] failed to load {src}: {msg}")
    if not docs:
        raise SystemExit("No documents loaded; check the source URLs.")
    chunks = rag.split_documents(docs)
    embeddings = rag.get_embeddings()
    retriever, _ = rag.build_retriever(chunks, embeddings, k=4)
    llm = rag.get_llm("OpenAI", "gpt-4o-mini", temperature=0.0, streaming=False)
    chain = rag.build_rag_chain(llm, retriever)
    return chain


def main():
    with open(os.path.join(HERE, "eval_dataset.json"), encoding="utf-8") as f:
        data = json.load(f)

    chain = build_chain_for_sources(data["sources"])

    print(f"Generating answers for {len(data['samples'])} questions...")
    rows = []
    for sample in data["samples"]:
        result = chain.invoke({"input": sample["question"], "chat_history": []})
        contexts = [d.page_content for d in result.get("context", [])]
        rows.append(
            {
                "user_input": sample["question"],
                "response": result["answer"],
                "retrieved_contexts": contexts,
                "reference": sample["ground_truth"],
            }
        )

    # --- RAGAS evaluation -------------------------------------------------
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    evaluator_llm = LangchainLLMWrapper(
        rag.get_llm("OpenAI", "gpt-4o-mini", temperature=0.0, streaming=False)
    )
    evaluator_emb = LangchainEmbeddingsWrapper(rag.get_embeddings())

    dataset = EvaluationDataset.from_list(rows)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    print("Scoring with RAGAS...")
    scores = evaluate(dataset=dataset, metrics=metrics)
    print("\n=== RAGAS results ===")
    print(scores)
    try:
        df = scores.to_pandas()
        out = os.path.join(HERE, "eval_results.csv")
        df.to_csv(out, index=False)
        print(f"\nPer-sample scores written to {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not export CSV: {exc})")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")
    main()
