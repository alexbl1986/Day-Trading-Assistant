"""Task 5 evaluation harness — RAGAS over the frozen 4-question dataset.

Layered like 10_LLM_Servers/rag_evaluation.py: each stage freezes its artifact
and is reused on later runs; delete a stage's file to re-run it.

  Layer 1  evals/eval_dataset.jsonl            (committed; built by build_eval_dataset.py)
  Layer 2  artifacts/runs_<variant>.jsonl      (retrieve top-5 + generate per question)
  Layer 3  artifacts/scores_<variant>.jsonl    (five RAGAS metrics)

Retrieval is byte-identical to production (the dev.py wiring): in-memory
Qdrant + ``text-embedding-3-large`` over ``data/reviews``, ``HybridRetriever``
with k=5 / first_stage_k=8. Generation is a minimal RAG prompt over
production-formatted ``[Source N: ...]`` blocks, answered by the production
gateway model. Each variant changes exactly one variable, so any score delta
attributes to that variable alone:

  baseline       DESK_RETRIEVAL=baseline retrieval, base prompt
  rerank         DESK_RETRIEVAL=rerank retrieval,   base prompt
  rerank_prompt  rerank retrieval, base prompt + production COMPLETENESS_RULE

Run with ``--variant``; the final table compares every variant that has
scores.

Judge: OpenAI ``gpt-5.4`` (the full model, not mini) called direct (not via
the gateway), wrapped exactly as in 10_LLM_Servers — override with
``RAGAS_JUDGE_MODEL``.

Run from the repo root:  .venv/bin/python evals/rag_evaluation.py [--variant baseline]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from app.rag.chunk import chunk_document  # noqa: E402
from app.rag.index import OPENAI_3_LARGE_DIM, CorpusIndex, openai_embedder  # noqa: E402
from app.graphs.trading_assistant.answer import COMPLETENESS_RULE  # noqa: E402
from app.rag.retrieve import (  # noqa: E402
    HybridRetriever,
    RetrievedDoc,
    apply_retrieval_mode,
)

load_dotenv(ROOT / ".env")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # must be set before any ragas import

# --- paths / config ----------------------------------------------------------
REVIEWS = ROOT / "data" / "reviews"
EVAL_DATASET = ROOT / "evals" / "eval_dataset.jsonl"  # frozen; committed
ARTIFACTS = ROOT / "artifacts"  # gitignored eval outputs

EVAL_USER = "eval-user"
RETRIEVE_K = 5  # production default (DeskReviewRetriever.k)

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
GENERATOR_MODEL = "openai/gpt-5.4-mini"  # the production gateway slug (dev.py)
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-5.4")

GEN_PROJECT = os.environ.get("LANGSMITH_EVAL_GEN_PROJECT", "dta-eval-generation")
EVAL_PROJECT = os.environ.get("LANGSMITH_EVAL_JUDGE_PROJECT", "dta-eval-ragas")

GENERATION_SYSTEM_PROMPT = (
    "You are a trading-desk assistant. Answer the trader's question using ONLY "
    "the desk-review excerpts provided (they may mix Hebrew and English). "
    "Answer in English, concisely and completely. If the excerpts do not "
    "contain the answer, say so explicitly."
)


# --- Layer 1: the frozen dataset ---------------------------------------------
def load_dataset() -> list[dict]:
    print("\n=== Layer 1: evaluation dataset ===")
    if not EVAL_DATASET.exists():
        raise SystemExit(f"{EVAL_DATASET} missing — run evals/build_eval_dataset.py")
    rows = [json.loads(line) for line in EVAL_DATASET.read_text(encoding="utf-8").splitlines()]
    for i, row in enumerate(rows):
        print(f"  [{i}] {row['question_type']:<20} {len(row['reference_contexts'])} ref ctx | {row['user_input']}")
    return rows


# --- Layer 2: run one retrieval variant + fixed generation -------------------
def _build_hybrid() -> HybridRetriever:
    """The production first stage, exactly as dev.py wires it: in-memory
    Qdrant, OpenAI text-embedding-3-large, the committed review PDFs."""
    pdfs = sorted(REVIEWS.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found under {REVIEWS}")
    index = CorpusIndex(
        QdrantClient(location=":memory:"),
        openai_embedder(),
        vector_size=OPENAI_3_LARGE_DIM,
    )
    for pdf in pdfs:
        index.replace_document(chunk_document(str(pdf)), user_id=EVAL_USER)
    return HybridRetriever(index)


# Each variant changes exactly one variable against its predecessor, so any
# score delta attributes to that variable alone. Retrieval modes are
# production's DESK_RETRIEVAL values via the shared apply_retrieval_mode
# factory; rerank_prompt appends the production COMPLETENESS_RULE (answer.py)
# to the generation prompt — imported, not copied, so eval and production
# can't drift.
VARIANTS: dict[str, tuple[str, str]] = {
    # name -> (retrieval mode, generation system prompt)
    "baseline": ("baseline", GENERATION_SYSTEM_PROMPT),
    "rerank": ("rerank", GENERATION_SYSTEM_PROMPT),
    "rerank_prompt": ("rerank", f"{GENERATION_SYSTEM_PROMPT} {COMPLETENESS_RULE}"),
}


def _generator() -> ChatOpenAI:
    return ChatOpenAI(
        model=GENERATOR_MODEL,
        base_url=GATEWAY_BASE_URL,
        api_key=os.environ["AI_GATEWAY_API_KEY"],
        temperature=0,
    )


def _source_block(index: int, doc: RetrievedDoc) -> str:
    """One retrieved chunk, formatted as the production desk tool presents it
    (mirrors tools._source_block, which takes LangChain Documents)."""
    score_text = f"{doc.score:.3f}" if isinstance(doc.score, (int, float)) else "n/a"
    pages = ",".join(str(p) for p in doc.pages)
    return (
        f"[Source {index}: {doc.source}, doc_type={doc.doc_type}, "
        f"review_date={doc.review_date or '—'}, "
        f"section={doc.section or '—'}, "
        f"chunk_id={doc.id}, pages={pages}, score={score_text}]\n"
        f"{doc.text}"
    )


def run_pipeline(variant: str, dataset: list[dict]) -> pd.DataFrame:
    print(f"\n=== Layer 2: pipeline runs [{variant}] ===")
    runs_path = ARTIFACTS / f"runs_{variant}.jsonl"
    if runs_path.exists():
        df = pd.read_json(runs_path, orient="records", lines=True)
        print(f"  reusing {runs_path}  ({len(df)} rows) — delete to re-run")
        return df

    mode, system_prompt = VARIANTS[variant]
    retriever = apply_retrieval_mode(_build_hybrid(), mode)
    generator = _generator()

    rows = []
    with tracing_context(project_name=GEN_PROJECT, tags=[variant, "layer2-generation"]):
        for i, example in enumerate(dataset):
            question = example["user_input"]
            docs = retriever.retrieve(question, user_id=EVAL_USER, k=RETRIEVE_K)
            sources = "\n\n".join(_source_block(n, d) for n, d in enumerate(docs, start=1))
            response = generator.invoke(
                [
                    ("system", system_prompt),
                    ("human", f"Question: {question}\n\nDesk-review excerpts:\n\n{sources}"),
                ],
                config={"run_name": f"gen-{variant}-q{i}", "metadata": {"variant": variant, "q_index": i}},
            ).text
            rows.append(
                {
                    "user_input": question,
                    "reference": example["reference"],
                    "reference_contexts": example["reference_contexts"],
                    "retrieved_contexts": [d.text for d in docs],
                    "retrieved_chunk_ids": [d.id for d in docs],
                    "response": response,
                }
            )
            mix = ", ".join(d.id for d in docs)
            print(f"\n  Q{i}: {question[:70]}")
            print(f"      retrieved: {mix}")
            print(f"      response: {response[:140].replace(chr(10), ' ')} ...")

    df = pd.DataFrame(rows)
    ARTIFACTS.mkdir(exist_ok=True)
    df.to_json(runs_path, orient="records", lines=True)
    print(f"\n  saved -> {runs_path}")
    return df


# --- Layer 3: RAGAS scoring ---------------------------------------------------
_INPUT_COLS = {"user_input", "reference", "reference_contexts", "retrieved_contexts", "response"}


def _apply_vertex_shim() -> None:
    """Ragas 0.4.x eagerly imports langchain_community.chat_models.vertexai,
    which newer langchain-community builds drop. Vertex is never used here —
    this just lets ``import ragas`` succeed (same shim as 10_LLM_Servers)."""
    import types

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        try:
            import langchain_community.chat_models.vertexai  # noqa: F401
        except Exception:
            vx = types.ModuleType("langchain_community.chat_models.vertexai")
            vx.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules["langchain_community.chat_models.vertexai"] = vx
    import langchain_community.llms as _llms

    if not hasattr(_llms, "VertexAI"):
        _llms.VertexAI = type("VertexAI", (), {})


def _judge():
    """Judge LLM (``JUDGE_MODEL``) + OpenAI embeddings, wrapped for RAGAS —
    the 10_LLM_Servers shape verbatim. The embeddings are defensive
    boilerplate (none of the five metrics here calls them); kept for parity."""
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    )
    return llm, emb


def score_pipeline(variant: str, runs_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== Layer 3: RAGAS scores [{variant}] ===")
    scores_path = ARTIFACTS / f"scores_{variant}.jsonl"
    if scores_path.exists():
        scores_df = pd.read_json(scores_path, orient="records", lines=True)
        print(f"  reusing {scores_path}  ({len(scores_df)} rows) — delete to re-score")
    else:
        _apply_vertex_shim()
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            AnswerAccuracy,
            ContextEntityRecall,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )

        samples = [
            {col: row[col] for col in _INPUT_COLS} for _, row in runs_df.iterrows()
        ]
        metrics = [
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            ContextEntityRecall(),
            Faithfulness(),
            AnswerAccuracy(),
        ]
        judge_llm, judge_emb = _judge()
        print(f"  scoring {len(samples)} examples with judge={JUDGE_MODEL} ...")
        with tracing_context(project_name=EVAL_PROJECT, tags=["ragas-eval", variant]):
            result = evaluate(
                dataset=EvaluationDataset.from_list(samples),
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_emb,
            )
        scores_df = result.to_pandas()
        ARTIFACTS.mkdir(exist_ok=True)
        scores_df.to_json(scores_path, orient="records", lines=True)
        print(f"  saved -> {scores_path}")

    metric_cols = [c for c in scores_df.columns if c not in _INPUT_COLS]
    for i, row in scores_df.iterrows():
        scores = "  ".join(
            f"{c}={row[c]:.3f}" if pd.notna(row[c]) else f"{c}=NaN" for c in metric_cols
        )
        print(f"  [{i}] {scores}")
        print(f"      Q: {str(row['user_input'])[:70]}")
    return scores_df


def compare_all_variants() -> None:
    """Mean-per-metric table across every variant that has frozen scores —
    one column per scored variant."""
    score_files = sorted(ARTIFACTS.glob("scores_*.jsonl"))
    if not score_files:
        return
    columns = {}
    for path in score_files:
        variant = path.stem.removeprefix("scores_")
        df = pd.read_json(path, orient="records", lines=True)
        metric_cols = [c for c in df.columns if c not in _INPUT_COLS]
        columns[variant] = df[metric_cols].mean(numeric_only=True)
    print("\n=== Comparison (mean per metric) ===")
    print(pd.DataFrame(columns).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline", choices=sorted(VARIANTS))
    args = parser.parse_args()

    dataset = load_dataset()
    runs_df = run_pipeline(args.variant, dataset)
    score_pipeline(args.variant, runs_df)
    compare_all_variants()


if __name__ == "__main__":
    main()
