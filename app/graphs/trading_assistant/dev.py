"""Local / Studio wiring: the real gateway-backed graph `langgraph dev` serves.

`langgraph.json` points at the module-level `graph` below. This builds a real
`AgentContext`:
  * chat model  -> the verified gateway slug `openai/gpt-5.4-mini`;
  * retriever   -> real OpenAI `text-embedding-3-large` over the committed desk
                   reviews in `data/reviews/`, in an in-memory Qdrant (no cloud
                   provisioning needed to iterate locally);
  * portfolio   -> the trader's anonymized book CSVs in `data/book/` parsed
                   into positions / trades / NAV, seeded under one dev user.

Everything degrades: if the embeddings/PDFs or the book CSVs are absent (a
fresh clone, no network), the corresponding loader returns `MissingData` and the
graph still starts — you can drive the scoper and see the cold-start path. The
pytest layer builds its own context with fakes, so importing this module (and its
network calls) is never required to run the tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY, PolicyRecord
from app.graphs.trading_assistant.tools import (
    make_desk_search_tool,
    make_market_quote_tool,
    make_search_web_tool,
    make_size_signal_tool,
)
from app.rag.chunk import chunk_document
from app.rag.index import OPENAI_3_LARGE_DIM, CorpusIndex, openai_embedder
from app.rag.retrieve import HybridRetriever, SharedCorpusRetriever
from app.trading.domain import MissingData
from app.trading.ingest.statement import (
    parse_account_nav,
    parse_activity_statement,
    parse_open_positions_cost,
    parse_splits,
    parse_statement_as_of,
)
from app.trading.ledger import apply_splits
from app.trading.ingest.tactical import parse_tactical_book

load_dotenv()

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
GATEWAY_MODEL = "openai/gpt-5.4-mini"  # verified live via the OpenAI-compat endpoint

# Seeded from the trader's anonymized book data, so the demo is coherent with
# the committed reviews. "alex-demo" stays reserved for synthetic isolation tests.
DEV_USER = "real-user"

_ROOT = Path(__file__).resolve().parents[3]  # .../Certification_Challenge
_REVIEWS = _ROOT / "data" / "reviews"
_BOOK = _ROOT / "data" / "book"
_TACTICAL_CSV = _BOOK / "Tactical_Boot.csv"
_STATEMENT_CSV = _BOOK / "IBKR YTD Statement.csv"


def gateway_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=GATEWAY_MODEL,
        base_url=GATEWAY_BASE_URL,
        api_key=os.environ["AI_GATEWAY_API_KEY"],
        temperature=0,  # a router wants stable routing (gateway accepts it for this slug)
        # No LLM call in this graph is user-facing (delivery is audit-gated;
        # visible messages are constructed in code), so keep every model run
        # out of the client's `messages` stream — otherwise the scoper's JSON
        # and the unaudited draft leak into the chat UI as extra bubbles.
        tags=["nostream"],
    )


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else None


def _build_retriever(user_id: str) -> HybridRetriever | None:
    """Index the committed reviews with real embeddings; None if unavailable."""
    pdfs = sorted(_REVIEWS.glob("*.pdf"))
    if not pdfs:
        return None
    try:
        index = CorpusIndex(
            QdrantClient(location=":memory:"),
            openai_embedder(),
            vector_size=OPENAI_3_LARGE_DIM,
        )
        for pdf in pdfs:
            index.replace_document(chunk_document(str(pdf)), user_id=user_id)
        # Cert-prototype mode: the committed reviews are baked in for every
        # username (coat-check identity, ADR-0005 amendment) — retrieval always
        # reads this one corpus regardless of the injected caller user_id.
        return SharedCorpusRetriever(HybridRetriever(index), owner=user_id)
    except Exception as exc:  # noqa: BLE001 - dev convenience: start even offline
        print(f"[dev] retriever unavailable ({exc!r}); desk questions will cold-start")
        return None


def _positions_loader():
    content = _read(_TACTICAL_CSV)
    positions = parse_tactical_book(content) if content else None
    missing = MissingData("positions snapshot", "Upload your tactical book export.")
    return lambda user_id: positions if positions is not None else missing


def _trades_loader():
    content = _read(_STATEMENT_CSV)
    trades = (
        apply_splits(parse_activity_statement(content), parse_splits(content))
        if content
        else None
    )
    missing = MissingData("ledger", "Upload a recent activity statement.")
    return lambda user_id: trades if trades is not None else missing


def _nav_loader():
    content = _read(_STATEMENT_CSV)
    nav = parse_account_nav(content) if content else None
    return lambda user_id: nav


def _as_of_loader():
    content = _read(_STATEMENT_CSV)
    as_of = parse_statement_as_of(content) if content else None
    return lambda user_id: as_of


def _entry_fallback_loader():
    content = _read(_STATEMENT_CSV)
    costs = parse_open_positions_cost(content) if content else {}
    return lambda user_id: costs


def _build_agent_tools(retriever, positions_loader, nav_loader, policy_loader):
    """The answering agent's live roster: quotes always (yfinance needs no key),
    signal sizing always (NAV-gated inside the tool), desk retrieval when the
    reviews indexed, Tavily web search only when its key is present. A missing
    piece just shrinks the roster."""
    tools = [
        make_market_quote_tool(),
        make_size_signal_tool(
            load_nav=nav_loader,
            load_policy=policy_loader,
            load_positions=positions_loader,
        ),
    ]
    if retriever is not None:
        tools.append(
            make_desk_search_tool(retriever, load_positions=positions_loader)
        )
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            tools.append(make_search_web_tool(TavilySearch(max_results=3)))
        except Exception as exc:  # noqa: BLE001 - dev convenience: start without web
            print(f"[dev] Tavily unavailable ({exc!r}); web search disabled")
    return tools


# In-process policy store: persists across turns within one `langgraph dev` run.
# The deploy build swaps this for the LangGraph Store / Postgres, same seam.
_POLICY: dict[str, PolicyRecord] = {}


def dev_context() -> AgentContext:
    positions_loader = _positions_loader()
    nav_loader = _nav_loader()
    policy_loader = lambda user_id: _POLICY.get(user_id, DEFAULT_POLICY)  # noqa: E731
    return AgentContext(
        chat_model=gateway_chat_model(),
        agent_tools=_build_agent_tools(
            _build_retriever(DEV_USER), positions_loader, nav_loader, policy_loader
        ),
        load_positions=positions_loader,
        load_trades=_trades_loader(),
        load_nav=nav_loader,
        load_as_of=_as_of_loader(),
        load_entry_fallback=_entry_fallback_loader(),
        load_policy=policy_loader,
        save_policy=lambda user_id, policy: _POLICY.__setitem__(user_id, policy),
        default_user_id=DEV_USER,
    )


graph = build_graph(dev_context())
