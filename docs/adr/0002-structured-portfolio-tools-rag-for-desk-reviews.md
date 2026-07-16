# Structured portfolio data via deterministic tools; RAG reserved for desk-review PDFs

The IBKR portfolio and trade CSVs are queried by **purpose-built deterministic tools**
(pandas-style: trade-history filter, exposure/hedge %, scale-out scan, position listing),
not embedded into the vector store. Retrieval reserved for the **desk-review corpus**
(prose PDFs). We rejected RAG-over-CSV because the portfolio questions (exposure %,
scale-out thresholds, "show me every NVDA trade", policy compliance) demand exactness and
completeness, and semantic top-k retrieval over rows of numbers is lossy by construction —
it silently drops rows and cannot do arithmetic. So "Agentic RAG" is satisfied by the desk
reviews (semantic), while the portfolio math stays a deterministic tool the agent calls.
We also rejected text-to-SQL in favour of fixed tools: reliability and a stable eval
target beat flexibility for a ~11-question eval set that collapses to ~4 computation types.
