# Data classification and access policy for a public repo and endpoint

The cert submission requires a public GitHub repo and a public deployment, but the working
data is someone else's real financial records (the trader's IBKR statements: name, account
number, full trade history) and the desk's paid research PDFs. Policy, by class:

1. **Real IBKR statements — never committed.** Gitignored from day one. The repo ships a
   **synthetic sample statement** (same 27-section IBKR format, fake identity, scaled/
   shuffled trades) as the unit-test fixture and the file graders upload to try the app.
2. **Real desk-review PDFs — never committed** (conservative default; may relax if the
   desk confirms the content is not proprietary — pending). The repo ships **synthetic
   Hebrew desk reviews** in the desk's template styles (one daily, one weekly — the
   runtime corpus holds only the latest of each; stance-shift history is Demo Day scope).
3. **Eval artifacts — split by sensitivity.** Committed: question sets, ground-truth
   judgments in our own words, summary metrics tables. Not committed: raw retrieval traces
   and quoted chunks — those live in the private LangSmith project (EU endpoint), which is
   the full evidence record. Tracing sits behind an env flag so real-use sessions can run
   untraced.
4. **The deployed endpoint is gated.** The running app — not the repo — is where real data
   lives (uploads land in Postgres/Qdrant). Simple credential login (no OAuth; two known
   users, server-side check, session token). The authenticated `user_id` scopes every
   LangGraph store/checkpointer namespace, the portfolio snapshot, **and the desk-review
   corpus** (a `user_id` payload filter in Qdrant), so the demo user (synthetic data) and
   the real trader (real data) are fully isolated — a shared corpus would let the demo
   user's synthetic desk reviews poison the real trader's answers. "Public endpoint"
   means reachable, not unauthenticated — graders get credentials privately in the
   submission.

Rationale: a push to a public repo is irreversible, and an ungated deployment holding a
real book is a larger leak than the repo could ever be. Synthetic fixtures beat excerpts
because excerpting paid research is still redistribution.

## Amendment (2026-07-15): anonymized statement, coat-check identity

The trader anonymized the IBKR YTD statement (identity fields removed, raw numbers only)
and approved its use in the prototype, which dissolves the premise behind item 4: the
deployed endpoint no longer holds data worth a credential gate. For the cert submission:

- **The anonymized statement and the desk-review PDFs are baked into the deployment**
  and shared by every user (`SharedCorpusRetriever` serves one corpus to all callers).
  Item 1's "never committed" applies to the *identified* statement; the anonymized
  version may ship with the prototype.
- **Login is identification, not authentication** (a coat-check ticket): any username,
  no password. A username owns its threads (LangGraph thread metadata `owner`) and its
  policy record; returning usernames resume their latest thread, "New chat" mints
  another. Isolation is therefore *demonstrated* (per-user policy + conversations over
  shared read-only data), not *enforced* — anyone can present any name, and that is an
  accepted, stated property of the prototype.
- **Credential auth (item 4 as originally written) moves to Demo Day hardening**, where
  per-user uploads and per-user corpora return and the graph's per-call user binding
  (already in place) gets a verified identity behind it.
