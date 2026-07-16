# Synthesis-first, upload-based scope; live IB feed out of scope

The assistant's headline value is automating the trader's **synthesis** (reconciling his
book against his exposure policy and the desk's thesis and returning exactly what to
change) — the one step he still does entirely by hand. The staleness pains (manual
export, repeated re-upload, oversized files) are only *softened*, by holding an uploaded
snapshot in session memory, not solved with a live Interactive Brokers data feed. We
rejected live IB integration for this build: broker auth plus position/price streaming is
a multi-week effort that fights both the certification timeline and the "runs in a browser
on my phone" constraint, and it is not what an Agentic RAG prototype needs to prove.
