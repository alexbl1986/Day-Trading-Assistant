# Trading rules as a runtime-writable, structured procedural-memory record

The trader's exposure policy lives as a **structured, persisted procedural-memory record**
(typed numeric fields in a long-term store, seeded from a default config), not as a
git-committed static file and not as free-text the LLM paraphrases. Because this is a
single-owner app, the trader owns his rules and may change a threshold at runtime by asking
the agent — so the record must persist across restarts on an ephemeral deploy filesystem,
which rules out a committed YAML. Writes go through a single **confirmation-gated, typed
tool** (`update_policy(rule, value)`) that reads the change back before committing and keeps
provenance (old→new, timestamp). The guardrail's purpose is not to stop the owner but to
stop the LLM from silently inferring a policy change from loose phrasing and then quietly
passing a later compliance check against a cap he never meant to set; the compliance math
always reads exact numbers from the record.
