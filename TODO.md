# TODO — candidate feature backlog

**Status: hypotheses, not commitments.** This is a harvest of features from other repos in
`~/dev` (muser, feed-demon, exp-notes-indexing, exp-daily-card, davinci-resolve-mcp,
mcp-app-harness) done 2026-06-17. **It is entirely possible none of these matter.** Default
is *don't build* — each item must pass a validation gate (a cheap test on the real ~2,255-note
corpus that shows it actually improves search/feed/synthesis) before it earns implementation.
Prefer killing items over building them. The only things here that are unconditionally worth
doing are the Tier-0 cleanups.

Effort = S/M/L. Value = IF it works (most are unproven).

---

## Tier 0 — cleanups / already-built (no validation needed)

- [ ] **Expose `synthesize` / `clusters` / `vote` as MCP tools.** Implemented but web-UI-only,
  so MCP clients (Claude Desktop) can't reach them. Just `server.tool()` registrations. — S, high.
- [ ] **Drop dead deps** `@ai-sdk/openai`, `run-jxa` (JXA abandoned for direct SQLite). — trivial.
- [ ] **Remove stale agent worktree** `.claude/worktrees/agent-a242c38f99e5d2ae1/` (~300 behind main). — trivial.

## Tier 1 — differentiator candidates (test before committing)

- [ ] **Force-directed bridge viz + verbatim quote-pair evidence.** Port `connections.html`
  (force sim, arcs bowing *through* the bridge note) + `verify_quote` (substring-verified A/C
  quotes; also caught a plaintext password). From exp-notes-indexing. — M, high.
  - *Validate:* does the graph view actually make bridges more legible than the list, on a real
    corpus? Gate the LLM quote call behind an explicit "explain this bridge" action ($0 default).
- [ ] **ε-greedy "explore" slot in the feed.** Inject one off-profile wildcard every ~6 items so
  the vote-trained loop doesn't narrow (McNee 2006). From feed-demon `inject_explore`. — S, high.
  - *Validate:* do explore items get engaged with, or ignored as noise? A/B the feed with/without.
- [ ] **Image / attachment CLIP search.** Embed note attachments (sketches/scans/photos) with
  CLIP-B/32 in the existing transformers.js stack; second LanceDB table + an image search arm.
  From muser. — L, high (the one *net-new* capability).
  - *Validate:* how many notes even have images, and would you ever query them? Count attachments
    in the real store FIRST — if it's <5% of notes, this L bet probably doesn't pay off.

## Tier 2 — depth candidates

- [ ] **Three-axis feed (relevance / taste / novelty).** Split the existing single sigmoid's
  separable signals into named axes; surface per-axis "why" on the card. feed-demon. — M, high.
  - *Validate:* is the decomposition actually more useful than the current single score, or just
    more UI? Only worth it if the axes let you tune behavior the single score can't.
- [ ] **Temporal-trajectory synthesis.** Cluster → order by date → "how your thinking evolved."
  The "arm nobody ships"; ingredients (clustering, dated chunks, cited synthesis) already in TS.
  exp-notes-indexing (unbuilt). — M, high *strategically*.
  - *Validate:* UNPROVEN — never built. Build a thin version, rate it through the feed loop. May
    read as a gimmick; don't claim it works until rated.
- [ ] **Daily-digest card + notification.** Surface today's new bridges as one card; turns a
  pull-tool into a daily habit. exp-daily-card. — S–M, high.
  - *Validate:* would you actually open it daily, or is it notification spam? Dogfood for a week.

## Tier 3 — polish candidates

- [ ] **Embedding dedup (`rep_of` Union-Find)** — collapse "TODO ×10" near-dupes in search + feed.
  muser. — M, med-high. *Validate:* count actual near-dupes at threshold ~0.99 first.
- [ ] **Per-folder vote-rate prior + spaced-rep novelty weighting** — one feature the SGD learns
  automatically; feed-demon's own TODO calls this its top lever. feed-demon + exp-daily-card. — S, med-high.
- [ ] **Punchline compression pass** — one-line hook per card ("slop was the rendering register").
  exp-notes-indexing. — S, med. Apply lazily (on-expand) to keep the feed $0/local.
- [ ] **MCP-app gallery UI** for search/feed/bridges (interactive in-host vs plain text). muser +
  mcp-app-harness `mcp-frame.ts`. — S–M. *Verify the MCP UI doesn't already use the AppBridge.*
- [ ] **LLM cluster labels on the Map** (vs TF-IDF terms). muser. — S, low-med.
- [ ] **Resumable bridge/cluster job cache** (create→slice→resume→cancel→archive + hash reuse).
  davinci-resolve-mcp. — M, med-high *only once the corpus is large.*

## Explicitly NOT building (decided — don't revisit)

- **Graphiti/Kuzu knowledge graph** — benchmarked + shelved (super-linear O(N²) cost, unstable,
  typed-relations scored weak 0.13; embeddings already relate 100% of notes for $0). See
  `~/dev/exp-notes-indexing/BENCHMARK.md`.
- **muser image-taste oracle / aesthetic scoring / C2PA / color search** — no analog in prose.
- **RAPTOR recursive cluster-tree** — scored *below* bridges (8.2 vs 9.0) in the bake-off.
- **muser spawn-a-service architecture** — this is one unified bun process; don't add IPC.
