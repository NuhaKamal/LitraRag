# LitraRag design decisions

## OCR resolution: `ocr_dpi = 200`
- Benchmark (ocr demo): text unreadable at 8 px, readable at 12 px, accuracy plateaus at ~20 px.
- 10 pt text at 200 DPI ≈ 28 px; 8 pt footnotes ≈ 22 px, both on the plateau.
- 300 DPI would cost ~2.25× the pixels for no accuracy gain.
- Recheck 150 vs 200 with real documents in Phase 5.

## OCR parallelism: `max_workers = 4`
- Benchmark (loader demo, 8 scanned pages):
  1 worker 2291 ms, 2 → 1170 ms, 4 → 812 ms, 6 → 748 ms.
- 4 workers = 2.8× speedup; 6 adds only 8% with 50% more Tesseract processes.
- Limits: batch barrier in `_ocr_pages`, serial rendering (PyMuPDF), CPU cores.
- Phase 5: sliding window instead of batches; test `OMP_THREAD_LIMIT=1`.

## Chunking: `chunk_size = 800`, `chunk_overlap = 100`
- Benchmark (chunker demo, ~313-char paragraphs):
  300 → 300 chunks (+11% stored, cuts mid-paragraph); 500 → 150; 800 → 75; 1200 → 60.
- Chunking takes milliseconds; it matters because it sets the embedder's workload.
- From 500 to 1200, total text stored is ~100%, so embedding compute is similar;
  batching removes most of the per-call difference.
- Overlap only applies when the splitter cuts inside a paragraph.
- 800 ≈ 2 paragraphs: precise retrieval, enough context, 4 chunks fit in 2048 tokens.
- Phase 5: compare 500 / 800 / 1200 on the golden set (retrieval quality).

## Embedding batch size: `embed_batch_size = 16`
- Benchmark (embedder demo, 64 texts × ~650 chars, nomic-embed-text on laptop CPU,
  reversed order, warm-up per batch size):
  64 → 17.8 s, 16 → 18.6 s, 4 → 17.2 s, 1 → 18.6 s, all within ~8% (noise).
- An earlier run in fixed order without per-size warm-up suggested "bigger is slower";
  that was an artifact of the first-request penalty (~2–3 s per new configuration).
- Embedding on CPU is compute-bound: ~3.5 texts/s for ~160-token texts, ~22 texts/s
  for ~10-token texts. Cost scales with total text, not request count.
- 16 is neutral on CPU, suits GPU batching, and limits work lost on a failed request.
- Recheck on Colab GPU, where batching should matter much more.
- Implication: on this laptop, ~3,000 chunks ≈ 14 min. The real levers are the
  embedding cache and GPU, not batching.

## Embedding cache: SQLite, key = hash(model, kind, text)
- Demo (32 texts × ~650 chars, nomic-embed-text, laptop CPU):
  cold 11,416 ms; warm 1 ms; 4 of 32 edited 1,244 ms (only the 4 re-embedded).
- 3,000 chunks with 1% changed: ~15 min without cache → ~9 s with it.
- Keyed by text, not chunk_id, so identical text is never embedded twice.
  Model and kind (document/query) are part of the key.
- Per-page chunking limits the ripple of an edit: shifted chunk boundaries
  stop at the page end, so only that page's chunks miss the cache.
- The cache is a decorator (CachedEmbedder wraps any Embedder); a separate file
  that can always be deleted safely.

## Vector store: Qdrant embedded mode (for now)
- No Docker: saves ~1.5–2 GB RAM on an 8 GB Windows laptop.
- Embedded mode is a pure-Python implementation: exact brute-force search, fine for a
  personal collection; the server (Rust, HNSW index) scales much further.
- Embedded storage allows one process at a time: close the chat app before ingesting.
- Switching is one setting (RAG_QDRANT_MODE=server); only open_qdrant() differs.
- Phase 5 experiment: benchmark embedded vs server (search latency, RAM) on real documents.
- Points store both doc_id (version) and source_path (file): an edited file gets a new
  doc_id, so replacement is by source_path.

## First real ingestion (laptop CPU, Qdrant embedded)
- 4 files (280-page manual, 2 smaller PDFs, 1 text file) → 888 chunks in 254.9 s.
- Embedding ~92% of total (3.7 texts/s, as the benchmark predicted).
  Loading + OCR ~3%, chunking ~0%, Qdrant storing ~3% (~10 ms per chunk).
- Unchanged re-run: 0.1 s for the ingest loop (hash + one lookup per file),
  excluding process startup.
- Biggest remaining lever for first-time ingestion: GPU embedding (Colab).

## Backfill and the shifting bottleneck
- Keyword backfill of 888 cached chunks: 22.2 s (vs 254.9 s first ingestion).
- New bottleneck: embedded Qdrant storing, 14.1 s (~16 ms/chunk), vs FTS5 0.13 s.
- Pipeline now writes only to the store that is missing a document version.
- Strong input for the Phase 5 embedded-vs-server Qdrant benchmark.

## Prompt choice and the limits of llama3.2:1b
- Compared V1, V2, V3 × with/without file-page labels (scripts/compare_prompts.py),
  3 runs each, on two cases: answer from a single short passage, and decline when
  the context lacks the answer.
- Every variant declined correctly (except V1, which echoed a passage label).
- No variant answered from a single short passage: llama3.2:1b over-refuses there.
  With real chunks (~800 chars, 4 passages) it answers in the chat.
- Kept V2 + labels: never hallucinated, answers real questions.
- The single-passage test is marked xfail; expect it to pass with a larger model
  (Colab gemma2:9b), which is the real fix.
