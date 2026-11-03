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
