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