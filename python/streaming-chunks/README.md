# Streaming chunks

**Goal:** process big inputs (files, API result pages, documents for RAG) without loading them all
into memory, using generators and a small context manager.

## What's here

- `chunks.py`: `chunked` (batches), `read_chunks` (file reader), `overlapping_chunks` (document
  windows with overlap), `timer` (context manager)
- `test_chunks.py`: tests including a laziness check

## Key ideas

- **Generators are lazy.** `test_chunked_is_lazy` proves only the first batch is pulled from the source.
- **Remainders matter.** The last partial batch must be yielded, or data is silently dropped.
- **Overlap for retrieval.** Overlapping windows keep sentences that cross a boundary findable. This is the
  chunking idea behind RAG pipelines; the right size and overlap are worth measuring, not guessing.
- **Context managers clean up on errors.** `timer` records the duration in a `finally` block, even if the body raises.
- **Walrus in the read loop.** `while chunk := f.read(size)` stops cleanly at end of file.

## Run

```bash
pytest python/streaming-chunks
```
