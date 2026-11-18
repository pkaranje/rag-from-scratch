# rag-from-scratch

My hands-on notes and code while learning Retrieval-Augmented Generation (RAG) with LangChain.
Each lesson comes in two versions: one using OpenAI, and one running fully locally with Ollama.

## Lessons

| # | Topic | What it covers | Notes |
| --- | --- | --- | --- |
| 01 | [RAG explained](01_rag_explained/) | The full retrieve → augment → generate loop | [docs](01_rag_explained/docs/rag_explained.md) |
| 02 | [Embeddings & similarity](02_embeddings_similarity/) | Turning text into vectors and measuring how close they are | [docs](02_embeddings_similarity/docs/embeddings_similarity.md) |
| 03 | [Chunking strategies](03_chunking_strategies/) | Splitting documents so retrieval finds the right context | [docs](03_chunking_strategies/docs/chunking_strategies.md) |
| 04 | [PDF & DOCX extraction](04_pdf_docx_extraction/) | Pulling text out of PDFs and Word files with PyMuPDF, Unstructured and Docling, and why tables need structure | [docs](04_pdf_docx_extraction/docs/pdf_docx_extraction.md) |
| 05 | [OCR & vision extraction](05_ocr_vision_extraction/) | Reading text from screenshots, scans, receipts and whiteboards with Tesseract OCR vs a vision LLM | [docs](05_ocr_vision_extraction/docs/ocr_vision_extraction.md) |

Each doc opens with a **Quick revision** section: what the lesson teaches, why it matters,
how the code does it, the gotchas, and self-check questions. Read that first when revising.

Each lesson folder has the same layout:

```text
NN_topic/
├── topic.ipynb / topic.py                # OpenAI version
├── topic_ollama.ipynb / topic_ollama.py  # local Ollama version
├── data/                                 # sample input the code reads (lessons 04-05 generate their own)
└── docs/topic.md                         # written walkthrough with diagrams and notes
```

## Prerequisites

- **Python 3.12** and [uv](https://docs.astral.sh/uv/). `make setup` installs both if missing.
- **For the OpenAI versions:** an API key in `.env`:

  ```bash
  cp .env.example .env   # then set OPENAI_API_KEY
  ```

  Models used: `gpt-5.4-nano` (chat) and `text-embedding-3-small` (embeddings).
- **For the Ollama versions:** [Ollama](https://ollama.com) running locally, with the models pulled:

  ```bash
  ollama pull llama3.2   # chat
  ollama pull bge-m3     # embeddings
  ```

- **For lesson 04:** poppler, which Unstructured uses to read PDFs:

  ```bash
  brew install poppler
  ```

  Docling downloads its layout model the first time it runs, so the first run is slower.
  Lesson 04's Ollama version uses only the chat model.

- **For lesson 05:** the Tesseract OCR engine, and for the Ollama version a vision model:

  ```bash
  brew install tesseract
  ollama pull gemma3:4b   # vision
  ```

## Run

```bash
make setup                                                   # first time only
make list                                                    # show runnable scripts
make run FILE=04_pdf_docx_extraction/pdf_docx_extraction_ollama.py
```

Or open the notebooks:

```bash
uv run jupyter lab
```

Other targets: `make format`, `make lint`, `make test`, `make clean`. Run `make` to see them all.

## Credits

Based on the YouTube series and notebooks by Yash Jain: [yashjainio/rag-youtube](https://github.com/yashjainio/rag-youtube)
(see [docs/original-source.md](docs/original-source.md)).
The Ollama versions and restructuring are my own additions.
