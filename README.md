# rag-from-scratch

My hands-on notes and code while learning Retrieval-Augmented Generation (RAG) with LangChain.
Each lesson comes in two versions: one using OpenAI, and one running fully locally with Ollama.

## Lessons

| # | Topic | What it covers |
|---|-------|----------------|
| 01 | RAG explained | The full retrieve → augment → generate loop |
| 02 | Embeddings & similarity | Turning text into vectors and measuring how close they are |
| 03 | Chunking strategies | Splitting documents so retrieval finds the right context |

## Run

make setup
make list
make run FILE=03_chunking_strategies/chunking_strategies_ollama.py

## Credits

Based on the YouTube series and notebooks by Yash Jain — https://github.com/yashjainio/rag-youtube.
The Ollama versions and restructuring are my own additions.
