from pathlib import Path
from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

DATA_DIR = (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "data"
# print(f"Data directory: {DATA_DIR.resolve()}")


# Deliberately about a fictional company - the base LLM can't know this,
# which is the point of the demo
QUESTION = "Who started Acme Robotics, and how long does their best robot's battery last?"

# Instruct the model to stick to the retrieved context instead of falling back on its own knowledge
RAG_PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def ask_naive(question: str) -> str | list[str | dict[Any, Any]]:
    # temperature=0 for reproducible output when we compare naive vs RAG answers
    llm = ChatOllama(model="llama3.2", temperature=0)
    response = llm.invoke(question)
    return response.content


def build_store() -> InMemoryVectorStore:
    """Embed every doc in data/ and load it into an in-memory vector store."""
    embeddings = OllamaEmbeddings(model="bge-m3")
    store = InMemoryVectorStore(embedding=embeddings)

    docs, metadatas = [], []
    # sorted() for a deterministic file order -- otherwise glob() order can vary by OS/filesystem,
    # which would make "chunk 2" mean a different doc on different runs
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(text)
        metadatas.append({"source": str(path.name)})

    # add_texts embeds all 5 docs in a single batched API call rather than one call per doc --
    # cheap here since the corpus is tiny, but this is the same batching instinct you want at scale
    store.add_texts(docs, metadatas=metadatas)
    print(f"Indexed {len(docs)} documents from {DATA_DIR}")
    return store


def ask_rag(
    store: InMemoryVectorStore, question: str, k: int = 3
) -> str | list[str | dict[Any, Any]]:
    # RETRIEVE - embed the question and pull the k most similar chunks
    results = store.similarity_search_with_score(question, k=k)
    print(f"\nRetrieved {len(results)} chunks:")
    for doc, score in results:
        print(f" score={score:.3f}  source={doc.metadata['source']}")

    # AUGMENT - stitch retrieved chunks into the prompt, separated so the model can tell them apart
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    prompt = RAG_PROMPT.format(context=context, question=question)

    # GENERATE - same naive LLM, but now grounded in the retrieved context
    llm = ChatOllama(model="llama3.2", temperature=0)
    return llm.invoke(prompt).content


def main() -> None:
    # print("=" * 70)
    # print("NAIVE LLM (no retrieval, no context)")
    # print("=" * 70)
    # print(f"Q: {QUESTION}\n")
    # answer = ask_naive(QUESTION)
    # print(f"A: {answer}")
    # print("\n^ Notice this is either a refusal or a hallucination — the model")
    # print("  has never seen the fictional 'Acme Robotics' company before.")

    print("Question:", QUESTION)

    print("\n" + "=" * 70)
    print("WITHOUT RAG")
    print("=" * 70)
    print(ask_naive(QUESTION))

    print("=" * 70)
    print("TINY RAG (retrieve -> augment -> generate)")
    print("=" * 70)
    store = build_store()
    print(f"\nQ: {QUESTION}")
    answer = ask_rag(store, QUESTION)
    print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
