from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)

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


def ask_naive(question: str) -> str:
    # temperature=0 for reproducible output when we compare naive vs RAG answers
    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
    response = llm.invoke(question)
    return response.content


def build_store() -> InMemoryVectorStore:
    """Embed every doc in data/ and load it into an in-memory vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = InMemoryVectorStore(embeddings)

    docs, metadatas = [], []
    # sorted() for a deterministic file order -- otherwise glob() order can vary by OS/filesystem,
    # which would make "chunk 2" mean a different doc on different runs
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append(path.read_text())
        # keep the filename around so we can show provenance for each retrieved chunk
        metadatas.append({"source": path.name})

    # add_texts embeds all 5 docs in a single batched API call rather than one call per doc --
    # cheap here since the corpus is tiny, but this is the same batching instinct you want at scale
    store.add_texts(docs, metadatas=metadatas)
    print(f"Indexed {len(docs)} documents from {DATA_DIR}")
    return store


def ask_rag(store: InMemoryVectorStore, question: str, k: int = 3) -> str:
    # RETRIEVE - embed the question and pull the k most similar chunks
    results = store.similarity_search_with_score(question, k=k)
    print(f"\nRetrieved {len(results)} chunks:")
    for i, (doc, score) in enumerate(results):
        print(f"  chunk {i + 1} (score={score:.4f}): {doc.metadata['source']}")

    # COMBINE - feed the retrieved context into the LLM with a prompt
    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
    context = "\n\n".join(doc.page_content for doc, _ in results)
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content


def main() -> None:
    # print("=" * 70)
    # print("NAIVE LLM (no retrieval, no context)")
    # print("=" * 70)
    # print(f"Q: {QUESTION}\n")
    # naive_answer = ask_naive(QUESTION)
    # print(f"A: {naive_answer}")
    # print("\n^ Notice this is either a refusal or a hallucination — the model")
    # print("  has never seen the fictional 'Acme Robotics' company before.")

    print("Question:", QUESTION)

    print("\n" + "=" * 70)
    print("WITHOUT RAG")
    print("=" * 70)
    print(ask_naive(QUESTION))

    print("\n" + "=" * 70)
    print("WITH RAG")
    print("=" * 70)
    # rebuilt here rather than reusing the `store` from cell 10 -- this cell is meant to be a
    # standalone "run everything" demo, and re-embedding 5 tiny docs is cheap enough not to matter
    store = build_store()
    print(ask_rag(store, QUESTION))


if __name__ == "__main__":
    main()
