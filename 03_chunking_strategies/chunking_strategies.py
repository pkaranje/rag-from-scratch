from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# override=True so re-running this cell picks up edits to .env without restarting the kernel
load_dotenv(override=True)

HANDBOOK_PATH = (
    (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "data" / "handbook.md"
)
print(f"Data directory: {HANDBOOK_PATH.resolve()}")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.4-nano"

QUERIES = [
    "When should the robot's wheels be replaced?",
    "What happens if two robots both want the same pod?",
]

ANSWER_PROMPT = """Answer using ONLY this context. Be concise.

Context:
{context}

Question: {question}
Answer:"""


def load_handbook_text() -> str:
    return HANDBOOK_PATH.read_text()


def fixed_size_chunks(text: str, size: int = 500) -> list[Document]:
    """Naive: cut every N characters, ignoring sentence/section boundaries."""
    # separator="" forces a hard cut at exactly `size` chars, mid-word if it lands there
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=0)
    return [Document(page_content=c) for c in splitter.split_text(text)]


def recursive_size_chunks(text: str, size: int = 500, overlap: int = 50) -> list[Document]:
    """Tries paragraph, then sentence, then word boundaries before hard-cutting."""
    # overlap keeps a little trailing context from the previous chunk so meaning
    # isn't lost when a sentence gets split right at the boundary
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [Document(page_content=c) for c in splitter.split_text(text)]


def semantic_chunks(text: str) -> list[Document]:
    """Splits where embedding similarity between consecutive sentences drops —
    i.e. where the *topic* actually changes, not at a fixed character count."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # percentile: flag a breakpoint wherever the similarity drop is in the top
    # percentile of drops seen across the whole doc, rather than a fixed threshold
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return splitter.create_documents([text])


def markdown_header_chunks(text: str) -> list[Document]:
    """Splits on `#`/`##`/`###` headers, so a chunk never spans two sections
    and each chunk's metadata records which chapter/subsection it came from."""
    # each tuple is (markdown header syntax, metadata key) -- the splitter both cuts the
    # text at every occurrence of that header level AND stamps the resulting chunk's
    # metadata with the header text under that key, so a chunk "knows" which chapter/
    # section/subsection it came from without you having to track it separately
    headers = [("#", "chapter"), ("##", "section"), ("###", "subsection")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    return splitter.split_text(text)


def build_store(chunks: list[Document]) -> InMemoryVectorStore:
    # fresh in-memory store per strategy so results aren't mixed across chunking approaches
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = InMemoryVectorStore(embeddings)
    store.add_texts([c.page_content for c in chunks])
    return store


def answer_with_strategy(
    store: InMemoryVectorStore, query: str, llm: ChatOpenAI
) -> tuple[str, list[str]]:
    results = store.similarity_search(query, k=2)
    context = "\n---\n".join(d.page_content for d in results)
    answer = llm.invoke(ANSWER_PROMPT.format(context=context, question=query)).content
    # truncate + flatten newlines just for readable console output, not passed to the LLM
    return answer, [d.page_content[:100].replace("\n", " ") for d in results]


def build_parent_document_retriever(text: str) -> ParentDocumentRetriever:
    # small child chunks give precise vector-search matches; large parent chunks
    # give the LLM enough surrounding context once a child chunk is matched
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)

    vectorstore = Chroma(
        collection_name="parent_doc_demo",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    # docstore holds the full parent chunks by id; only the small child chunks get embedded
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents([Document(page_content=text)])
    return retriever


def compare_sizes(text: str) -> None:
    print(f"Handbook length: {len(text)} chars, {len(text.split())} words\n")
    for name, chunks in [
        ("fixed_size", fixed_size_chunks(text)),
        ("recursive", recursive_size_chunks(text)),
        ("markdown_header", markdown_header_chunks(text)),
    ]:
        # fixed_size should be near-uniform; the others vary because they respect boundaries
        sizes = [len(c.page_content) for c in chunks]
        print(f"{name:16s} -> {len(chunks):3d} chunks, sizes {min(sizes)}-{max(sizes)} chars")
    print()


def compare_retrieval(text: str) -> None:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    strategies = {
        "fixed_size (500 chars, no overlap)": fixed_size_chunks(text),
        "recursive (500 chars, 50 overlap)": recursive_size_chunks(text),
        "markdown_header (split on # / ## / ###)": markdown_header_chunks(text),
        "semantic (embedding-similarity breakpoints)": semantic_chunks(text),
    }
    # one vector store per strategy so the same query runs against each independently
    stores = {name: build_store(chunks) for name, chunks in strategies.items()}
    for query in QUERIES:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        for name, store in stores.items():
            answer, snippets = answer_with_strategy(store, query, llm)
            print(f"\n[{name}] ({len(strategies[name])} chunks)")
            for s in snippets:
                print(f"  retrieved: ...{s}...")
            print(f"  answer: {answer}")
        print()


def parent_document_demo(text: str) -> None:
    retriever = build_parent_document_retriever(text)
    query = QUERIES[0]

    # what the vector search actually matched against (small, precise)
    child_hits = retriever.vectorstore.similarity_search(query, k=1)
    print(f"Query: {query}\n")
    print(f"Matched CHILD chunk ({len(child_hits[0].page_content)} chars):")
    print(f"  {child_hits[0].page_content!r}\n")

    # what actually gets returned to the LLM (larger, more context)
    parent_hits = retriever.invoke(query)
    print(f"Returned PARENT chunk ({len(parent_hits[0].page_content)} chars):")
    print(f"  {parent_hits[0].page_content[:300]!r}...")


def main():
    handbook_text = load_handbook_text()
    compare_sizes(handbook_text)
    compare_retrieval(handbook_text)
    parent_document_demo(handbook_text)


if __name__ == "__main__":
    main()
