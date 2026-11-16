from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

HANDBOOK_PATH = (
    (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "data" / "handbook.md"
)
print(f"Data directory: {HANDBOOK_PATH.resolve()}")


def load_handbook_text() -> str:
    return HANDBOOK_PATH.read_text()


def fixed_size_chunks(text: str, size: int = 500) -> list[Document]:
    """Naive: cut every N characters, ignoring sentence/section boundaries."""
    # separator="" forces a hard cut at exactly `size` chars, mid-word if it lands there
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=0)
    return [Document(page_content=c) for c in splitter.split_text(text)]


def recursive_size_chunks(text: str, size: int = 500) -> list[Document]:
    """Naive: cut every N characters, ignoring sentence/section boundaries."""
    # separator="" forces a hard cut at exactly `size` chars, mid-word if it lands there
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=50)
    return [Document(page_content=c) for c in splitter.split_text(text)]
