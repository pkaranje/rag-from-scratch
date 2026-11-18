import json
from pathlib import Path

from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

# override=True so re-running this cell picks up edits to .env without restarting the kernel
load_dotenv(override=True)

DATA_DIR = (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "data"
OUT_DIR = DATA_DIR / "extracted"

CHAT_MODEL = "gpt-5.4-nano"

# {text} (not {context}) to match the .format(text=..., question=...) call below
ANSWER_PROMPT = """Answer using ONLY the extracted text below. Be concise.

Extracted text:
{text}

Question: {question}
Answer:"""

print(f"DATA_DIR: {DATA_DIR.resolve()}")
print(f"OUT_DIR: {OUT_DIR.resolve()}")

# Generate the sample data

STYLES = getSampleStyleSheet()


def make_clean_digital_pdf():
    # simplest case: single column, no tables — the baseline every extractor should handle
    path = DATA_DIR / "clean_digital.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    body: list[Flowable] = [
        Paragraph("Meridian Analytics — Q3 Platform Update", STYLES["Title"]),
        Paragraph(
            "Meridian Analytics shipped three major platform updates in Q3: a rewritten "
            "ingestion pipeline, a new anomaly-detection model, and a redesigned billing "
            "dashboard. Customer-reported latency on the ingestion pipeline dropped by "
            "41 percent after the rewrite, measured across the top 50 accounts by volume.",
            STYLES["BodyText"],
        ),
        Paragraph(
            "The anomaly-detection model, internally called Lighthouse, reduced false "
            "positive alerts from 1 in 12 to 1 in 47 during the beta period. Lighthouse "
            "will roll out to all paying customers in Q4.",
            STYLES["BodyText"],
        ),
    ]
    doc.build(body)
    print(f"wrote {path}")


def make_multi_column_pdf():
    path = DATA_DIR / "multi_column.pdf"

    doc = BaseDocTemplate(str(path), pagesize=letter)
    margin = 0.75 * inch

    # two frames side-by-side (with a 0.3" gutter between them) is what makes this a
    # true multi-column layout rather than one wide text block
    col_width = (letter[0] - 2 * margin - 0.3 * inch) / 2
    title_height = 0.8 * inch
    col_height = letter[1] - 2 * margin - title_height

    # full-width title frame on top so both columns start at the same height
    title_frame = Frame(
        margin, margin + col_height, letter[0] - 2 * margin, title_height, id="title"
    )
    left_frame = Frame(margin, margin, col_width, col_height, id="left")
    right_frame = Frame(margin + col_width + 0.3 * inch, margin, col_width, col_height, id="right")

    doc.addPageTemplates([PageTemplate(id="TwoCol", frames=[title_frame, left_frame, right_frame])])

    left_text = (
        "Left column: Meridian's support team resolved 2,300 tickets in Q3, an increase "
        "of 15 percent over Q2. Average first-response time held steady at 6 minutes "
        "despite the higher volume, largely due to the new triage-bot rollout in August."
    )
    right_text = (
        "Right column: Headcount grew from 84 to 96 employees in Q3. Two new hires joined "
        "the Lighthouse anomaly-detection team, and one new hire joined support. No "
        "departures were recorded in the quarter."
    )
    body: list[Flowable] = [
        Paragraph("Meridian Analytics — Internal Ops Digest", STYLES["Title"]),
        FrameBreak(),
        Paragraph(left_text, STYLES["BodyText"]),
        # short text never overflows the left frame, so force the jump to the right column
        FrameBreak(),
        Paragraph(right_text, STYLES["BodyText"]),
    ]
    # naive PDF text extractors often read column 1 then column 2 out of order,
    # or interleave them line-by-line — that's the failure mode this file demos
    doc.build(body)
    print(f"wrote {path}")


def make_table_heavy_pdf():
    # the answer lives in row/column structure: flat extractors lose which number
    # belongs to which column, and the "Results" header spans two columns
    path = DATA_DIR / "table_heavy.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    cell = STYLES["BodyText"]
    rows = [
        [
            "Disability Category",
            "Participants",
            "Ballots Completed",
            "Ballots Incomplete/ Terminated",
            "Results",
            "",
        ],
        ["", "", "", "", "Accuracy", "Time to complete"],
        ["Blind", "5", "1", "4", "34.5%, n=1", "1199 sec, n=1"],
        [
            "Low Vision",
            "5",
            "2",
            "3",
            "98.3% n=2<br/>(97.7%, n=3)",
            "1716 sec, n=3<br/>(1934 sec, n=2)",
        ],
        ["Dexterity", "5", "4", "1", "98.3%, n=4", "1672.1 sec, n=4"],
        ["Mobility", "3", "3", "0", "95.4%, n=3", "1416 sec, n=3"],
    ]
    # explicit widths: Paragraph cells otherwise collapse and wrap letter-by-letter
    table = Table(
        [[Paragraph(c, cell) for c in r] for r in rows],
        colWidths=[1.1 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch, 1.2 * inch, 1.3 * inch],
        repeatRows=2,
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 1), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("SPAN", (4, 0), (5, 0)),  # "Results" over Accuracy + Time
                *[("SPAN", (c, 0), (c, 1)) for c in range(4)],  # first 4 headers span 2 rows
            ]
        )
    )
    body: list[Flowable] = [
        Paragraph("Example table", STYLES["Title"]),
        Paragraph("This is an example of a data table.", STYLES["BodyText"]),
        table,
    ]
    doc.build(body)
    print(f"wrote {path}")


def make_docx():
    path = DATA_DIR / "sample.docx"
    doc = DocxDocument()
    doc.add_heading("Meridian Analytics — Hiring Plan", level=1)
    doc.add_paragraph(
        "Meridian Analytics plans to hire 18 engineers in Q4, split evenly between the "
        "ingestion team and the Lighthouse anomaly-detection team."
    )
    doc.add_heading("Roles", level=2)
    for role in ["Senior Backend Engineer", "ML Engineer", "Platform SRE"]:
        doc.add_paragraph(role, style="List Bullet")
    doc.save(str(path))
    print(f"wrote {path}")


def extract_with_pymupdf(pdf_path: Path) -> str:
    # imported lazily inside the function so this cell can run even if an extractor's
    # heavy optional dependency isn't installed, as long as you don't call that one
    from langchain_community.document_loaders import PyMuPDFLoader

    docs = PyMuPDFLoader(str(pdf_path)).load()
    return "\n\n".join(d.page_content for d in docs)


def extract_with_unstructured(pdf_path: Path) -> str:
    from langchain_community.document_loaders import UnstructuredPDFLoader

    # mode="elements" keeps each detected layout element (title, table, list item, ...)
    # as its own Document instead of flattening the whole page into one blob
    docs = UnstructuredPDFLoader(str(pdf_path), mode="elements").load()
    return "\n\n".join(d.page_content for d in docs)


def extract_with_docling(pdf_path: Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    # markdown output preserves table/heading structure that plain-text extractors lose
    return result.document.export_to_markdown()


def extract_docx(docx_path: Path) -> str:
    from langchain_community.document_loaders import Docx2txtLoader

    docs = Docx2txtLoader(str(docx_path)).load()
    return "\n\n".join(d.page_content for d in docs)


def diff_extractors(pdf_path: Path) -> dict[str, str]:
    print(f"\n--- {pdf_path.name} ---")
    results = {}
    for name, fn in [
        ("pymupdf", extract_with_pymupdf),
        ("unstructured", extract_with_unstructured),
        ("docling", extract_with_docling),
    ]:
        try:
            text = fn(pdf_path)
            results[name] = text
            print(f"  {name:14s} -> {len(text):5d} chars")
        except Exception as e:  # extractors have heavy optional deps; report and continue
            print(f"  {name:14s} -> FAILED: {e}")
    return results


def chunk_and_save(file_name: str, text: str):
    OUT_DIR.mkdir(exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # tag each chunk with its source filename so downstream retrieval can cite provenance
    chunks = [{"text": c, "metadata": {"source": file_name}} for c in splitter.split_text(text)]
    out_path = OUT_DIR / f"{Path(file_name).stem}.chunks.json"
    out_path.write_text(json.dumps(chunks, indent=2))
    print(f"  wrote {len(chunks)} chunks -> {out_path}")


def bad_vs_good_extraction_demo():
    """pymupdf reads table cells as one flat left-to-right text stream with no
    row/column structure. On a small table an LLM can usually still untangle it,
    but that's luck, not a guarantee. docling exports proper markdown table
    syntax, so the LLM never has to guess which number belongs to which column.
    Bigger or merged-cell tables (Episode 06) are where the flat version starts
    giving wrong answers."""
    pdf_path = DATA_DIR / "table_heavy.pdf"
    question = (
        "For the Low Vision category, how many ballots were completed, "
        "what was the accuracy, and how long did they take to complete?"
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    flat_text = extract_with_pymupdf(pdf_path)
    structured_text = extract_with_docling(pdf_path)

    print(f"\nQuestion: {question}")
    for label, text in [
        ("pymupdf (flat, structure lost)", flat_text),
        ("docling (structured markdown table)", structured_text),
    ]:
        answer = llm.invoke(ANSWER_PROMPT.format(text=text, question=question)).content
        print(f"\n[{label}]")
        print(f"  extracted: {text[:150]!r}...")
        print(f"  answer: {answer}")


def main():
    make_clean_digital_pdf()
    make_multi_column_pdf()
    make_table_heavy_pdf()
    make_docx()

    for pdf_name in ["clean_digital.pdf", "multi_column.pdf", "table_heavy.pdf"]:
        results = diff_extractors(DATA_DIR / pdf_name)
        # only pymupdf's output gets chunked/saved; it's the baseline used for
        # downstream retrieval, while unstructured/docling are just compared above
        if "pymupdf" in results:
            chunk_and_save(pdf_name, results["pymupdf"])

    print("\n--- sample.docx ---")
    docx_text = extract_docx(DATA_DIR / "sample.docx")
    print(f"  docx2txt -> {len(docx_text)} chars")
    chunk_and_save("sample.docx", docx_text)

    print("\n" + "=" * 70)
    print("BAD EXTRACTION -> BAD ANSWER  vs  GOOD EXTRACTION -> GOOD ANSWER")
    print("=" * 70)
    bad_vs_good_extraction_demo()


if __name__ == "__main__":
    main()
