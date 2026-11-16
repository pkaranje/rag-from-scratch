import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from langchain_ollama import OllamaEmbeddings

DATA_PATH = (
    (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "data" / "sentences.json"
)
# print(f"Data directory: {DATA_PATH.resolve()}")
OUTPUT_PATH = "sentence_embeddings_2d.png"


def load_sentences() -> tuple[list[str], list[str]]:
    """Returns (sentences, topics) as two parallel lists of equal length."""
    raw = json.loads(DATA_PATH.read_text())
    sentences, topics = [], []
    # flatten the {topic: [sentences]} dict into parallel lists so index i always
    # maps sentences[i] to the topic it came from
    for topic, sents in raw.items():
        for s in sents:
            sentences.append(s)
            topics.append(topic)
    return sentences, topics


def embed_sentences(sentences: list[str]) -> np.ndarray:
    embedder = OllamaEmbeddings(model="bge-m3")
    vectors = embedder.embed_documents(sentences)
    return np.array(vectors)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    # normalize every row to unit length first, so a plain dot product IS the cosine similarity
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return norms @ norms.T


def dot_product_matrix(vectors: np.ndarray) -> np.ndarray:
    # unnormalized dot product - unlike cosine, this is sensitive to vector magnitude
    return vectors @ vectors.T


def euclidean_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, expanded to avoid an O(n^2) python loop
    sq_norms = np.sum(vectors**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * (vectors @ vectors.T)
    return np.sqrt(np.maximum(sq_dists, 0))


def project_to_2d(vectors):
    # n_neighbors must be < number of points; we only have 30 sentences
    reducer = umap.UMAP(n_neighbors=8, min_dist=0.3, random_state=42)
    return reducer.fit_transform(vectors)


def main():
    sentences, topics = load_sentences()
    print(f"Loaded {len(sentences)} sentences across {len(set(topics))} topics")
    print(list(set(topics)))

    print(f"Embedding {len(sentences)} sentences with text-embedding-3-small...")

    vectors = embed_sentences(sentences)
    print(f"Vector shape: {vectors.shape}  (1536 dimensions per sentence)\n")

    cosine = cosine_similarity_matrix(vectors)
    dot = dot_product_matrix(vectors)
    euclid = euclidean_distance_matrix(vectors)

    # Pick one sentence from each topic and show its nearest neighbors by cosine sim
    anchors = [0, 10, 20]  # first sentence of cooking, football, programming
    for i in anchors:
        print(f'Nearest neighbors of: "{sentences[i]}"  [{topics[i]}]')
        # sort all sentence indices by similarity to i, descending
        ranked = sorted(range(len(sentences)), key=lambda j: -cosine[i, j])
        for j in ranked[1:4]:  # skip itself
            print(
                f"  cos={cosine[i, j]:.3f}  dot={dot[i, j]:.1f}  "
                f"euclid={euclid[i, j]:.3f}  [{topics[j]}] {sentences[j]}"
            )
        print()

    # Compare average similarity within a topic vs across topics - the gap is the
    # signal that embeddings actually cluster by meaning, not just by wording
    same_topic = [
        cosine[i, j] for i in range(30) for j in range(30) if i != j and topics[i] == topics[j]
    ]
    diff_topic = [
        cosine[i, j] for i in range(30) for j in range(30) if i != j and topics[i] != topics[j]
    ]
    print(f"Avg cosine similarity within same topic:      {np.mean(same_topic):.3f}")
    print(f"Avg cosine similarity across different topics: {np.mean(diff_topic):.3f}")

    coords_2d = project_to_2d(vectors)

    colors = {"cooking": "tab:orange", "football": "tab:green", "programming": "tab:blue"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for topic, color in colors.items():
        # boolean mask selects just this topic's rows out of the full 2D projection
        mask = [t == topic for t in topics]
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], label=topic, color=color, s=80)

    ax.set_title("30 sentences projected to 2D (UMAP over text-embedding-3-small)")
    ax.legend()
    fig.tight_layout()
    # fig.savefig(OUTPUT_PATH, dpi=150)
    plt.show()
    print(f"Saved plot to {OUTPUT_PATH}")
    print("Expect 3 visually separated clusters — one per topic — with zero")
    print("topic labels ever passed into the embedding model.")

    print("Raw similarity matrices computed:", cosine.shape, dot.shape, euclid.shape)
    print("2D projection saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
