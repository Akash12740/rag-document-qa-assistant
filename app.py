import argparse
from dataclasses import dataclass
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    text: str


class LocalRAGPipeline:
    """Lightweight FAISS-backed Retrieval-Augmented Generation pipeline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None

    @staticmethod
    def chunk_text(text: str, doc_id: str, chunk_size: int = 350, overlap: int = 60) -> list[DocumentChunk]:
        words = text.split()
        chunks: list[DocumentChunk] = []
        step = max(1, chunk_size - overlap)
        for idx, start in enumerate(range(0, len(words), step)):
            segment = " ".join(words[start : start + chunk_size])
            if segment:
                chunks.append(DocumentChunk(doc_id=doc_id, chunk_index=idx, text=segment))
        return chunks

    def ingest(self, documents: dict[str, str]) -> None:
        self.chunks = []
        for doc_id, content in documents.items():
            self.chunks.extend(self.chunk_text(content, doc_id=doc_id))

        embeddings = self.encoder.encode([c.text for c in self.chunks], normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.25) -> list[tuple[DocumentChunk, float]]:
        if self.index is None or not self.chunks:
            return []
        q_vec = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.asarray(q_vec, dtype="float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and float(score) >= min_score:
                results.append((self.chunks[int(idx)], float(score)))
        return results

    def build_grounded_prompt(self, query: str, top_k: int = 3) -> str:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return "INSUFFICIENT_CONTEXT: No relevant source passages exceeded the similarity threshold."

        context_blocks = [
            f"[{chunk.doc_id} | chunk {chunk.chunk_index} | score={score:.3f}]\n{chunk.text}"
            for chunk, score in retrieved
        ]
        context_str = "\n\n".join(context_blocks)
        return (
            "System: Answer strictly using the verified context below. Cite [doc_id | chunk] for every claim. "
            "If the answer is not present, state that the context does not contain enough information.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\nGrounded Answer:"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Local FAISS RAG Retrieval Demo")
    parser.add_argument("--query", type=str, default="How is hallucination mitigated in the pipeline?")
    args = parser.parse_args()

    sample_docs = {
        "llm_safety_guide.txt": (
            "Instruction fine-tuning aligns large language models with human intent and safety boundaries. "
            "To mitigate hallucinations during knowledge-intensive tasks, Retrieval-Augmented Generation (RAG) "
            "grounds model responses in retrieved vector database passages with explicit citation constraints "
            "and minimum cosine similarity thresholds."
        ),
        "mlops_deployment.txt": (
            "Containerizing embedding services with Docker and orchestrating inference workloads on Kubernetes "
            "ensures low-latency FAISS vector lookups and automated CI/CD evaluation via GitHub Actions."
        ),
    }

    rag = LocalRAGPipeline()
    rag.ingest(sample_docs)
    print(rag.build_grounded_prompt(args.query))
