import os
from typing import Any
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter #chunking technique (we can use other type of chunking technique like semantic, agentic chunking etc.)
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import EnsembleRetriever #for RRF(Reciprocal Rank fusion of BM25 + Faiss retrieved docs/chunks)
from langchain_community.retrievers import BM25Retriever #Keyword + freq. based similarity (sparse indexing)
from langchain_community.vectorstores.faiss import FAISS #Semantic similarity (vector indexing)
from dotenv import load_dotenv
from logger import get_logger, log_stage
from retrieval_telemetry import document_id, log_component_results, log_fused_results, rrf_fuse

load_dotenv()
api_key = os.getenv('API_KEY')
logger = get_logger(__name__)

# when the class -> HybridRetriever will be initialised then following steps executed in sequence:
#   file_path setup/configured, environment setup, embedding initialisation, chunking, BM25 + FAISS retriever initialised, Hybrid_retriever initialised
class HybridRetriever:
    def __init__(self, file_path, api_key):
        self.file_path = file_path
        self.api_key = api_key
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "4"))
        self.rrf_weights = (0.5, 0.5)
        self.rrf_rank_constant = int(os.getenv("RRF_RANK_CONSTANT", "60"))
        self.faiss_score_threshold = float(os.getenv("FAISS_SCORE_THRESHOLD", "0.8"))
        with log_stage(logger, "retriever_setup", path=file_path):
            if not self.api_key:
                logger.error("stage=retriever_setup event=configuration_invalid reason=missing_api_key")
                raise ValueError("API_KEY is required to initialize NVIDIA embeddings")
            os.environ["NVIDIA_API_KEY"] = api_key
            self.embeddings = self.initialize_nvidia_components()
            self.doc_splits = self.load_and_split_documents()
            self.bm25_retriever, self.faiss_retriever = self.create_retrievers()
            self.hybrid_retriever = self.create_hybrid_retriever()
            logger.info("stage=retriever_setup event=documents_count count=%d", len(self.doc_splits))

    def initialize_nvidia_components(self):
        model = "nvidia/nv-embedqa-e5-v5"
        embeddings = NVIDIAEmbeddings(model=model, nvidia_api_key=self.api_key)
        logger.info("stage=embeddings event=initialized model=%s provider=nvidia", model)
        return  embeddings

    def load_and_split_documents(self):
        loader = TextLoader(self.file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)
        logger.info("stage=load_documents event=complete source_documents=%d chunks=%d", len(docs), len(doc_splits))
        return doc_splits

    def create_retrievers(self):
        bm25_retriever = BM25Retriever.from_documents(self.doc_splits)
        faiss_vectorstore = FAISS.from_documents(self.doc_splits, self.embeddings)
        self.faiss_vectorstore = faiss_vectorstore
        faiss_retriever = faiss_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": self.faiss_score_threshold, "k": self.retrieval_k},
        )
        logger.info(
            "stage=create_retrievers event=complete chunks=%d bm25_k=%d faiss_k=%d faiss_score_threshold=%.3f rrf_weights=%s rrf_rank_constant=%d",
            len(self.doc_splits),
            self.retrieval_k,
            self.retrieval_k,
            self.faiss_score_threshold,
            self.rrf_weights,
            self.rrf_rank_constant,
        )
        return bm25_retriever, faiss_retriever

#Reciprocal Rank fusion with weightage of -> 50% , 50%
    def create_hybrid_retriever(self):
        hybrid_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.faiss_retriever], weights=[0.5, 0.5])
        return hybrid_retriever

    def get_retriever(self):
        return self.hybrid_retriever

    def retrieve_with_diagnostics(self, question: str, diagnostic_logger: Any):
        """Retrieve with observable BM25, FAISS, and weighted RRF contributions."""
        with log_stage(
            diagnostic_logger,
            "retrieve_components",
            question_length=len(question),
            top_k=self.retrieval_k,
        ):
            bm25_documents = self.bm25_retriever.invoke(question)
            bm25_documents = bm25_documents[: self.retrieval_k]
            bm25_scores_by_id = self._bm25_scores(question)
            bm25_scores = [bm25_scores_by_id.get(self._document_id(document)) for document in bm25_documents]
            log_component_results(diagnostic_logger, "bm25", bm25_documents, bm25_scores)

            faiss_matches = self.faiss_vectorstore.similarity_search_with_relevance_scores(
                question,
                k=self.retrieval_k,
                score_threshold=self.faiss_score_threshold,
            )
            faiss_documents = [document for document, _score in faiss_matches]
            faiss_scores = [score for _document, score in faiss_matches]
            log_component_results(diagnostic_logger, "faiss", faiss_documents, faiss_scores)

            fused = rrf_fuse(
                [("bm25", bm25_documents, bm25_scores), ("faiss", faiss_documents, faiss_scores)],
                self.rrf_weights,
                self.rrf_rank_constant,
            )[: self.retrieval_k]
            log_fused_results(diagnostic_logger, fused)
            diagnostic_logger.info(
                "stage=retrieve event=rrf_complete bm25_count=%d faiss_count=%d fused_count=%d",
                len(bm25_documents),
                len(faiss_documents),
                len(fused),
            )
            return [item.document for item in fused]

    @staticmethod
    def _document_id(document) -> str:
        return document_id(document)

    def _bm25_scores(self, question: str) -> dict[str, float]:
        """Read BM25's native scores for the same query used by its retriever."""
        tokens = self.bm25_retriever.preprocess_func(question)
        scores = self.bm25_retriever.vectorizer.get_scores(tokens)
        return {
            self._document_id(document): float(score)
            for document, score in zip(self.doc_splits, scores)
        }


