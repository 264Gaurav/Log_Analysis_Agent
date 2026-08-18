from contextlib import redirect_stdout, redirect_stderr
import io

from langchain_community.document_compressors import FlashrankRerank
import os
from multiagent import HybridRetriever
from utils import automation
from dotenv import load_dotenv
from logger import fingerprint, get_logger, invoke_with_logging, log_stage, loggable_text
from retrieval_telemetry import document_details
load_dotenv()
api_key = os.getenv('API_KEY')
logger = get_logger(__name__)

class Nodes:
    @staticmethod
    def retrieve(state):    
        question = state["question"]
        path = state["path"]
        with log_stage(
            logger,
            "retrieve",
            path=path,
            question_length=len(question),
            question_sha256=fingerprint(question),
        ):
            hybrid_retriever_instance = HybridRetriever(path, api_key)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                documents = hybrid_retriever_instance.retrieve_with_diagnostics(question, logger)
            logger.info(
                "stage=retrieve event=documents_returned count=%d question=%r",
                len(documents),
                loggable_text(question),
            )
            for rank, document in enumerate(documents, start=1):
                logger.info(
                    "stage=retrieve event=returned_chunk details=%r",
                    document_details(document, rank=rank),
                )

        return {"documents": documents, "question": question}

    # @staticmethod
    # def rerank(state):
    #     print("NVIDIA--RERANKER")
    #     question = state["question"]
    #     documents = state["documents"]
    #     reranker = NVIDIARerank(
    #         model="nvidia/nv-rerankqa-mistral-4b-v3",
    #         nvidia_api_key=api_key
    #     )
    #     documents = reranker.compress_documents(query=question, documents=documents)
    #     return {"documents": documents, "question": question}
    
    @staticmethod
    def rerank(state):
        question = state["question"]
        documents = state["documents"]

        with log_stage(logger, "rerank", input_documents=len(documents)):
            compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
            reranked_docs = compressor.compress_documents(documents=documents, query=question)
            logger.info("stage=rerank event=documents_count count=%d", len(reranked_docs))
        return {"documents": reranked_docs, "question": question}
    

    @staticmethod
    def generate(state):    
        question = state["question"]
        documents = state["documents"]

        with log_stage(logger, "generate", documents=len(documents)):
            generation = invoke_with_logging(
                logger,
                automation.rag_chain,
                "rag_generation",
                {"context": documents, "question": question},
            )
            logger.info("stage=generate event=response_length characters=%d", len(generation))
        return {"documents": documents, "question": question, "generation": generation}

    @staticmethod
    def grade_documents(state):  
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []  
        with log_stage(logger, "grade_documents", input_documents=len(documents)):
            for index, document in enumerate(documents, start=1):
                score = invoke_with_logging(
                    logger,
                    automation.retrieval_grader,
                    "retrieval_grader",
                    {"question": question, "document": document},
                )
                grade = getattr(score, "binary_score", None) if score is not None else None
                if grade in (None, "yes"):
                    filtered_docs.append(document)
                logger.info("stage=grade_documents event=document index=%d grade=%r", index, grade)
            logger.info("stage=grade_documents event=documents_count count=%d", len(filtered_docs))
        return {"documents": filtered_docs, "question": question}
    @staticmethod
    def transform_query(state):
        question = state["question"]
        documents = state["documents"]

        with log_stage(logger, "transform_query", transform_count=state.get("transform_count", 0)):
            better_question = invoke_with_logging(
                logger,
                automation.question_rewriter,
                "query_rewriter",
                {"question": question},
            )
            logger.info(
                "stage=transform_query event=query_rewritten original_sha256=%s rewritten_sha256=%s rewritten_length=%d",
                fingerprint(question),
                fingerprint(better_question),
                len(better_question),
            )
        return {
            "documents": documents,
            "question": better_question,
            "transform_count": state.get("transform_count", 0) + 1,
        }
