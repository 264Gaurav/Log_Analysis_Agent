# from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_community.document_compressors import FlashrankRerank
import os
from multiagent import HybridRetriever
import io
from contextlib import redirect_stdout, redirect_stderr
from utils import automation
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('API_KEY')

class Nodes:
    @staticmethod
    def retrieve(state):    
        print("---RETRIEVE---")
        question = state["question"]
        path = state["path"]
        hybrid_retriever_instance = HybridRetriever(path, api_key)
        hybrid_retriever = hybrid_retriever_instance.get_retriever()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            documents = hybrid_retriever.get_relevant_documents(question)

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
        print("---LOCAL FLASHRANK CROSS-ENCODER---")
        question = state["question"]
        documents = state["documents"]

        # Use the valid FlashRank model name
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
        
        reranked_docs = compressor.compress_documents(documents=documents, query=question)
        return {"documents": reranked_docs, "question": question}
    

    @staticmethod
    def generate(state):    
        print("GENERATE USING LLM")
        question = state["question"]
        documents = state["documents"]

        generation = automation.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    @staticmethod
    def grade_documents(state):  
        print("CHECKING DOCUMENT RELEVANCE TO QUESTION")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []  
        
        for d in documents:
            score = automation.retrieval_grader.invoke({"question": question, "document": d.page_content})
            
            # Check if score is None before accessing binary_score
            if score is not None and getattr(score, "binary_score", None) == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            elif score is None:
                # Fallback if structured output parser returns None
                print("---GRADE: FALLBACK (PARSER RETURNED NONE), KEEPING DOCUMENT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
        # print("CHECKING DOCUMENT RELEVANCE TO QUESTION")
        # question = state["question"]
        # ret_documents = state["documents"]

        # filtered_docs = []
        # for doc in ret_documents:
        #     score = automation.retrieval_grader.invoke(
        #         {"question": question, "document": doc.page_content}
        #     )
        #     grade = score.binary_score
        #     if grade == "yes":
        #         print("---GRADE: DOCUMENT RELEVANT---")
        #         filtered_docs.append(doc)
        #     else:
        #         print("---GRADE: DOCUMENT NOT RELEVANT---")
        # return {"documents": filtered_docs, "question": question}

    @staticmethod
    def transform_query(state):
        
        print("REWRITE PROMPT")
        question = state["question"]
        documents = state["documents"]

        better_question = automation.question_rewriter.invoke({"question": question})
        print(f"actual query : {question} \n Transformed query:{better_question}")
        return {"documents": documents, "question": better_question}
