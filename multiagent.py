import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter #chunking technique (we can use other type of chunking technique like semantic, agentic chunking etc.)
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import EnsembleRetriever #for RRF(Reciprocal Rank fusion of BM25 + Faiss retrieved docs/chunks)
from langchain_community.retrievers import BM25Retriever #Keyword + freq. based similarity (sparse indexing)
from langchain_community.vectorstores.faiss import FAISS #Semantic similarity (vector indexing)
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')

# when the class -> HybridRetriever will be initialised then following steps executed in sequence:
#   file_path setup/configured, environment setup, embedding initialisation, chunking, BM25 + FAISS retriever initialised, Hybrid_retriever initialised
class HybridRetriever:
    def __init__(self, file_path, api_key):
        self.file_path = file_path
        os.environ["NVIDIA_API_KEY"] = api_key
        self.embeddings = self.initialize_nvidia_components()
        self.doc_splits = self.load_and_split_documents()
        self.bm25_retriever, self.faiss_retriever = self.create_retrievers()
        self.hybrid_retriever = self.create_hybrid_retriever()

    def initialize_nvidia_components(self):
        embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", nvidia_api_key=api_key)
        return  embeddings

    def load_and_split_documents(self):
        loader = TextLoader(self.file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)
        return doc_splits

    def create_retrievers(self):
        bm25_retriever = BM25Retriever.from_documents(self.doc_splits)
        faiss_vectorstore = FAISS.from_documents(self.doc_splits, self.embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.8})
        return bm25_retriever, faiss_retriever

#Reciprocal Rank fusion with weightage of -> 50% , 50%
    def create_hybrid_retriever(self):
        hybrid_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.faiss_retriever], weights=[0.5, 0.5])
        return hybrid_retriever

    def get_retriever(self):
        return self.hybrid_retriever


