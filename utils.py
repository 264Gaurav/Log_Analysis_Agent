from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from binary_score_models import GradeAnswer,GradeDocuments,GradeHallucinations
import os
from dotenv import load_dotenv
from logger import get_logger, log_stage
load_dotenv()
import json

logger = get_logger(__name__)

class Nodeoutputs:
    def __init__(self, api_key, model, prompts_file):
        with log_stage(logger, "automation_setup", model=model, prompts_file=prompts_file):
            os.environ["NVIDIA_API_KEY"] = api_key
            self.llm = ChatNVIDIA(api_key=api_key, model=model)
            self.prompts = self.load_prompts(prompts_file)
            self.setup_prompts()

    def load_prompts(self, prompts_file):
        with open(prompts_file, 'r') as file:
            prompts = json.load(file)
        logger.info("stage=load_prompts event=complete prompt_count=%d", len(prompts))
        return prompts

    def setup_prompts(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["qa_system_prompt"]),
                ("user", self.prompts["qa_user_prompt"])
            ]
        )
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["re_write_system"]),
                ("human", self.prompts["re_write_human"]),
            ]
        )
        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["grade_system"]),
                ("human", self.prompts["grade_human"]),
            ]
        )
        self.retrieval_grader = grade_prompt | self.llm.with_structured_output(GradeDocuments)

        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["hallucination_system"]),
                ("human", self.prompts["hallucination_human"]),
            ]
        )
        self.hallucination_grader = hallucination_prompt | self.llm.with_structured_output(GradeHallucinations)

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["answer_system"]),
                ("human", self.prompts["answer_human"]),
            ]
        )
        self.answer_grader = answer_prompt | self.llm.with_structured_output(GradeAnswer)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

# Usage


# Access the API key from environment variables
api_key = os.getenv('API_KEY')
model = "meta/llama-3.1-70b-instruct"
prompts_file = "prompt.json"
automation = Nodeoutputs(api_key, model, prompts_file)
