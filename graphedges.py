from utils import automation
from logger import get_logger, invoke_with_logging, log_stage

logger = get_logger(__name__)

class Edge:
    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Returns:
            str: Binary decision for next node to call
        """

        filtered_documents = state["documents"]
        transform_count = state.get("transform_count", 0)

        # If we've transformed too many times, force generation
        if transform_count >= 2:
            logger.info("stage=route_documents decision=generate reason=max_transforms transform_count=%d", transform_count)
            return "generate"

        if not filtered_documents:
            logger.info("stage=route_documents decision=transform_query reason=no_relevant_documents")
            return "transform_query"
        logger.info("stage=route_documents decision=generate documents=%d", len(filtered_documents))
        return "generate"
        
    def grade_generation_vs_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Returns:
            str: Decision for next node to call
        """

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        try:
            with log_stage(logger, "grade_generation", documents=len(documents)):
                score_text = invoke_with_logging(
                    logger,
                    automation.answer_grader,
                    "answer_grader",
                    {"question": question, "generation": generation},
                )
                grade = getattr(score_text, "binary_score", str(score_text)).lower()
                if grade == "yes":
                    logger.info("stage=route_generation decision=useful grade=%s", grade)
                    return "useful"
                transform_count = state.get("transform_count", 0)
                if transform_count >= 2:
                    logger.info("stage=route_generation decision=useful reason=max_transforms grade=%s", grade)
                    return "useful"
                logger.info("stage=route_generation decision=transform_query grade=%s", grade)
                return "not useful"
        except Exception:
            logger.exception("stage=route_generation event=grading_failed decision=useful")
            return "useful"