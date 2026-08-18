import bat_ai
import argparse
import hashlib
from pathlib import Path
from logger import fingerprint, get_logger, log_run, log_stage

logger = get_logger(__name__)

def process_input(question, file):
    input_path = Path(file)
    run_details = {
        "path": str(input_path),
        "question_length": len(question),
        "question_sha256": fingerprint(question),
    }
    with log_run(logger, **run_details):
        if not input_path.is_file():
            logger.error("stage=analysis event=input_invalid reason=file_not_found path=%s", input_path)
            raise FileNotFoundError(f"Log file does not exist: {file}")
        try:
            input_size = input_path.stat().st_size
            file_sha256 = _file_fingerprint(input_path)
        except OSError:
            logger.exception("stage=analysis event=input_invalid reason=file_unreadable path=%s", input_path)
            raise
        logger.info(
            "stage=analysis event=input_valid path=%s file_size_bytes=%d file_sha256=%s",
            input_path,
            input_size,
            file_sha256,
        )
        with log_stage(logger, "analysis", path=str(input_path), question_length=len(question)):
            inputs = {"question": question, "path": file, "transform_count": 0}
            final_state = None
            for graph_step, output in enumerate(bat_ai.app.stream(inputs), start=1):
                node_name, final_state = next(iter(output.items()))
                logger.info(
                    "stage=analysis event=graph_step step=%d node=%s keys=%s question_sha256=%s transform_count=%s document_count=%s",
                    graph_step,
                    node_name,
                    list(final_state),
                    fingerprint(final_state.get("question", question)),
                    final_state.get("transform_count", 0),
                    len(final_state.get("documents", [])),
                )

            if not final_state or "generation" not in final_state:
                raise RuntimeError("Analysis completed without generating an answer")
            generation = final_state["generation"]
            print(f"Output: {generation}")
            logger.info("stage=analysis event=response_ready response_length=%d", len(generation))
            return generation


def _file_fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze log file for errors")
    parser.add_argument("log_path", help="Path to the log file")
    parser.add_argument("--question", default="Analyze the log file and find the failure messages from the same", help="Question to ask about the log file")
    args = parser.parse_args()
    resposne = process_input(args.question,args.log_path)
