"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

LLM_EVAL:
This script evaluates a conversation using a rubric-based approach. 
It utilizes pre-formatted prompts to elicit LLM responses within allowed values 
and retrieves the probability distribution over the outputs.
"""

import argparse
import json
import yaml
import logging
from openai import OpenAI
import time
from typing import List, Dict, Union
import numpy as np
from config import Config
from utils import CacheManager, EvaluationResult
from pathlib import Path
from datetime import datetime
from llm_call import OpenAILLMCaller, LLMEvaluator

def setup_logging(output_dir: str) -> None:
    """
    Configures logging to write logs to both a file and the console.

    Args:
        output_dir (str): Directory where logs should be saved.
    """

    log_file = Path(output_dir) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main() -> None:
    """
    Main function to evaluate a conversation using an LLM and a predefined rubric.
    """
    parser = argparse.ArgumentParser(description="LLM-RUBRIC Conversation Evaluation")
    parser.add_argument("--rubric_guide", type=str, required=True, help="Path to system evaluation file (YAML).")
    parser.add_argument("--conversation", type=str, required=True, help="Path to conversation file")
    parser.add_argument("--output", type=str, default="/export/fs06/psingh54/LLMRubric/outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-16k", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(args.output)
    logger = logging.getLogger(__name__)

    try:

        # Load configuration
        config = Config(
            model=args.model,
            temperature=args.temperature,
            cache_dir=str(output_dir / "cache")
        )

        # Initialize evaluator and LLM caller
        evaluator = LLMEvaluator(args.rubric_guide)
        llm_caller = OpenAILLMCaller(config, evaluator.rubric, logger)

        # Load the conversation data
        with open(args.conversation, "r") as file:
            conversation_data: Dict[str, List[List[str]]] = json.load(file)

        # Extract and format the conversation
        conversation_pairs = conversation_data.get("conv_1", [])
        formatted_conversation: List[str] = [
            f"User: {user_msg}\nAssistant: {assistant_msg}"
            for user_msg, assistant_msg in conversation_pairs
        ]

        conversation_text = "\n".join(formatted_conversation)

        # Run evaluation
        logger.info("Starting conversation evaluation...")
        result = evaluator.evaluate_conversation(conversation_text, llm_caller)

        # Save results
        output_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save(output_file)
        logger.info(f"Results saved to {output_file}")

        # Print results to console
        print("\nEvaluation Results:")
        print(json.dumps(result.to_dict(), indent=2))

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
    