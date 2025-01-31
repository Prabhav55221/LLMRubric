'''
Author: Prabhav Singh
Implentation Of: Eisner et al. (https://aclanthology.org/2024.acl-long.745v2.pdf)
LLM_EVAL: Code to you use preformated prompts to elicit LLMs output over allowed values
and then getting probability distribution over the outputs.
'''

import argparse
import json
import yaml
import logging
from openai import OpenAI
import time
import json
import logging
from typing import List, Optional, Dict, Union
import numpy as np
from config import Config
from utils import CacheManager, EvaluationResult
from pathlib import Path
from datetime import datetime
from llm_call import OpenAILLMCaller, LLMEvaluator

def setup_logging(output_dir: str):
    log_file = Path(output_dir) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='LLM-RUBRIC Conversation Evaluation')
    parser.add_argument('--rubric', type=str, required=True, help='Path to rubric JSON/YAML file')
    parser.add_argument('--conversation', type=str, required=True, help='Path to conversation file')
    parser.add_argument('--output', type=str, default='/export/fs06/psingh54/LLMRubric/outputs', help='Output directory')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-16k", help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(args.output)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config(
            model=args.model,
            temperature=args.temperature,
            cache_dir=str(output_dir / "cache")
        )
        
        # Initialize components
        evaluator = LLMEvaluator(args.rubric)
        llm_caller = OpenAILLMCaller(config)

        with open(args.conversation, 'r') as file:
            conversation_data = json.load(file)

        conversation = conversation_data.get("conv_1", [])
        formatted_conversation = []
        
        for user_message, assistant_message in conversation:
            formatted_conversation.append(f"User: {user_message}")
            formatted_conversation.append(f"Assistant: {assistant_message}")
        
        conversation = "\n".join(formatted_conversation)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        result = evaluator.evaluate_conversation(conversation, llm_caller)
        
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
    