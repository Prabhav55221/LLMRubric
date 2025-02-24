"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

MAIN:

Implements and argparse script to allow command line calling of the Calibrator and Evaluator.
"""

# System Imports
import os
import sys
import json
import yaml
import time
import logging
import argparse
from tqdm import tqdm 
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union

# Internal Imports
from src.config import Config
from src.utils import CacheManager, EvaluationResult
from src.llm_call import OpenAILLMCaller, LLMEvaluator
from src.calibrate import calibrate

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

def main():
    """
    Main function to evaluate a conversation using an LLM and a predefined rubric.
    """
    
    parser = argparse.ArgumentParser(description="LLM-RUBRIC Conversation Evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Dataset YAML. Ensure it follows structure shown in ReadMe!")
    parser.add_argument("--output", type=str, default="/export/fs06/psingh54/LLMRubric/outputs", help="Output directory for logs and intermediate outputs.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(args.output)
    logger = logging.getLogger(__name__)

    try:

        # Load configuration
        # TODO: Implement calibration configs once implemented.
        config = Config(
            cache_dir=str(output_dir / "cache")
        )

        # Setup Paths
        dataset_yaml = args.dataset
        with open(dataset_yaml, "r") as f:
            dataset_main = yaml.safe_load(f)

        # Vars
        experiment_name = dataset_main['NAME']
        rubric_guide = dataset_main['SYSTEM_PATH']
        dataset_csv = dataset_main['CSV_PATH']
        judges = dataset_main['JUDGE_IDS']
        dimensions = dataset_main['DIMENSIONS']

        logger.info(f"=== EXPERIMENT NAME: {experiment_name} ===")
        logger.info(f"Stage 0: Starting LLM Evaluation for {experiment_name} with {len(judges.keys())}.")

        # Initialize evaluator and LLM caller
        evaluator = LLMEvaluator(rubric_guide, logger)
        llm_caller = OpenAILLMCaller(config, evaluator.rubric, logger)

        # Load Dataset
        df = pd.read_csv(dataset_csv)
        if 'TEXT' not in list(df.columns):
            logger.error("Please ensure CSV followes defined structure. No TEXT column found.")
            sys.exit()

        text_units = list(df['TEXT'].values)
        logger.info(f"Evaluation will be for {len(text_units)} text units.")

        all_results = []
        for i in tqdm(text_units):
            unit = '\n' + i
            results = evaluator.evaluate_conversation(unit, llm_caller)
            all_results.extend(results)

        # Save results
        output_file = output_dir / f"llm_results/RESULTS_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        EvaluationResult().save(output_file, [r.to_dict() for r in all_results])
        logger.info(f"LLM Probabilites saved to {output_file}")

        original_size = len(text_units)
        expanded_size = len(all_results)
        logger.info(f"Dataset expanded from {original_size} to {expanded_size} examples")

        logger.info(f"========")
        logger.info(f"\nStage 1: Starting calibration with network!")

        metrics = calibrate(dataset_csv, str(output_file), dataset_yaml, logger)
        save_path = output_dir / f"calibration_results/METRICS_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # Save Evaluation
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"=== END OF EXPERIMENT ===")

    except Exception as e:
        logger.error(f"Error during llm evaluation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    