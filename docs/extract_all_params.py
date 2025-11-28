#!/usr/bin/env python
"""Extract all parameters for each AITraining command"""

import sys


sys.path.insert(0, "../src")

from autotrain.cli.utils import get_field_info
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


commands = {
    "llm": LLMTrainingParams,
    "text-classification": TextClassificationParams,
    "text-regression": TextRegressionParams,
    "token-classification": TokenClassificationParams,
    "seq2seq": Seq2SeqParams,
    "sentence-transformers": SentenceTransformersParams,
    "image-classification": ImageClassificationParams,
    "image-regression": ImageRegressionParams,
    "object-detection": ObjectDetectionParams,
    "extractive-qa": ExtractiveQuestionAnsweringParams,
    "vlm": VLMTrainingParams,
    "tabular": TabularParams,
}

for cmd_name, param_class in commands.items():
    print(f"\n{'='*60}")
    print(f"Command: aitraining {cmd_name}")
    print(f"{'='*60}")

    params = get_field_info(param_class)
    print(f"Total parameters: {len(params)}")
    print("\nParameters:")

    # Group parameters by category
    required_params = []
    optional_params = []

    for param in params:
        if param.get("required", False):
            required_params.append(param)
        else:
            optional_params.append(param)

    if required_params:
        print("\nREQUIRED:")
        for param in sorted(required_params, key=lambda x: x["arg"]):
            print(f"  {param['arg']:40} {param.get('help', 'No description')[:60]}")

    if optional_params:
        print("\nOPTIONAL:")
        for param in sorted(optional_params, key=lambda x: x["arg"]):
            help_text = param.get("help", "No description")[:60]
            default = param.get("default")
            if default is not None and default != "":
                print(f"  {param['arg']:40} {help_text} (default: {default})")
            else:
                print(f"  {param['arg']:40} {help_text}")
