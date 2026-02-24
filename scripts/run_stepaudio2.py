#!/usr/bin/env python3
"""
Step-Audio-2-mini Hallucination Classification Experiment.

Requires: transformers==4.49.0, cloned Step-Audio2 repository.
    git clone https://github.com/stepfun-ai/Step-Audio2.git
    pip install transformers==4.49.0 torchaudio librosa onnxruntime s3tokenizer diffusers hyperpyyaml accelerate soundfile

Usage:
    python scripts/run_stepaudio2.py --data-dir /path/to/Audio_data --step-audio-repo /path/to/Step-Audio2
    python scripts/run_stepaudio2.py --data-dir /path/to/Audio_data --step-audio-repo /path/to/Step-Audio2 --batch-size 4
"""

import os
import sys
import time
import torch
import json
from typing import Dict, List

from experiment_utils import (
    parse_common_args, main_runner, BaseClassifier,
    PROMPTS, PARSERS,
)

MODEL_NAME = "stepfun-ai/Step-Audio-2-mini"


def setup_model(model_name: str, use_flash_attn: bool = False):
    """Load Step-Audio-2 model."""
    from stepaudio2 import StepAudio2

    print(f"Loading model: {model_name}")
    model = StepAudio2(model_name)
    print("Model loaded successfully")
    return model, None


class HallucinationClassifier(BaseClassifier):
    def __init__(self, model_name: str, experiment_type: str, use_flash_attn: bool = False):
        super().__init__(model_name, experiment_type)
        self.model, _ = setup_model(model_name, use_flash_attn)

    def _generate_response(self, prompt: str, audio_path: str = None,
                           text_content: str = None) -> str:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]

            if audio_path:
                content = [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ]
                messages.append({"role": "human", "content": content})
            elif text_content:
                messages.append({
                    "role": "human",
                    "content": f"{prompt}\n\nText to analyze: {text_content}",
                })
            else:
                messages.append({"role": "human", "content": prompt})

            # Assistant turn with None triggers generation
            messages.append({"role": "assistant", "content": None})

            _, text, _ = self.model(
                messages, max_new_tokens=256, do_sample=False, temperature=0.0)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return text.strip() if text else ""

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(2)
            return ""
        except Exception as e:
            print(f"Error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ""

    def classify_task(self, input_data: str, task: str, approach: str) -> Dict[str, str]:
        prompt = PROMPTS[self.experiment_type][task][approach]

        if self.experiment_type == 'audio':
            response = self._generate_response(prompt, audio_path=input_data)
        else:
            response = self._generate_response(prompt, text_content=input_data)

        value = PARSERS[task](response)
        return {'value': value, 'raw_response': response}


def main():
    parser = parse_common_args("Step-Audio-2-mini Hallucination Classification")
    parser.add_argument('--step-audio-repo', type=str, required=True,
                        help='Path to cloned Step-Audio2 repository')
    args = parser.parse_args()

    # Add Step-Audio2 repo to Python path
    repo_path = os.path.abspath(args.step_audio_repo)
    if not os.path.isdir(repo_path):
        print(f"Error: Step-Audio2 repo not found at {repo_path}")
        print("Clone it: git clone https://github.com/stepfun-ai/Step-Audio2.git")
        sys.exit(1)
    sys.path.insert(0, repo_path)
    print(f"Added {repo_path} to Python path")

    main_runner(HallucinationClassifier, MODEL_NAME, args)


if __name__ == "__main__":
    main()
