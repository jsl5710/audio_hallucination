#!/usr/bin/env python3
"""
Gemma 3n E4B Hallucination Classification Experiment.

Requires: transformers>=4.53.0, HuggingFace access to google/gemma-3n-E4B-it

Usage:
    python scripts/run_gemma3n.py --data-dir /path/to/Audio_data --batch-size 4
    python scripts/run_gemma3n.py --data-dir /path/to/Audio_data --hf-token YOUR_TOKEN
"""

import os
import time
import torch
import json
from typing import Dict, List

from experiment_utils import (
    parse_common_args, main_runner, BaseClassifier,
    PROMPTS, PARSERS,
)

MODEL_NAME = "google/gemma-3n-E4B-it"


def setup_model(model_name: str, use_flash_attn: bool = False):
    """Load Gemma 3n model and processor."""
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    print(f"Loading model: {model_name}")
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Gemma3nForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded successfully")
    return model, processor


class HallucinationClassifier(BaseClassifier):
    def __init__(self, model_name: str, experiment_type: str, use_flash_attn: bool = False):
        super().__init__(model_name, experiment_type)
        self.model, self.processor = setup_model(model_name, use_flash_attn)

    def _prepare_conversation_audio(self, prompt: str, audio_path: str) -> List[Dict]:
        return [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]

    def _prepare_conversation_text(self, prompt: str, text_content: str) -> List[Dict]:
        return [
            {"role": "user", "content": [
                {"type": "text", "text": f"{prompt}\n\nText to analyze: {text_content}"},
            ]},
        ]

    def _generate_response(self, conversation: List[Dict]) -> str:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            inputs = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)

            input_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generated_tokens = output[0][input_len:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)

            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response.strip()

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
            conversation = self._prepare_conversation_audio(prompt, input_data)
        else:
            conversation = self._prepare_conversation_text(prompt, input_data)

        response = self._generate_response(conversation)
        value = PARSERS[task](response)
        return {'value': value, 'raw_response': response}


def main():
    parser = parse_common_args("Gemma 3n E4B Hallucination Classification")
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    # Handle HuggingFace authentication
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    if os.environ.get('HF_TOKEN'):
        from huggingface_hub import login
        login(token=os.environ['HF_TOKEN'])
        print("Logged in to HuggingFace")
    else:
        print("Warning: No HF token provided. Run 'huggingface-cli login' if access is needed.")

    main_runner(HallucinationClassifier, MODEL_NAME, args)


if __name__ == "__main__":
    main()
