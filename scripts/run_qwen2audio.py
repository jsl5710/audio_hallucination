#!/usr/bin/env python3
"""
Qwen2-Audio-7B-Instruct Hallucination Classification Experiment.

Usage:
    python scripts/run_qwen2audio.py --data-dir /path/to/Audio_data --batch-size 4
    python scripts/run_qwen2audio.py --data-dir /path/to/Audio_data --experiment-types audio text
"""

import time
import torch
import json
import librosa
from typing import Dict, List, Tuple

from experiment_utils import (
    parse_common_args, main_runner, BaseClassifier,
    PROMPTS, PARSERS,
)

MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"


def setup_model(model_name: str, use_flash_attn: bool = False):
    """Load Qwen2-Audio model and processor."""
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    print(f"Loading model: {model_name}")
    kwargs = {"torch_dtype": "auto", "device_map": "auto"}

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded successfully")
    return model, processor


class HallucinationClassifier(BaseClassifier):
    def __init__(self, model_name: str, experiment_type: str, use_flash_attn: bool = False):
        super().__init__(model_name, experiment_type)
        self.model, self.processor = setup_model(model_name, use_flash_attn)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _prepare_conversation_audio(self, prompt: str, audio_path: str) -> Tuple[List[Dict], List]:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        audio_data, _ = librosa.load(audio_path, sr=self.sampling_rate)
        return conversation, [audio_data]

    def _prepare_conversation_text(self, prompt: str, text_content: str) -> Tuple[List[Dict], List]:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"{prompt}\n\nText to analyze: {text_content}"},
            ]},
        ]
        return conversation, []

    def _generate_response(self, conversation: List[Dict], audios: List) -> str:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False)

            if audios:
                inputs = self.processor(
                    text=text, audios=audios, return_tensors="pt", padding=True)
            else:
                inputs = self.processor(
                    text=text, return_tensors="pt", padding=True)

            inputs.input_ids = inputs.input_ids.to(self.model.device)
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generate_ids = generate_ids[:, input_len:]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            del generate_ids
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
            conversation, audios = self._prepare_conversation_audio(prompt, input_data)
        else:
            conversation, audios = self._prepare_conversation_text(prompt, input_data)

        response = self._generate_response(conversation, audios)
        value = PARSERS[task](response)
        return {'value': value, 'raw_response': response}


def main():
    parser = parse_common_args("Qwen2-Audio-7B-Instruct Hallucination Classification")
    args = parser.parse_args()
    main_runner(HallucinationClassifier, MODEL_NAME, args)


if __name__ == "__main__":
    main()
