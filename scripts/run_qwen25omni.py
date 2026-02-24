#!/usr/bin/env python3
"""
Qwen2.5-Omni-3B Hallucination Classification Experiment.

Usage:
    python scripts/run_qwen25omni.py --data-dir /path/to/Audio_data --batch-size 4
    python scripts/run_qwen25omni.py --data-dir /path/to/Audio_data --experiment-types audio text
    python scripts/run_qwen25omni.py --data-dir /path/to/Audio_data --force-restart --flash-attn
"""

import time
import torch
import json
from typing import Dict, List

from experiment_utils import (
    parse_common_args, main_runner, BaseClassifier,
    PROMPTS, SYSTEM_PROMPT, PARSERS,
)

MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"


def setup_model(model_name: str, use_flash_attn: bool = False):
    """Load Qwen2.5-Omni model and processor."""
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    print(f"Loading model: {model_name}")
    kwargs = {"torch_dtype": "auto", "device_map": "auto"}

    if use_flash_attn:
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kwargs)
            print("Flash Attention 2 loaded")
        except Exception as e:
            print(f"Flash Attention 2 failed: {e}, falling back to standard")
            kwargs.pop("attn_implementation", None)
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kwargs)
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kwargs)

    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model.disable_talker()
    print("Model loaded successfully")
    return model, processor


class HallucinationClassifier(BaseClassifier):
    def __init__(self, model_name: str, experiment_type: str, use_flash_attn: bool = False):
        super().__init__(model_name, experiment_type)
        self.model, self.processor = setup_model(model_name, use_flash_attn)

    def _prepare_conversation_audio(self, prompt: str, audio_path: str) -> List[Dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": audio_path},
            ]},
        ]

    def _prepare_conversation_text(self, prompt: str, text_content: str) -> List[Dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": f"Text to analyze: {text_content}"},
            ]},
        ]

    def _generate_response(self, conversation: List[Dict]) -> str:
        from qwen_omni_utils import process_mm_info

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=False)

            inputs = self.processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            with torch.no_grad():
                text_ids = self.model.generate(
                    **inputs, return_audio=False, max_new_tokens=256,
                    do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            response = self.processor.batch_decode(
                text_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            del text_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            return response

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
    parser = parse_common_args("Qwen2.5-Omni-3B Hallucination Classification")
    args = parser.parse_args()
    main_runner(HallucinationClassifier, MODEL_NAME, args)


if __name__ == "__main__":
    main()
