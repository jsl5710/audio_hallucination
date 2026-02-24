#!/usr/bin/env python3
"""
LFM2-Audio-1.5B Hallucination Classification Experiment.

Note: This model is English-only.
Requires: pip install liquid-audio

Usage:
    python scripts/run_lfm2audio.py --data-dir /path/to/Audio_data --batch-size 4
    python scripts/run_lfm2audio.py --data-dir /path/to/Audio_data --languages english
"""

import time
import torch
import json
from typing import Dict, List

from experiment_utils import (
    parse_common_args, main_runner, BaseClassifier,
    PROMPTS, PARSERS,
)

MODEL_NAME = "LiquidAI/LFM2-Audio-1.5B"


def setup_model(model_name: str, use_flash_attn: bool = False):
    """Load LFM2-Audio model and processor."""
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    print(f"Loading model: {model_name}")
    processor = LFM2AudioProcessor.from_pretrained(model_name).eval()
    model = LFM2AudioModel.from_pretrained(model_name).eval()
    print("Model loaded successfully")
    return model, processor


class HallucinationClassifier(BaseClassifier):
    def __init__(self, model_name: str, experiment_type: str, use_flash_attn: bool = False):
        super().__init__(model_name, experiment_type)
        self.model, self.processor = setup_model(model_name, use_flash_attn)

    def _generate_response(self, prompt: str, audio_path: str = None,
                           text_content: str = None) -> str:
        import torchaudio
        from liquid_audio import ChatState

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            chat = ChatState(self.processor)

            # System prompt
            chat.new_turn("system")
            chat.add_text("You are a helpful assistant. Respond with text only.")
            chat.end_turn()

            # User turn
            chat.new_turn("user")
            if audio_path:
                wav, sampling_rate = torchaudio.load(audio_path)
                chat.add_audio(wav, sampling_rate)
            if text_content:
                chat.add_text(text_content)
            chat.add_text(prompt)
            chat.end_turn()

            # Generate
            chat.new_turn("assistant")
            text_tokens = []
            with torch.no_grad():
                for t in self.model.generate_interleaved(
                    **chat, max_new_tokens=256, audio_temperature=1.0, audio_top_k=4
                ):
                    if t.numel() == 1:  # text token
                        text_tokens.append(t)

            if text_tokens:
                response = self.processor.text.decode(torch.stack(text_tokens, 1))
            else:
                response = ""

            del chat
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
            response = self._generate_response(prompt, audio_path=input_data)
        else:
            response = self._generate_response(prompt, text_content=input_data)

        value = PARSERS[task](response)
        return {'value': value, 'raw_response': response}


def main():
    parser = parse_common_args("LFM2-Audio-1.5B Hallucination Classification")
    args = parser.parse_args()
    main_runner(HallucinationClassifier, MODEL_NAME, args)


if __name__ == "__main__":
    main()
