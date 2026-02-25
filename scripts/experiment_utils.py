#!/usr/bin/env python3
"""
Shared utilities for hallucination classification experiments.

Contains prompts, data loading, checkpoint management, experiment runner,
and common CLI argument parsing used by all model scripts.
"""

import argparse
import hashlib
import json
import os
import pickle
import time
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm


# ========================== CONSTANTS ==========================

LANGUAGE_FOLDERS = {
    'english': 'English',
    'kazakh': 'Kazakh',
    'russian': 'Russian',
}

EXPERIMENT_TYPES = ['audio', 'text']
TASKS = ['binary', 'type', 'degree']
APPROACHES = ['direct', 'cot']

# Ground-truth column and valid classes per task (used for stratified sampling)
TASK_CLASSES = {
    'binary': {
        'column': 'hallucination',
        'classes': ['yes', 'no'],
    },
    'type': {
        'column': 'hallucination_type',
        'classes': ['factual_contradiction', 'factual_fabrication', 'contextual_inconsistency'],
    },
    'degree': {
        'column': 'hallucination_level',
        'classes': ['mild', 'moderate', 'severe'],
    },
}

# Prompt structure version (increment when prompt structure changes)
PROMPT_VERSION = 2  # v2 = separated single-task prompts

# ========================== PROMPTS ==========================

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# --- TASK 1: Binary hallucination detection ---

AUDIO_BINARY_DIRECT_PROMPT = """You are an expert assistant specialized in analyzing audio content and detecting hallucinations.

Your task is to determine whether there are any hallucinations in the audio content.

A hallucination is any statement that contains factual contradictions, fabricated details, or contextual inconsistencies relative to the actual audio content.

Given the audio content, determine:
Is there a hallucination? (yes/no)

Output format: {{"binary": "yes/no"}}"""

AUDIO_BINARY_COT_PROMPT = """You are an expert assistant specialized in analyzing audio content and detecting hallucinations.

Your task is to determine whether there are any hallucinations in the audio content.

A hallucination is any statement that contains factual contradictions, fabricated details, or contextual inconsistencies relative to the actual audio content.

Given the audio content, think step-by-step about whether hallucinations are present:

Then provide your final answer:
Is there a hallucination? (yes/no)

Output format: {{"binary": "yes/no"}}"""

# --- TASK 2: Hallucination type classification ---

AUDIO_TYPE_DIRECT_PROMPT = """You are an expert assistant specialized in analyzing audio content and classifying hallucination types.

Your task is to identify the type of hallucination present in the audio content.

Hallucination Types:
- Factual Contradiction: Statements that directly conflict with known facts or information provided in the audio content.
- Factual Fabrication: Insertion of fabricated yet plausible-sounding details not grounded in the audio content.
- Contextual Inconsistency: Subtle alterations that distort the meaning, emphasis, or context of the audio content without introducing explicit factual errors.

Given the audio content, classify the hallucination type:
What type of hallucination is present? (factual_contradiction/factual_fabrication/contextual_inconsistency/none)

Output format: {{"type": "factual_contradiction/factual_fabrication/contextual_inconsistency/none"}}"""

AUDIO_TYPE_COT_PROMPT = """You are an expert assistant specialized in analyzing audio content and classifying hallucination types.

Your task is to identify the type of hallucination present in the audio content.

Hallucination Types:
- Factual Contradiction: Statements that directly conflict with known facts or information provided in the audio content.
- Factual Fabrication: Insertion of fabricated yet plausible-sounding details not grounded in the audio content.
- Contextual Inconsistency: Subtle alterations that distort the meaning, emphasis, or context of the audio content without introducing explicit factual errors.

Given the audio content, think step-by-step about what type of hallucination may be present:

Then provide your final classification:
What type of hallucination is present? (factual_contradiction/factual_fabrication/contextual_inconsistency/none)

Output format: {{"type": "factual_contradiction/factual_fabrication/contextual_inconsistency/none"}}"""

# --- TASK 3: Hallucination degree/severity classification ---

AUDIO_DEGREE_DIRECT_PROMPT = """You are an expert assistant specialized in analyzing audio content and assessing hallucination severity.

Your task is to assess the severity level of any hallucination present in the audio content.

Severity Levels:
- Mild: Subtle distortions or minor deviations that preserve the main narrative and plausibility of the audio content.
- Moderate: Noticeable inconsistencies or factual alterations that affect key details or context while maintaining partial alignment with the audio content.
- Severe: Major contradictions, fabrications, or contextual breakdowns that substantially misrepresent or conflict with the audio content's facts or intent.

Given the audio content, classify the hallucination severity:
What is the severity level? (mild/moderate/severe/none)

Output format: {{"degree": "mild/moderate/severe/none"}}"""

AUDIO_DEGREE_COT_PROMPT = """You are an expert assistant specialized in analyzing audio content and assessing hallucination severity.

Your task is to assess the severity level of any hallucination present in the audio content.

Severity Levels:
- Mild: Subtle distortions or minor deviations that preserve the main narrative and plausibility of the audio content.
- Moderate: Noticeable inconsistencies or factual alterations that affect key details or context while maintaining partial alignment with the audio content.
- Severe: Major contradictions, fabrications, or contextual breakdowns that substantially misrepresent or conflict with the audio content's facts or intent.

Given the audio content, think step-by-step about the severity of any hallucination:

Then provide your final assessment:
What is the severity level? (mild/moderate/severe/none)

Output format: {{"degree": "mild/moderate/severe/none"}}"""

# --- Text prompts (derived from audio prompts) ---

TEXT_BINARY_DIRECT_PROMPT = AUDIO_BINARY_DIRECT_PROMPT.replace("audio content", "text content")
TEXT_BINARY_COT_PROMPT = AUDIO_BINARY_COT_PROMPT.replace("audio content", "text content")
TEXT_TYPE_DIRECT_PROMPT = AUDIO_TYPE_DIRECT_PROMPT.replace("audio content", "text content")
TEXT_TYPE_COT_PROMPT = AUDIO_TYPE_COT_PROMPT.replace("audio content", "text content")
TEXT_DEGREE_DIRECT_PROMPT = AUDIO_DEGREE_DIRECT_PROMPT.replace("audio content", "text content")
TEXT_DEGREE_COT_PROMPT = AUDIO_DEGREE_COT_PROMPT.replace("audio content", "text content")

# Prompt lookup: PROMPTS[modality][task][approach]
PROMPTS = {
    'audio': {
        'binary': {'direct': AUDIO_BINARY_DIRECT_PROMPT, 'cot': AUDIO_BINARY_COT_PROMPT},
        'type':   {'direct': AUDIO_TYPE_DIRECT_PROMPT,   'cot': AUDIO_TYPE_COT_PROMPT},
        'degree': {'direct': AUDIO_DEGREE_DIRECT_PROMPT, 'cot': AUDIO_DEGREE_COT_PROMPT},
    },
    'text': {
        'binary': {'direct': TEXT_BINARY_DIRECT_PROMPT, 'cot': TEXT_BINARY_COT_PROMPT},
        'type':   {'direct': TEXT_TYPE_DIRECT_PROMPT,   'cot': TEXT_TYPE_COT_PROMPT},
        'degree': {'direct': TEXT_DEGREE_DIRECT_PROMPT, 'cot': TEXT_DEGREE_COT_PROMPT},
    }
}


# ========================== CLI ARGUMENT PARSING ==========================

def parse_common_args(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common arguments for all model scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to audio data directory (contains English/, Kazakh/, Russian/ folders)')
    parser.add_argument('--output-dir', type=str, default='hallucination_results',
                        help='Output directory for results (default: hallucination_results)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of samples to process per batch (default: 1)')
    parser.add_argument('--experiment-types', nargs='+', default=['audio'],
                        choices=['audio', 'text'],
                        help='Experiment types to run (default: audio)')
    parser.add_argument('--languages', nargs='+', default=['english', 'kazakh', 'russian'],
                        choices=['english', 'kazakh', 'russian'],
                        help='Languages to process (default: all)')
    parser.add_argument('--force-restart', action='store_true',
                        help='Ignore existing results and start fresh')
    parser.add_argument('--flash-attn', action='store_true',
                        help='Use Flash Attention 2')
    parser.add_argument('--filter-hallucinated-only', action='store_true',
                        help='Only process hallucinated samples')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip smart resume validation')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Number of samples per class for stratified sampling. '
                             'Each task uses its own ground-truth column to define classes. '
                             'If a class has fewer samples than requested, all available are used. '
                             'When omitted, all samples are processed (original behavior).')
    return parser


# ========================== RESPONSE PARSING ==========================

def parse_binary(response: str) -> str:
    """Parse binary (yes/no) from model response."""
    if not response:
        return 'no'
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            result = json.loads(response[start:end])
            binary = result.get('binary', 'no').lower().strip()
            if binary in ('yes', 'no'):
                return binary
    except (json.JSONDecodeError, AttributeError):
        pass
    response_lower = response.lower()
    if any(phrase in response_lower for phrase in
           ['"binary": "yes"', "'binary': 'yes'", 'binary: yes', '"yes"']):
        return 'yes'
    return 'no'


def parse_type(response: str) -> str:
    """Parse hallucination type from model response."""
    if not response:
        return 'none'
    valid_types = {'factual_contradiction', 'factual_fabrication',
                   'contextual_inconsistency', 'none'}
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            result = json.loads(response[start:end])
            type_val = result.get('type', 'none').lower().strip()
            if type_val in valid_types:
                return type_val
    except (json.JSONDecodeError, AttributeError):
        pass
    response_lower = response.lower()
    if 'factual_fabrication' in response_lower:
        return 'factual_fabrication'
    elif 'factual_contradiction' in response_lower:
        return 'factual_contradiction'
    elif 'contextual_inconsistency' in response_lower or 'contextual inconsistency' in response_lower:
        return 'contextual_inconsistency'
    elif 'fabrication' in response_lower:
        return 'factual_fabrication'
    elif 'contradiction' in response_lower:
        return 'factual_contradiction'
    return 'none'


def parse_degree(response: str) -> str:
    """Parse hallucination degree/severity from model response."""
    if not response:
        return 'none'
    valid_degrees = {'mild', 'moderate', 'severe', 'none'}
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            result = json.loads(response[start:end])
            degree = result.get('degree', 'none').lower().strip()
            if degree in valid_degrees:
                return degree
    except (json.JSONDecodeError, AttributeError):
        pass
    response_lower = response.lower()
    if 'severe' in response_lower:
        return 'severe'
    elif 'moderate' in response_lower:
        return 'moderate'
    elif 'mild' in response_lower:
        return 'mild'
    return 'none'


PARSERS = {
    'binary': parse_binary,
    'type': parse_type,
    'degree': parse_degree,
}


# ========================== BASE CLASSIFIER ==========================

class BaseClassifier:
    """Base classifier interface for hallucination classification."""

    def __init__(self, model_name: str, experiment_type: str):
        self.model_name = model_name.split('/')[-1]
        self.experiment_type = experiment_type

    def classify_task(self, input_data: str, task: str, approach: str) -> Dict[str, str]:
        """Classify a single sample for a given task and approach.

        Returns: {'value': str, 'raw_response': str}
        """
        raise NotImplementedError

    def classify_task_batch(self, input_data_list: List[str], task: str,
                            approach: str) -> List[Dict[str, str]]:
        """Classify a batch of samples. Default: sequential fallback."""
        results = []
        for data in input_data_list:
            result = self.classify_task(data, task, approach)
            results.append(result)
            time.sleep(0.1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


# ========================== DATA LOADING ==========================

def load_transcription_data(base_data_dir: str, language_folder: str,
                            language: str,
                            filter_hallucinated_only: bool = False) -> pd.DataFrame:
    """Load transcription CSV from a language folder."""
    folder_path = Path(base_data_dir) / language_folder
    if not folder_path.exists():
        raise FileNotFoundError(f"Language folder not found: {folder_path}")

    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {folder_path}")
    if len(csv_files) > 1:
        print(f"Multiple CSV files found in {folder_path}, using: {csv_files[0]}")

    csv_file = csv_files[0]
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {csv_file}")
    except Exception as e:
        try:
            df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')
            print(f"Loaded {csv_file} (tab-separated)")
        except Exception as e2:
            raise Exception(f"Failed to load {csv_file}: {e}, {e2}")

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    required_columns = ['filename', 'text', 'hallucination',
                        'hallucination_type', 'hallucination_level']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_file}: {missing}")

    if filter_hallucinated_only:
        df = df[df['hallucination'].str.lower() == 'yes'].copy()
        print(f"Filtered data (hallucination=yes): {df.shape}")
    else:
        df = df.copy()
        print(f"Processing all data: {df.shape}")

    df['language'] = language
    df['language_folder'] = language_folder
    df.reset_index(drop=True, inplace=True)
    return df


def load_all_transcription_data(base_data_dir: str,
                                languages: List[str] = None,
                                filter_hallucinated_only: bool = False) -> Dict[str, pd.DataFrame]:
    """Load transcription data for all requested languages."""
    if languages is None:
        languages = list(LANGUAGE_FOLDERS.keys())

    all_data = {}
    for language in languages:
        folder_name = LANGUAGE_FOLDERS.get(language)
        if folder_name is None:
            print(f"Unknown language: {language}")
            continue
        filter_msg = "hallucinated only" if filter_hallucinated_only else "all samples"
        print(f"\nLoading {language} from {folder_name}/ ({filter_msg})...")
        try:
            data = load_transcription_data(base_data_dir, folder_name, language,
                                           filter_hallucinated_only)
            all_data[language] = data
        except Exception as e:
            print(f"Failed to load {language} data: {e}")
    return all_data


# ========================== SMART CHECKPOINT MANAGER ==========================

class SmartCheckpointManager:
    def __init__(self, model_name: str, language: str, experiment_type: str,
                 output_dir: str = 'hallucination_results',
                 checkpoint_dir: str = 'checkpoints'):
        self.model_name = model_name.split('/')[-1]
        self.language = language
        self.experiment_type = experiment_type
        self.output_dir = Path(output_dir) / self.model_name / experiment_type
        self.checkpoint_dir = Path(checkpoint_dir) / self.model_name / experiment_type

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / f"{language}_results.csv"
        self.checkpoint_file = self.checkpoint_dir / f"{language}_checkpoint.pkl"
        self.progress_file = self.checkpoint_dir / f"{language}_progress.json"
        self.sample_mapping_file = self.checkpoint_dir / f"{language}_sample_mapping.pkl"

    def _create_sample_key(self, row: pd.Series) -> str:
        filename = str(row.get('filename', ''))
        text = str(row.get('text', ''))[:100]
        language = str(row.get('language', ''))
        key_string = f"{filename}|{text}|{language}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def _create_sample_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        mapping = {}
        for idx, row in df.iterrows():
            key = self._create_sample_key(row)
            mapping[key] = idx
        return mapping

    def _load_existing_sample_mapping(self) -> Dict[str, Dict]:
        if self.sample_mapping_file.exists():
            try:
                with open(self.sample_mapping_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load sample mapping: {e}")
        return {}

    def _save_sample_mapping(self, mapping: Dict[str, Dict]):
        try:
            with open(self.sample_mapping_file, 'wb') as f:
                pickle.dump(mapping, f)
        except Exception as e:
            print(f"Warning: Failed to save sample mapping: {e}")

    def smart_load_existing_results(self, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[int]]:
        results_df = self._initialize_results_df(current_df)
        processed_indices: Set[int] = set()
        current_mapping = self._create_sample_mapping(current_df)
        print(f"Current dataset has {len(current_mapping)} unique samples")

        if not self._check_prompt_version_compatible():
            print(f"Prompt version changed (now v{PROMPT_VERSION}) - starting fresh")
            return results_df, processed_indices

        if not self.results_file.exists():
            print("No existing results found - starting fresh")
            return results_df, processed_indices

        try:
            existing_df = pd.read_csv(self.results_file, encoding='utf-8')
            print(f"Found existing results with {len(existing_df)} samples")
            matched = 0
            unmatched = 0

            partial = 0
            for _, existing_row in existing_df.iterrows():
                existing_key = self._create_sample_key(existing_row)
                if existing_key in current_mapping:
                    current_idx = current_mapping[existing_key]
                    pred_columns = [c for c in existing_df.columns if c.startswith('pred_')]
                    # Always copy any existing predictions (supports partial results
                    # from stratified sampling where not all tasks are filled)
                    has_any = False
                    for col in pred_columns:
                        if col in existing_df.columns:
                            val = existing_row[col]
                            if not pd.isna(val) and str(val).strip() not in ('', 'nan'):
                                results_df.loc[current_idx, col] = val
                                has_any = True
                    if self._check_sample_processed(existing_row, pred_columns):
                        processed_indices.add(current_idx)
                        matched += 1
                    elif has_any:
                        partial += 1
                    else:
                        unmatched += 1
                else:
                    unmatched += 1

            print(f"Matched {matched} fully processed, {partial} partially processed, "
                  f"{unmatched} unmatched")

            updated_mapping = {}
            for idx in processed_indices:
                row = results_df.iloc[idx]
                key = self._create_sample_key(row)
                pred_data = {c: results_df.loc[idx, c]
                             for c in results_df.columns if c.startswith('pred_')}
                updated_mapping[key] = {'index': idx, 'predictions': pred_data,
                                        'processed': True}
            self._save_sample_mapping(updated_mapping)

        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Starting fresh...")

        return results_df, processed_indices

    def _check_prompt_version_compatible(self) -> bool:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                return progress.get('prompt_version', 1) == PROMPT_VERSION
            except Exception:
                pass
        return True

    def _check_sample_processed(self, row: pd.Series, pred_columns: List[str]) -> bool:
        required = ['pred_direct_binary', 'pred_direct_type', 'pred_direct_degree',
                     'pred_cot_binary', 'pred_cot_type', 'pred_cot_degree']
        for col in required:
            if col not in pred_columns:
                return False
            value = row.get(col, '')
            if pd.isna(value) or str(value).strip() == '' or str(value).lower() == 'nan':
                return False
        return True

    def save_checkpoint(self, df: pd.DataFrame, processed_indices: set, current_idx: int):
        checkpoint_data = {
            'processed_indices': processed_indices,
            'current_idx': current_idx,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'experiment_type': self.experiment_type,
            'smart_resume': True,
            'prompt_version': PROMPT_VERSION,
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        progress_info = {
            'model': self.model_name,
            'language': self.language,
            'experiment_type': self.experiment_type,
            'processed_count': len(processed_indices),
            'total_count': len(df),
            'progress_percentage': (len(processed_indices) / len(df) * 100) if len(df) > 0 else 0,
            'last_updated': datetime.now().isoformat(),
            'smart_resume_enabled': True,
            'prompt_version': PROMPT_VERSION,
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_info, f, indent=2)

        df.to_csv(self.results_file, index=False, encoding='utf-8')

        sample_mapping = {}
        for idx in processed_indices:
            if idx < len(df):
                key = self._create_sample_key(df.iloc[idx])
                pred_data = {c: df.loc[idx, c]
                             for c in df.columns if c.startswith('pred_')}
                sample_mapping[key] = {'index': idx, 'predictions': pred_data,
                                       'processed': True}
        self._save_sample_mapping(sample_mapping)

    def _initialize_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
        results_df = df.copy()
        for approach in APPROACHES:
            for task in TASKS:
                for suffix in ['', '_raw']:
                    col = f'pred_{approach}_{task}{suffix}'
                    if col not in results_df.columns:
                        results_df[col] = ''
        return results_df

    def cleanup_checkpoint(self):
        for fp in [self.checkpoint_file, self.progress_file, self.sample_mapping_file]:
            try:
                if fp.exists():
                    fp.unlink()
            except Exception as e:
                print(f"Failed to cleanup {fp}: {e}")
        print(f"Cleaned up checkpoint files for {self.language} ({self.experiment_type})")

    def get_smart_resume_stats(self, current_df: pd.DataFrame) -> Dict:
        current_mapping = self._create_sample_mapping(current_df)
        existing_mapping = self._load_existing_sample_mapping()
        return {
            'current_dataset_size': len(current_df),
            'current_unique_samples': len(current_mapping),
            'existing_processed_samples': len(existing_mapping),
            'potentially_matchable': len(set(current_mapping.keys()) & set(existing_mapping.keys())),
        }


# ========================== STRATIFIED SAMPLING ==========================

def stratified_sample_indices(df: pd.DataFrame, task: str,
                              samples_per_class: int) -> List[int]:
    """Select indices via stratified sampling for a given task.

    For each valid class of the task, sorts the class's rows by filename
    (for deterministic, incremental selection) and takes the first N.
    If a class has fewer than N rows, all are taken and a warning is printed.

    Returns a sorted list of DataFrame integer indices.
    """
    task_info = TASK_CLASSES[task]
    column = task_info['column']
    valid_classes = task_info['classes']

    selected_indices: List[int] = []
    for cls in valid_classes:
        # Select rows belonging to this class (case-insensitive)
        mask = df[column].str.lower().str.strip() == cls.lower()
        class_df = df[mask].sort_values('filename')

        available = len(class_df)
        take = min(samples_per_class, available)
        if available < samples_per_class:
            print(f"  Warning: {task}/{cls} has only {available} samples "
                  f"(requested {samples_per_class})")

        # Take first N indices (deterministic — increasing N keeps prior selection)
        selected_indices.extend(class_df.index[:take].tolist())

    selected_indices.sort()
    return selected_indices


def check_task_completed(results_df: pd.DataFrame, indices: List[int],
                         task: str) -> Set[int]:
    """Return the subset of indices where both approaches are filled for a task."""
    done = set()
    cols = [f'pred_{approach}_{task}' for approach in APPROACHES]
    for idx in indices:
        if all(_value_is_filled(results_df.loc[idx, c]) for c in cols):
            done.add(idx)
    return done


def _value_is_filled(value) -> bool:
    """Check if a prediction cell has a real value."""
    if pd.isna(value):
        return False
    s = str(value).strip()
    return s != '' and s.lower() != 'nan'


# ========================== EXPERIMENT RUNNER ==========================

def _prepare_input(row: pd.Series, experiment_type: str,
                   base_data_dir: str) -> Optional[str]:
    """Prepare input data for a single sample. Returns None if invalid."""
    if experiment_type == 'audio':
        audio_filename = row['filename']
        language_folder = row['language_folder']
        audio_path = os.path.join(base_data_dir, language_folder, audio_filename)
        if not os.path.exists(audio_path):
            print(f"\nAudio file not found: {audio_path}")
            return None
        return audio_path
    else:
        text = row.get('text', '')
        if pd.isna(text) or str(text).strip() == '':
            return None
        return str(text)


def run_experiment(classifier: BaseClassifier, model_name: str,
                   data_dict: Dict[str, pd.DataFrame], experiment_type: str,
                   batch_size: int = 1, base_data_dir: str = '',
                   output_dir: str = 'hallucination_results',
                   checkpoint_dir: str = 'checkpoints',
                   force_restart: bool = False,
                   samples_per_class: Optional[int] = None):
    """Run classification experiment with smart checkpointing and batch support.

    When samples_per_class is set, uses stratified sampling: each task gets its
    own subset of N samples per ground-truth class. The loop becomes task-first
    so only the needed task is run on each subset.  When None, all samples are
    processed for all tasks (original behaviour).
    """

    print(f"\nStarting {experiment_type} experiment with {model_name}")
    print(f"Prompt version: v{PROMPT_VERSION} | Batch size: {batch_size}")
    if samples_per_class is not None:
        print(f"Stratified sampling: {samples_per_class} samples per class per task")
    print(f"Tasks: {TASKS} x Approaches: {APPROACHES} = {len(TASKS) * len(APPROACHES)} calls per sample")
    print("=" * 60)

    print_gpu_memory_info()

    for language, df in data_dict.items():
        print(f"\nProcessing {language} ({len(df)} samples) - {experiment_type} mode")

        ckpt_mgr = SmartCheckpointManager(model_name, language, experiment_type,
                                          output_dir, checkpoint_dir)

        stats = ckpt_mgr.get_smart_resume_stats(df)
        print(f"Dataset: {stats['current_dataset_size']} samples, "
              f"{stats['existing_processed_samples']} previously processed, "
              f"{stats['potentially_matchable']} matchable")

        if force_restart:
            print("Force restart - ignoring existing results")
            results_df = ckpt_mgr._initialize_results_df(df)
            processed_indices: Set[int] = set()
        else:
            results_df, processed_indices = ckpt_mgr.smart_load_existing_results(df)

        print(f"Resuming: {len(processed_indices)}/{len(df)} already processed")

        if samples_per_class is not None:
            _run_stratified(classifier, df, results_df, processed_indices,
                            ckpt_mgr, experiment_type, base_data_dir,
                            batch_size, language, samples_per_class)
        else:
            _run_all_samples(classifier, df, results_df, processed_indices,
                             ckpt_mgr, experiment_type, base_data_dir,
                             batch_size, language)

        # Final save
        ckpt_mgr.save_checkpoint(results_df, processed_indices, len(df) - 1)

        # Generate summary
        summary = generate_summary_stats(results_df, language,
                                         model_name.split('/')[-1],
                                         experiment_type, samples_per_class)
        summary_file = ckpt_mgr.output_dir / f"{language}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nCompleted {language}")
        print(f"Results: {ckpt_mgr.results_file}")
        print(f"Summary: {summary_file}")

        if samples_per_class is None and len(processed_indices) == len(df):
            ckpt_mgr.cleanup_checkpoint()


# --------------- original (all-samples) loop ---------------

def _run_all_samples(classifier, df, results_df, processed_indices,
                     ckpt_mgr, experiment_type, base_data_dir,
                     batch_size, language):
    """Original loop: every sample gets all 6 classifications."""

    if len(processed_indices) == len(df):
        print(f"All samples for {language} already processed!")
        return

    unprocessed = [idx for idx in range(len(df)) if idx not in processed_indices]

    pbar = tqdm(total=len(df), initial=len(processed_indices),
                desc=f"{language} ({experiment_type})", unit="samples")

    try:
        for batch_start in range(0, len(unprocessed), batch_size):
            batch_indices = unprocessed[batch_start:batch_start + batch_size]

            batch_inputs = []
            valid_indices = []
            for idx in batch_indices:
                row = df.iloc[idx]
                input_data = _prepare_input(row, experiment_type, base_data_dir)
                if input_data is not None:
                    batch_inputs.append(input_data)
                    valid_indices.append(idx)

            if not batch_inputs:
                pbar.update(len(batch_indices))
                continue

            if batch_start % (batch_size * 50) == 0 and torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_res = torch.cuda.memory_reserved() / 1024**3
                print(f"\nGPU Memory: {mem_alloc:.1f}GB allocated, {mem_res:.1f}GB reserved")

            for approach in APPROACHES:
                for task in TASKS:
                    results = classifier.classify_task_batch(batch_inputs, task, approach)
                    for i, idx in enumerate(valid_indices):
                        results_df.loc[idx, f'pred_{approach}_{task}'] = results[i]['value']
                        results_df.loc[idx, f'pred_{approach}_{task}_raw'] = results[i]['raw_response']

            processed_indices.update(valid_indices)
            pbar.update(len(batch_indices))

            ckpt_mgr.save_checkpoint(results_df, processed_indices, batch_indices[-1])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving progress...")
        ckpt_mgr.save_checkpoint(results_df, processed_indices,
                                 batch_indices[-1] if 'batch_indices' in locals() else 0)
        print("Progress saved. Resume by running the script again.")
        return

    finally:
        pbar.close()

    print(f"  {language}: {len(processed_indices)}/{len(df)} samples processed")


# --------------- stratified-sampling loop ---------------

def _run_stratified(classifier, df, results_df, processed_indices,
                    ckpt_mgr, experiment_type, base_data_dir,
                    batch_size, language, samples_per_class):
    """Task-first loop: each task gets its own stratified subset."""

    for task in TASKS:
        target_indices = stratified_sample_indices(df, task, samples_per_class)
        done = check_task_completed(results_df, target_indices, task)
        remaining = [idx for idx in target_indices if idx not in done]

        print(f"\n  Task '{task}': {len(target_indices)} target samples, "
              f"{len(done)} already done, {len(remaining)} to process")

        if not remaining:
            continue

        pbar = tqdm(total=len(target_indices), initial=len(done),
                    desc=f"{language}/{task}", unit="samples")

        try:
            for batch_start in range(0, len(remaining), batch_size):
                batch_indices = remaining[batch_start:batch_start + batch_size]

                batch_inputs = []
                valid_indices = []
                for idx in batch_indices:
                    row = df.iloc[idx]
                    input_data = _prepare_input(row, experiment_type, base_data_dir)
                    if input_data is not None:
                        batch_inputs.append(input_data)
                        valid_indices.append(idx)

                if not batch_inputs:
                    pbar.update(len(batch_indices))
                    continue

                if batch_start % (batch_size * 50) == 0 and torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_res = torch.cuda.memory_reserved() / 1024**3
                    print(f"\nGPU Memory: {mem_alloc:.1f}GB allocated, {mem_res:.1f}GB reserved")

                # Only run the current task (both approaches)
                for approach in APPROACHES:
                    results = classifier.classify_task_batch(batch_inputs, task, approach)
                    for i, idx in enumerate(valid_indices):
                        results_df.loc[idx, f'pred_{approach}_{task}'] = results[i]['value']
                        results_df.loc[idx, f'pred_{approach}_{task}_raw'] = results[i]['raw_response']

                # Track fully-processed samples (all 6 columns filled)
                for idx in valid_indices:
                    if all(_value_is_filled(results_df.loc[idx, f'pred_{a}_{t}'])
                           for a in APPROACHES for t in TASKS):
                        processed_indices.add(idx)

                pbar.update(len(batch_indices))

                ckpt_mgr.save_checkpoint(results_df, processed_indices, batch_indices[-1])

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print(f"\nInterrupted! Saving progress...")
            ckpt_mgr.save_checkpoint(results_df, processed_indices,
                                     batch_indices[-1] if 'batch_indices' in locals() else 0)
            print("Progress saved. Resume by running the script again.")
            return

        finally:
            pbar.close()

        print(f"  Task '{task}' complete for {language}")


def generate_summary_stats(df: pd.DataFrame, language: str, model_name: str,
                           experiment_type: str,
                           samples_per_class: Optional[int] = None) -> Dict:
    summary = {
        'language': language,
        'model': model_name,
        'experiment_type': experiment_type,
        'total_samples': len(df),
        'prompt_version': PROMPT_VERSION,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if samples_per_class is not None:
        summary['samples_per_class'] = samples_per_class
        # Record per-task sample counts
        per_task = {}
        for task in TASKS:
            indices = stratified_sample_indices(df, task, samples_per_class)
            completed = check_task_completed(df, indices, task)
            per_task[task] = {
                'target_samples': len(indices),
                'completed_samples': len(completed),
            }
        summary['stratified_sampling'] = per_task
    for approach in APPROACHES:
        binary_col = f'pred_{approach}_binary'
        if binary_col in df.columns:
            binary_yes = (df[binary_col] == 'yes').sum()
            type_col = f'pred_{approach}_type'
            degree_col = f'pred_{approach}_degree'
            summary[approach] = {
                'binary_yes_predictions': int(binary_yes),
                'binary_accuracy': float(binary_yes / len(df)) if len(df) > 0 else 0,
                'type_distribution': df[type_col].value_counts().to_dict() if type_col in df.columns else {},
                'degree_distribution': df[degree_col].value_counts().to_dict() if degree_col in df.columns else {},
            }
    return summary


# ========================== UTILITIES ==========================

def print_gpu_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        print(f"GPU Memory: {allocated:.1f}GB/{total:.1f}GB used "
              f"({reserved:.1f}GB reserved, {free:.1f}GB free)")
    else:
        print("CUDA not available")


def show_progress_summary(checkpoint_dir: str = 'checkpoints'):
    print("EXPERIMENT PROGRESS SUMMARY")
    print("=" * 60)
    checkpoint_base = Path(checkpoint_dir)
    if not checkpoint_base.exists():
        print("No experiments in progress.")
        return
    for model_dir in sorted(checkpoint_base.iterdir()):
        if not model_dir.is_dir():
            continue
        print(f"\nModel: {model_dir.name}")
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            print(f"  {exp_dir.name}:")
            for pf in sorted(exp_dir.glob("*_progress.json")):
                try:
                    with open(pf) as f:
                        p = json.load(f)
                    lang = p.get('language', '?')
                    done = p.get('processed_count', 0)
                    total = p.get('total_count', 0)
                    pct = p.get('progress_percentage', 0)
                    print(f"    {lang:10} {done:4d}/{total:4d} ({pct:5.1f}%)")
                except Exception as e:
                    print(f"    Error: {e}")


def merge_all_results(output_dir: str = 'hallucination_results'):
    for model_dir in Path(output_dir).iterdir():
        if not model_dir.is_dir():
            continue
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            dfs = [pd.read_csv(f, encoding='utf-8')
                   for f in exp_dir.glob("*_results.csv")]
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                out = exp_dir / "all_languages_results.csv"
                merged.to_csv(out, index=False, encoding='utf-8')
                print(f"Merged: {out}")


# ========================== MAIN RUNNER ==========================

def main_runner(classifier_class, model_name: str, args):
    """Shared main logic for all model scripts.

    Args:
        classifier_class: Class that takes (model_name, experiment_type, use_flash_attn)
        model_name: Full model name (e.g. "Qwen/Qwen2.5-Omni-3B")
        args: Parsed CLI arguments
    """
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    samples_per_class = getattr(args, 'samples_per_class', None)

    print(f"Hallucination Classification Experiment")
    print(f"Model: {model_name}")
    print(f"Prompt version: v{PROMPT_VERSION}")
    print(f"Batch size: {args.batch_size}")
    if samples_per_class is not None:
        print(f"Stratified sampling: {samples_per_class} samples per class")
    print("=" * 60)

    valid_types = [t for t in args.experiment_types if t in EXPERIMENT_TYPES]
    if not valid_types:
        print(f"Invalid experiment types. Valid options: {EXPERIMENT_TYPES}")
        return

    print(f"Experiment types: {valid_types}")
    filter_msg = "hallucinated only" if args.filter_hallucinated_only else "all samples"
    print(f"Data mode: {filter_msg}")

    print("\nLoading transcription data...")
    data_dict = load_all_transcription_data(
        args.data_dir, args.languages, args.filter_hallucinated_only)

    if not data_dict:
        print("No data loaded. Check --data-dir path.")
        return

    total = sum(len(df) for df in data_dict.values())
    print(f"\nTotal samples: {total}")
    for lang, df in data_dict.items():
        if 'hallucination' in df.columns:
            yes_ct = (df['hallucination'].str.lower() == 'yes').sum()
            no_ct = (df['hallucination'].str.lower() == 'no').sum()
            print(f"  {lang}: {len(df)} (hallucinated: {yes_ct}, non-hallucinated: {no_ct})")
        else:
            print(f"  {lang}: {len(df)}")

    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    for experiment_type in valid_types:
        print(f"\n{'='*60}")
        print(f"Initializing classifier for {experiment_type} experiment...")
        classifier = classifier_class(model_name, experiment_type, args.flash_attn)
        print_gpu_memory_info()

        try:
            run_experiment(
                classifier=classifier,
                model_name=model_name,
                data_dict=data_dict,
                experiment_type=experiment_type,
                batch_size=args.batch_size,
                base_data_dir=args.data_dir,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                force_restart=args.force_restart,
                samples_per_class=samples_per_class,
            )
        except Exception as e:
            print(f"Failed {experiment_type} experiment: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll experiments completed! Results in: {args.output_dir}")
