#!/usr/bin/env python3
"""CLI script to run hallucination detection on audio files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Detect hallucinations in audio transcriptions"
    )
    parser.add_argument("audio_path", type=str, help="Path to audio file or directory")
    parser.add_argument(
        "--transcript", type=str, default=None, help="Reference transcript for comparison"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for hallucination detection (default: 0.5)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Detection methods to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON format)",
    )
    args = parser.parse_args()

    from audio_hallucination.detection import detect_hallucinations

    results = detect_hallucinations(
        audio_path=args.audio_path,
        transcript=args.transcript,
        confidence_threshold=args.threshold,
        methods=args.methods,
    )

    output = [
        {
            "segment_start": r.segment_start,
            "segment_end": r.segment_end,
            "text": r.text,
            "confidence": r.confidence,
            "is_hallucination": r.is_hallucination,
            "method": r.method,
        }
        for r in results
    ]

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    hallucination_count = sum(1 for r in results if r.is_hallucination)
    print(f"\nFound {hallucination_count} hallucination(s) in {len(results)} segment(s).")


if __name__ == "__main__":
    main()
