"""
Compare human annotations with OpenAI vision model predictions.

Samples 100 images from Mumbai, gets AI labels via batch API, and compares
with MSE and summary statistics.

Usage:
    python scripts/12_validate_annotations.py --submit   # Submit batch job
    python scripts/12_validate_annotations.py --check    # Check status / get results
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mumbai"
FRAMES_DIR = PROJECT_ROOT / "data" / "annotation_task" / "frames"
FRAMES_NEW_DIR = PROJECT_ROOT / "data" / "annotation_task" / "frames_new"

SAMPLE_SIZE = 100
RANDOM_SEED = 42
MODEL = "gpt-4o-2024-11-20"

PROMPT = """Count the people visible in this street scene image:
- Number of men walking (pedestrians, not on vehicles)
- Number of women walking (pedestrians, not on vehicles)
- Number of men on two-wheelers (motorcycles, scooters, bicycles)
- Number of women on two-wheelers

Return only JSON: {"men_count": N, "women_count": N, "men_twowheeler": N, "women_twowheeler": N}"""


def build_frame_lookup() -> dict:
    """Build lookup from (video_id, frame_number) to file path."""
    lookup = {}

    for f in FRAMES_DIR.glob("*.jpg"):
        match = re.match(r"(.+)_frame(\d+)\.jpg", f.name)
        if match:
            video_id, frame_num = match.groups()
            lookup[(video_id, int(frame_num))] = f

    for f in FRAMES_NEW_DIR.glob("*.jpg"):
        match = re.match(r"(.+)_frame(\d+)\.jpg", f.name)
        if match:
            video_id, frame_num = match.groups()
            lookup[(video_id, int(frame_num))] = f

    return lookup


def resolve_image_path(row: pd.Series, lookup: dict) -> Path | None:
    """Resolve image path from row data."""
    vid = row["base_video_id"]
    fn = row["frame_number"]
    return lookup.get((vid, fn))


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def sample_images(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Sample images with resolved paths."""
    df = df.copy()
    df["resolved_path"] = df.apply(lambda r: resolve_image_path(r, lookup), axis=1)
    df_with_images = df[df["resolved_path"].notna()].copy()

    sample: pd.DataFrame = df_with_images.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    return sample


def create_batch_input(sample: pd.DataFrame, output_path: Path, model: str) -> None:
    """Create JSONL file for OpenAI batch API."""
    requests = []

    for idx, row in sample.iterrows():
        image_path = row["resolved_path"]
        b64 = encode_image_base64(image_path)

        request = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 100,
            },
        }
        requests.append(request)

    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Created batch input: {output_path} ({len(requests)} requests)")


def submit_batch(client: OpenAI, input_path: Path) -> str:
    """Submit batch job to OpenAI."""
    batch_file = client.files.create(file=open(input_path, "rb"), purpose="batch")

    print(f"Uploaded file: {batch_file.id}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Created batch: {batch.id}")
    print(f"Status: {batch.status}")

    batch_id_path = DATA_DIR / "validation_batch_id.txt"
    with open(batch_id_path, "w") as f:
        f.write(batch.id)

    print(f"Batch ID saved to: {batch_id_path}")
    return batch.id


def check_batch_status(client: OpenAI) -> dict | None:
    """Check batch status and download results if complete."""
    batch_id_path = DATA_DIR / "validation_batch_id.txt"

    if not batch_id_path.exists():
        print("No batch ID found. Run with --submit first.")
        return None

    batch_id = batch_id_path.read_text().strip()
    batch = client.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Request counts: {batch.request_counts}")

    if batch.status == "completed":
        output_file_id = batch.output_file_id
        if output_file_id is None:
            print("Error: No output file ID in completed batch")
            return {"status": "failed"}
        content = client.files.content(output_file_id)

        output_path = DATA_DIR / "validation_batch_output.jsonl"
        with open(output_path, "wb") as f:
            f.write(content.content)

        print(f"Downloaded results to: {output_path}")
        return {"status": "completed", "output_path": output_path}

    elif batch.status == "failed":
        print(f"Batch failed: {batch.errors}")
        return {"status": "failed"}

    else:
        print("Batch still processing. Check again later.")
        return {"status": batch.status}


def parse_ai_response(response_text: str) -> dict | None:
    """Parse JSON from AI response."""
    try:
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return None


def process_results(sample: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Process batch results and compute metrics."""
    results = {}

    with open(output_path) as f:
        for line in f:
            data = json.loads(line)
            custom_id = data["custom_id"]
            response = data.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])

            if choices:
                content = choices[0].get("message", {}).get("content", "")
                parsed = parse_ai_response(content)
                if parsed:
                    results[int(custom_id)] = parsed
                else:
                    print(f"Failed to parse response for {custom_id}: {content}")
                    results[int(custom_id)] = {
                        "men_count": None,
                        "women_count": None,
                        "men_twowheeler": None,
                        "women_twowheeler": None,
                    }

    sample = sample.copy()
    sample["ai_men_count"] = sample.index.map(lambda x: results.get(x, {}).get("men_count"))
    sample["ai_women_count"] = sample.index.map(lambda x: results.get(x, {}).get("women_count"))
    sample["ai_men_twowheeler"] = sample.index.map(
        lambda x: results.get(x, {}).get("men_twowheeler")
    )
    sample["ai_women_twowheeler"] = sample.index.map(
        lambda x: results.get(x, {}).get("women_twowheeler")
    )

    return sample


def compute_metrics(df: pd.DataFrame) -> None:
    """Compute and print validation metrics."""
    fields = ["men_count", "women_count", "men_twowheeler", "women_twowheeler"]

    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)

    for field in fields:
        human_col = field
        ai_col = f"ai_{field}"

        valid = df[[human_col, ai_col]].dropna()
        n = len(valid)

        if n == 0:
            print(f"\n{field}: No valid comparisons")
            continue

        human = valid[human_col].astype(float)
        ai = valid[ai_col].astype(float)

        mse = ((human - ai) ** 2).mean()
        rmse = mse**0.5
        mae = (human - ai).abs().mean()
        mean_diff = (human - ai).mean()
        corr = human.corr(ai)

        human_sum = human.sum()
        ai_sum = ai.sum()
        sum_diff_pct = 100 * (human_sum - ai_sum) / human_sum if human_sum > 0 else 0

        print(f"\n{field} (n={n}):")
        print(f"  MSE:  {mse:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE:  {mae:.3f}")
        print(f"  Mean diff (human - AI): {mean_diff:.3f}")
        print(f"  Correlation: {corr:.3f}")
        print(f"  Sum human: {human_sum:.0f}, Sum AI: {ai_sum:.0f}")
        print(f"  Sum diff: {sum_diff_pct:+.1f}%")

    # Compare ratios only on frames where the AI response parsed, so both
    # totals cover the same sample.
    ai_cols = ["ai_men_count", "ai_women_count", "ai_men_twowheeler", "ai_women_twowheeler"]
    parsed = df.dropna(subset=ai_cols)
    total_human_men = parsed["men_count"].sum() + parsed["men_twowheeler"].sum()
    total_human_women = parsed["women_count"].sum() + parsed["women_twowheeler"].sum()
    total_ai_men = parsed["ai_men_count"].sum() + parsed["ai_men_twowheeler"].sum()
    total_ai_women = parsed["ai_women_count"].sum() + parsed["ai_women_twowheeler"].sum()

    human_ratio = total_human_women / total_human_men if total_human_men > 0 else 0
    ai_ratio = total_ai_women / total_ai_men if total_ai_men > 0 else 0

    print("\n" + "-" * 60)
    print(f"GENDER RATIOS (women / men, {len(parsed)} frames with parsed AI output):")
    print(f"  Human annotations: {human_ratio:.3f}")
    print(f"  AI predictions:    {ai_ratio:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare annotations with OpenAI")
    parser.add_argument("--submit", action="store_true", help="Submit batch job to OpenAI")
    parser.add_argument("--check", action="store_true", help="Check batch status and get results")
    parser.add_argument("--model", default=MODEL, help="Pinned image-capable model ID")
    args = parser.parse_args()

    if not args.submit and not args.check:
        parser.print_help()
        parser.error("one of --submit or --check is required")

    from openai import OpenAI

    client = OpenAI()

    df = pd.read_parquet(DATA_DIR / "analysis_data.parquet")
    print(f"Loaded {len(df)} rows from analysis_data.parquet")

    lookup = build_frame_lookup()
    print(f"Built frame lookup with {len(lookup)} entries")

    sample_path = DATA_DIR / "validation_sample.csv"
    batch_input_path = DATA_DIR / "validation_batch_input.jsonl"
    results_path = DATA_DIR / "validation_results.csv"

    if args.submit:
        sample = sample_images(df, lookup)
        print(f"Sampled {len(sample)} images")

        sample.to_csv(sample_path, index=True)
        print(f"Saved sample to: {sample_path}")

        create_batch_input(sample, batch_input_path, args.model)
        submit_batch(client, batch_input_path)

    elif args.check:
        if not sample_path.exists():
            print(f"Sample file not found: {sample_path}")
            print("Run with --submit first.")
            parser.error("run with --submit before --check")

        sample = pd.read_csv(sample_path, index_col=0)
        sample["resolved_path"] = sample.apply(lambda r: resolve_image_path(r, lookup), axis=1)

        result = check_batch_status(client)

        if result and result.get("status") == "completed":
            output_path = result["output_path"]
            sample_with_ai = process_results(sample, output_path)

            sample_with_ai.to_csv(results_path, index=True)
            print(f"Saved results to: {results_path}")

            compute_metrics(sample_with_ai)


if __name__ == "__main__":
    main()
