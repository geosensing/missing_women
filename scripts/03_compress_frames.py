#!/usr/bin/env python3
"""
Compress high-resolution frame images for annotation tasks.

Resizes frames and reduces JPEG quality for faster upload/download
while preserving sufficient detail for human annotation.

Inputs:
    - Input directory: High-resolution JPEG frames

Outputs:
    - Output directory: Compressed JPEG frames at target resolution

Usage:
    python scripts/03_compress_frames.py \\
        -i data/annotation_task/delhi_frames \\
        -o data/annotation_task/delhi_frames_compressed \\
        -r 1280x720 \\
        -q 75
"""

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm


def compress_image(
    input_path: Path,
    output_path: Path,
    target_size: Tuple[int, int] = (1280, 720),
    quality: int = 75,
) -> bool:
    """Compress a single image file.

    Args:
        input_path: Path to input image
        output_path: Path to output compressed image
        target_size: Target resolution (width, height)
        quality: JPEG quality (1-100)

    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(input_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_resized = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
            img_resized.save(output_path, "JPEG", quality=quality, optimize=True)

        return True

    except Exception as e:
        print(f"Error: Failed to process {input_path}: {e}")
        return False


def compress_batch(
    input_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (1280, 720),
    quality: int = 75,
    workers: Optional[int] = None,
    pattern: str = "*.jpg",
) -> Tuple[int, int, int]:
    """Compress all images in a directory using parallel processing.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for compressed images
        target_size: Target resolution (width, height)
        quality: JPEG quality (1-100)
        workers: Number of worker threads (default: CPU count)
        pattern: File pattern to match

    Returns:
        Tuple of (total_files, successful, failed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob(pattern))
    total_files = len(image_files)

    if total_files == 0:
        print(f"Warning: No image files found in {input_dir}")
        return 0, 0, 0

    print(f"Found {total_files} image files to compress")
    print(f"Target resolution: {target_size[0]}x{target_size[1]}")
    print(f"JPEG quality: {quality}%")

    def process_image(input_path: Path) -> bool:
        output_path = output_dir / input_path.name
        return compress_image(input_path, output_path, target_size, quality)

    successful = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(process_image, img_path): img_path
            for img_path in image_files
        }

        with tqdm(total=total_files, desc="Compressing images") as pbar:
            for future in concurrent.futures.as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error: Exception processing {img_path}: {e}")
                    failed += 1
                finally:
                    pbar.update(1)

    return total_files, successful, failed


def main() -> int:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compress high-resolution frame images for annotation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_compress_frames.py -i ../frames -o ../small_frames
  python 03_compress_frames.py -i ../frames -o ../small_frames -q 80 -r 1920x1080
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input directory containing high-resolution frames",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for compressed frames",
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=str,
        default="1280x720",
        help="Target resolution (format: WIDTHxHEIGHT, default: 1280x720)",
    )

    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=75,
        choices=range(1, 101),
        metavar="QUALITY",
        help="JPEG quality (1-100, default: 75)",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: CPU count)",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.jpg",
        help="File pattern to match (default: *.jpg)",
    )

    args = parser.parse_args()

    try:
        width, height = map(int, args.resolution.split("x"))
        target_size = (width, height)
    except ValueError:
        print(
            f"Error: Invalid resolution format: {args.resolution}. "
            "Use WIDTHxHEIGHT (e.g., 1280x720)"
        )
        return 1

    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    if not args.input.is_dir():
        print(f"Error: Input path is not a directory: {args.input}")
        return 1

    print(f"Starting frame compression")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    start_time = time.time()

    try:
        total_files, successful, failed = compress_batch(
            input_dir=args.input,
            output_dir=args.output,
            target_size=target_size,
            quality=args.quality,
            workers=args.workers,
            pattern=args.pattern,
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total images: {total_files}")
        print(f"  Successful: {successful}")
        print(f"  Failed:     {failed}")
        print(f"\nExecution time: {elapsed:.1f} seconds")
        print(f"Output: {args.output}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
