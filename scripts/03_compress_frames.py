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
from typing import Tuple

from PIL import Image, ImageOps
from tqdm import tqdm


def compress_image(input_path: Path, output_path: Path, target_size: Tuple[int, int] = (1280, 720),
                   quality: int = 75) -> bool:
    """Compress a single image file.
    
    Args:
        input_path (Path): Path to input image
        output_path (Path): Path to output compressed image
        target_size (Tuple[int, int]): Target resolution (width, height)
        quality (int): JPEG quality (1-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image maintaining aspect ratio
            img_resized = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
            
            # Save with specified quality
            img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


def compress_batch(input_dir: Path, output_dir: Path, target_size: Tuple[int, int] = (1280, 720),
                   quality: int = 75, workers: int = None, pattern: str = "*.jpg") -> None:
    """Compress all images in a directory using parallel processing.
    
    Args:
        input_dir (Path): Input directory containing images
        output_dir (Path): Output directory for compressed images
        target_size (Tuple[int, int]): Target resolution (width, height)
        quality (int): JPEG quality (1-100)
        workers (int): Number of worker threads (default: CPU count)
        pattern (str): File pattern to match
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = list(input_dir.glob(pattern))
    total_files = len(image_files)
    
    if total_files == 0:
        logger.warning(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {total_files} image files to compress")
    logger.info(f"Target resolution: {target_size[0]}x{target_size[1]}")
    logger.info(f"JPEG quality: {quality}%")
    
    # Prepare arguments for parallel processing
    def process_image(input_path: Path) -> bool:
        output_path = output_dir / input_path.name
        return compress_image(input_path, output_path, target_size, quality)
    
    # Process images in parallel
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_image, img_path): img_path 
            for img_path in image_files
        }
        
        # Process completed tasks with progress bar
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
                    logger.error(f"Exception processing {img_path}: {e}")
                    failed += 1
                finally:
                    pbar.update(1)
    
    logger.info(f"Compression completed: {successful} successful, {failed} failed")


def main() -> int:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Compress high-resolution frame images for annotation tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compress_frames.py --input ../frames --output ../small_frames
  python compress_frames.py --input ../frames --output ../small_frames --quality 80 --resolution 1920x1080
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory containing high-resolution frames'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for compressed frames'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=str,
        default='1280x720',
        help='Target resolution (format: WIDTHxHEIGHT, default: 1280x720)'
    )
    
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=75,
        choices=range(1, 101),
        help='JPEG quality (1-100, default: 75)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of worker threads (default: CPU count)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.jpg',
        help='File pattern to match (default: *.jpg)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        target_size = (width, height)
    except ValueError:
        logger.error(f"Invalid resolution format: {args.resolution}. Use WIDTHxHEIGHT (e.g., 1280x720)")
        return 1
    
    # Validate input directory
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    if not args.input.is_dir():
        logger.error(f"Input path is not a directory: {args.input}")
        return 1
    
    logger.info(f"Starting frame compression")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    start_time = time.time()
    
    try:
        compress_batch(
            input_dir=args.input,
            output_dir=args.output,
            target_size=target_size,
            quality=args.quality,
            workers=args.workers,
            pattern=args.pattern
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Total execution time: {elapsed:.1f} seconds")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())