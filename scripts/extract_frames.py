import os
import argparse
import cv2
import hashlib
import pandas as pd
import subprocess
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob
import piexif
import json
import concurrent.futures
from tqdm import tqdm  # Regular tqdm instead of notebook version
import numpy as np
import time
from functools import partial

def convert_log_to_csv(log_file_path, csv_file_path):
    """Convert metadata log file to CSV format.
    
    Args:
        log_file_path (str or Path): Path to the metadata log file.
        csv_file_path (str or Path): Path for output CSV file.
    
    Returns:
        int: Number of frames processed.
    """
    log_file_path = Path(log_file_path)
    csv_file_path = Path(csv_file_path)
    
    if not log_file_path.exists():
        print(f"Log file not found: {log_file_path}")
        return 0
    
    data = []
    with open(log_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 10:  # Ensure we have all fields
                data.append({
                    'video_id': parts[0],
                    'source_folder': parts[1],
                    'video_name': parts[2],
                    'original_video_filename': parts[3],
                    'frame_filename': parts[4],
                    'frame_number': int(parts[5]),
                    'frame_timestamp_sec': float(parts[6]),
                    'frame_timestamp_hhmmss': parts[7],
                    'video_fps': float(parts[8]),
                    'video_duration_sec': float(parts[9])
                })
    
    df = pd.DataFrame(data)
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    
    print(f"Converted {len(data)} log entries to CSV: {csv_file_path}")
    return len(data)

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration for the application.
    
    Args:
        verbose (bool): If True, set logging level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('frame_extraction.log', mode='a')
        ]
    )

def extract_frames_optimized(video_path, output_folder, frame_interval, log_file_path):
    """Extract frames from a video file at specified intervals.
    
    Args:
        video_path (str or Path): Path to the input video file.
        output_folder (str or Path): Directory where extracted frames will be saved.
        frame_interval (int): Extract every Nth frame from the video.
        log_file_path (str or Path): Path to the log file for storing frame metadata.
    
    Returns:
        tuple: A tuple containing:
            - int: Number of frames extracted.
            - bool: True if video was processed, False if skipped (already processed).
    
    Notes:
        - Frames are named with pattern: {folder}_{video_name}_frame{number}_t{timestamp}.jpg
        - Metadata logged to file for post-processing into CSV.
        - Skips videos that already exist in the log file.
    """
    # Convert paths to Path objects
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    log_file_path = Path(log_file_path) if log_file_path else None
    
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get video file information
    video_name = video_path.stem
    source_folder = video_path.parent.name
    
    # Generate globally unique video identifier using path hash
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    current_video_id = f"{source_folder}_{video_name}_{path_hash}"
    
    # Check if this video has already been processed in the log file
    if log_file_path and log_file_path.exists():
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
                if current_video_id in log_content:
                    return 0, False
        except Exception as e:
            logger.warning(f"Error reading log file: {str(e)}. Will proceed with processing.")
    
    # Open the video file
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return pd.DataFrame(), False
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate which frames to extract
    frames_to_extract = list(range(0, frame_count, frame_interval))
    
    # Setup metadata logging with unique logger name to avoid conflicts
    logger_name = f'frame_metadata_{id(video_path)}'
    metadata_logger = logging.getLogger(logger_name)
    metadata_logger.handlers.clear()  # Clear any existing handlers
    metadata_handler = logging.FileHandler(log_file_path, mode='a')
    metadata_handler.setFormatter(logging.Formatter('%(message)s'))
    metadata_logger.addHandler(metadata_handler)
    metadata_logger.setLevel(logging.INFO)
    metadata_logger.propagate = False
    
    # Process video frames more efficiently
    frames_extracted = 0
    
    # Set JPEG compression parameters for faster writing
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    
    for frame_idx in frames_to_extract:
        # Seek to the specific frame directly instead of reading all frames
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        
        if not success:
            continue
        
        # Calculate time in seconds
        frame_time_seconds = frame_idx / fps
        
        # Format time as HH:MM:SS.ms
        time_obj = timedelta(seconds=frame_time_seconds)
        hours, remainder = divmod(time_obj.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(time_obj.microseconds / 1000)
        
        # Format time string for filename (HHMMSS_ms)
        time_str_filename = f"{hours:02d}{minutes:02d}{seconds:02d}_{milliseconds:03d}"
        
        # Format time string for display (HH:MM:SS.ms)
        time_str_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        # Format frame number with leading zeros
        frame_num = str(frame_idx).zfill(len(str(frame_count)))
        
        # Create filename with folder prefix for global uniqueness
        filename = f"{source_folder}_{video_name}_frame{frame_num}_t{time_str_filename}.jpg"
        filepath = output_folder / filename
        
        # Save the frame with optimized parameters
        cv2.imwrite(str(filepath), frame, encode_params)
        
        # We'll just use the filename for logging (no relative paths needed)
        
        # Log frame metadata in pipe-delimited format
        log_entry = f"{current_video_id}|{source_folder}|{video_name}|{video_path.name}|{filename}|{frame_idx}|{frame_time_seconds}|{time_str_display}|{fps}|{duration}"
        metadata_logger.info(log_entry)
        frames_extracted += 1
    
    video.release()
    
    # Close metadata logger handler to ensure all data is written
    metadata_handler.close()
    metadata_logger.removeHandler(metadata_handler)
    
    return frames_extracted, True

def process_video_folder_parallel(input_folder, output_folder, frame_interval, log_file_path, 
                                  video_extensions=['.mp4', '.avi', '.mov', '.mkv'], 
                                  max_workers=None):
    """Process all videos in a folder using parallel processing.
    
    Args:
        input_folder (str or Path): Root folder containing videos.
        output_folder (str or Path): Directory to save extracted frames.
        frame_interval (int): Extract every Nth frame from videos.
        log_file_path (str or Path): Path to the log file for metadata.
        video_extensions (list): List of video file extensions to process.
            Defaults to ['.mp4', '.avi', '.mov', '.mkv'].
        max_workers (int, optional): Maximum number of parallel workers.
            If None, uses CPU count.
    
    Returns:
        tuple: A tuple containing:
            - int: Number of videos successfully processed.
            - int: Number of videos skipped (already processed).
    
    Notes:
        - Maintains folder structure from input to output.
        - Skips macOS metadata files (._*).
        - Shows real-time progress with tqdm.
    """
    # Convert to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    log_file_path = Path(log_file_path)
    
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Ensure the directory for the log file exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize log file if it doesn't exist (will be created automatically during logging)
    
    # Find all video files in the directory and subdirectories
    all_files = []
    
    def process_extension(ext: str) -> list:
        """Process a single file extension and return matching files."""
        ext_lower = ext.lower()
        match ext_lower:
            case '.mp4' | '.mov' | '.avi' | '.mkv' | '.m4v':
                # Common video formats
                files = list(input_folder.rglob(f'*{ext}'))
                files.extend(input_folder.rglob(f'*{ext.upper()}'))
                return files
            case _:
                # Generic extension handling for other formats
                files = list(input_folder.rglob(f'*{ext}'))
                files.extend(input_folder.rglob(f'*{ext.upper()}'))
                return files
    
    # Process each extension
    for ext in video_extensions:
        all_files.extend(process_extension(ext))
    
    # Filter out macOS metadata files and remove duplicates
    all_files = list(set([f for f in all_files if not f.name.startswith('._')]))
    
    # Sort files for consistent processing order
    all_files.sort()
    
    print(f"Found {len(all_files)} video files")
    
    # Prepare args for each file - use flat structure for consistency with EXIF
    file_infos = []
    for video_path in all_files:
        # Use flat structure: all frames go directly to output folder
        file_infos.append((video_path, output_folder))
    
    # Process videos in parallel
    processed_count = 0
    skipped_count = 0
    
    # Create a partial function with fixed parameters
    process_func = partial(
        _process_single_video, 
        frame_interval=frame_interval, 
        log_file_path=log_file_path
    )
    
    # Use ThreadPoolExecutor for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get future objects
        future_to_file = {
            executor.submit(process_func, video_path, output_dir): (video_path, i) 
            for i, (video_path, output_dir) in enumerate(file_infos)
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(file_infos), 
                          desc="Processing Videos"):
            video_path, idx = future_to_file[future]
            try:
                frames_extracted, was_processed = future.result()
                if was_processed:
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
    
    # Generate detailed statistics
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Videos processed: {processed_count}")
    print(f"Videos skipped: {skipped_count} (already processed)")
    print(f"Total videos found: {processed_count + skipped_count}")
    
    # Check log file for frame statistics
    if log_file_path.exists():
        try:
            with open(log_file_path, 'r') as f:
                log_lines = f.readlines()
            
            print(f"\nFrame Statistics:")
            print(f"  Total frames extracted: {len(log_lines)}")
            if processed_count > 0:
                print(f"  Average frames per video: {len(log_lines)/processed_count:.1f}")
            
            # Show log file info
            log_size = log_file_path.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"\nLog file: {log_file_path}")
            print(f"Log size: {log_size:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not read log statistics: {e}")
    
    print("=" * 60)
    return processed_count, skipped_count

def _process_single_video(video_path, output_dir, frame_interval, log_file_path):
    """Helper function for parallel video processing.
    
    Args:
        video_path (Path): Path to the video file to process.
        output_dir (Path): Directory to save extracted frames.
        frame_interval (int): Extract every Nth frame.
        log_file_path (Path): Path to the log file.
    
    Returns:
        tuple: (frames_extracted, was_processed)
    """
    try:
        frames_extracted, was_processed = extract_frames_optimized(
            video_path, 
            output_dir, 
            frame_interval,
            log_file_path
        )
        return (frames_extracted, was_processed)
    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")
        return (0, False)

def main():
    """Main function for command-line interface.
    
    Parses command-line arguments and processes videos in parallel.
    Provides comprehensive progress reporting and final statistics.
    """
    parser = argparse.ArgumentParser(
        description='Extract frames from videos in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/videos -o /path/to/frames -c metadata.csv --interval 333
  %(prog)s -i videos/ -o frames/ -c data.csv --workers 4 --verbose
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input folder containing videos')
    parser.add_argument('--output', '-o', required=True, help='Output folder for extracted frames')
    parser.add_argument('--interval', '-n', type=int, default=30, help='Extract every Nth frame')
    parser.add_argument('--log', '-l', required=True, help='Path for metadata log file')
    parser.add_argument('--workers', '-w', type=int, default=None, 
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--extensions', '-e', default='.mp4,.avi,.mov,.mkv',
                        help='Comma-separated list of video extensions to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    # Setup logging based on verbose flag
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Starting frame extraction with {args.workers or 'CPU count'} workers")
    logger.info(f"Input: {args.input}, Output: {args.output}")
    logger.debug(f"Frame interval: {args.interval}, Extensions: {args.extensions}")
    
    # Split extensions string into a list
    video_extensions = args.extensions.split(',')
    
    start_time = time.time()
    
    try:
        # Process videos in parallel
        processed, skipped = process_video_folder_parallel(
            args.input, 
            args.output, 
            args.interval, 
            args.log, 
            video_extensions=video_extensions,
            max_workers=args.workers
        )
        
        elapsed = time.time() - start_time
        
        # Final summary
        print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        if processed > 0:
            print(f"Average time per video: {elapsed/processed:.1f} seconds")
            
        logger.info(f"Extraction completed successfully. Processed: {processed}, Skipped: {skipped}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        
        # Handle different error types with match/case
        match error_type:
            case 'FileNotFoundError':
                logger.error(f"Input directory not found: {args.input}")
                logger.error("Please check if the path exists and is accessible")
                return 1
            case 'PermissionError':
                logger.error(f"Permission denied accessing files")
                logger.error("Please check file/directory permissions")
                return 1
            case 'KeyboardInterrupt':
                logger.info(f"Process interrupted by user after {elapsed:.1f} seconds")
                return 130  # Standard exit code for SIGINT
            case 'MemoryError':
                logger.error("Insufficient memory to process videos")
                logger.error("Try reducing the number of workers or processing fewer videos at once")
                return 1
            case _:
                logger.error(f"Unexpected error ({error_type}): {str(e)}")
                logger.debug("Full traceback:", exc_info=True)
                return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())