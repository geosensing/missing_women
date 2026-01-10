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

def extract_frames_optimized(video_path, output_folder, frame_interval, master_csv_path):
    """Extract frames from a video file at specified intervals.
    
    Args:
        video_path (str or Path): Path to the input video file.
        output_folder (str or Path): Directory where extracted frames will be saved.
        frame_interval (int): Extract every Nth frame from the video.
        master_csv_path (str or Path): Path to the CSV file for storing frame metadata.
    
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with extracted frame information.
            - bool: True if video was processed, False if skipped (already processed).
    
    Notes:
        - Frames are named with pattern: {folder}_{video_name}_frame{number}_t{timestamp}.jpg
        - CSV contains relative paths for portability.
        - Skips videos that already exist in the master CSV.
    """
    # Convert paths to Path objects
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    master_csv_path = Path(master_csv_path) if master_csv_path else None
    
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get video file information
    video_name = video_path.stem
    
    # Generate globally unique video identifier using path hash
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    
    # Check if this video has already been processed in the master CSV
    if master_csv_path and master_csv_path.exists():
        try:
            existing_df = pd.read_csv(master_csv_path)
            if 'video_name' in existing_df.columns and video_name in existing_df['video_name'].values:
                return pd.DataFrame(), False
        except Exception as e:
            logger.warning(f"Error reading master CSV: {str(e)}. Will proceed with processing.")
    
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
    
    # Process video frames more efficiently
    frame_data = []
    
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
        
        # Extract folder name from video path for unique naming
        source_folder = video_path.parent.name
        
        # Create filename with folder prefix for global uniqueness
        filename = f"{source_folder}_{video_name}_frame{frame_num}_t{time_str_filename}.jpg"
        filepath = output_folder / filename
        
        # Save the frame with optimized parameters
        cv2.imwrite(str(filepath), frame, encode_params)
        
        # Calculate relative paths from CSV location
        csv_dir = master_csv_path.parent if master_csv_path else Path.cwd()
        
        # Handle cases where paths cannot be relative (e.g., external drives)
        try:
            relative_frame_path = filepath.relative_to(csv_dir)
        except ValueError:
            # If can't make relative, use path relative to current directory
            relative_frame_path = filepath.relative_to(Path.cwd()) if filepath.is_relative_to(Path.cwd()) else filepath
            
        try:
            relative_source_path = video_path.relative_to(csv_dir)
        except ValueError:
            # For external drives, just use the filename or a simplified path
            relative_source_path = Path(source_folder) / video_path.name
        
        # Store frame info with consistent column names
        frame_info = {
            # Core Identity (consistent across EXIF and frame CSVs)
            'video_id': f"{source_folder}_{video_name}_{path_hash}",
            'source_folder': source_folder,
            'video_name': video_name,
            'original_video_filename': video_path.name,
            'unique_video_filename': f"{source_folder}_{video_path.name}_{path_hash}",
            
            # Video Properties (consistent with EXIF CSV)
            'video_fps': fps,
            'video_duration_sec': duration,
            
            # Frame-specific Properties
            'frame_filename': filename,
            'frame_file_path': str(relative_frame_path),
            'frame_number': frame_idx,
            'frame_timestamp_sec': frame_time_seconds,
            'frame_timestamp_hhmmss': time_str_display
        }
        
        # Add frame info to our list
        frame_data.append(frame_info)
    
    video.release()
    
    # Create DataFrame from frame data
    new_frame_df = pd.DataFrame(frame_data)
    
    # Update master CSV with thread/process safe approach
    if not new_frame_df.empty and master_csv_path:
        # Use a lock file to prevent race conditions
        lock_file = master_csv_path.with_suffix('.csv.lock')
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Try to create a lock file
                if not lock_file.exists():
                    lock_file.write_text(str(os.getpid()))
                    
                    try:
                        if master_csv_path.exists():
                            existing_df = pd.read_csv(master_csv_path)
                            # Fix FutureWarning: filter out empty DataFrames before concat
                            if not new_frame_df.empty:
                                updated_df = pd.concat([existing_df, new_frame_df], ignore_index=True)
                                updated_df.to_csv(master_csv_path, index=False)
                        else:
                            new_frame_df.to_csv(master_csv_path, index=False)
                        
                        # Success - remove lock and break
                        lock_file.unlink()
                        break
                    except Exception as e:
                        # If an error occurred, remove the lock and retry
                        if lock_file.exists():
                            lock_file.unlink()
                        raise e
                else:
                    # Lock exists, wait and retry
                    time.sleep(0.5)
                    retry_count += 1
            except Exception as e:
                logger.error(f"CSV update error: {str(e)}. Retry {retry_count+1}/{max_retries}")
                retry_count += 1
                time.sleep(1)
                
        # If we couldn't update the master CSV after all retries
        if retry_count >= max_retries:
            logger.warning(f"Could not update master CSV after {max_retries} attempts")
    
    return new_frame_df, True

def process_video_folder_parallel(input_folder, output_folder, frame_interval, master_csv_path, 
                                  video_extensions=['.mp4', '.avi', '.mov', '.mkv'], 
                                  max_workers=None):
    """Process all videos in a folder using parallel processing.
    
    Args:
        input_folder (str or Path): Root folder containing videos.
        output_folder (str or Path): Directory to save extracted frames.
        frame_interval (int): Extract every Nth frame from videos.
        master_csv_path (str or Path): Path to the master CSV file for metadata.
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
    master_csv_path = Path(master_csv_path)
    
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Ensure the directory for the master CSV exists
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create master CSV with proper headers if it doesn't exist
    if not master_csv_path.exists():
        columns = [
            'video_id', 'source_folder', 'video_name', 'original_video_filename', 'unique_video_filename',
            'video_fps', 'video_duration_sec',
            'frame_filename', 'frame_file_path', 'frame_number', 'frame_timestamp_sec', 'frame_timestamp_hhmmss'
        ]
        pd.DataFrame(columns=columns).to_csv(master_csv_path, index=False)
    
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
        master_csv_path=master_csv_path
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
                was_processed = future.result()
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
    
    # Check CSV for frame statistics
    if master_csv_path.exists():
        try:
            df = pd.read_csv(master_csv_path)
            if len(df) > 0:
                print(f"\nFrame Statistics:")
                print(f"  Total frames extracted: {len(df)}")
                print(f"  Average frames per video: {len(df)/df['video_name'].nunique():.1f}")
                
                # Count frames by source folder
                if 'source_folder' in df.columns:
                    folder_counts = df['source_folder'].value_counts().sort_index()
                    print(f"\nFrames by folder:")
                    for folder, count in folder_counts.items():
                        print(f"  {folder}: {count} frames")
                
                # Show CSV file info
                csv_size = master_csv_path.stat().st_size / (1024 * 1024)  # Convert to MB
                print(f"\nCSV file: {master_csv_path}")
                print(f"CSV size: {csv_size:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not read CSV statistics: {e}")
    
    print("=" * 60)
    return processed_count, skipped_count

def _process_single_video(video_path, output_dir, frame_interval, master_csv_path):
    """Helper function for parallel video processing.
    
    Args:
        video_path (Path): Path to the video file to process.
        output_dir (Path): Directory to save extracted frames.
        frame_interval (int): Extract every Nth frame.
        master_csv_path (Path): Path to the master CSV file.
    
    Returns:
        bool: True if video was processed, False if failed or skipped.
    """
    try:
        _, was_processed = extract_frames_optimized(
            video_path, 
            output_dir, 
            frame_interval,
            master_csv_path
        )
        return was_processed
    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")
        return False

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
    parser.add_argument('--csv', '-c', required=True, help='Path for master CSV file')
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
            args.csv, 
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