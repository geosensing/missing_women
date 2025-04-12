import os
import argparse
import cv2
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from glob import glob
import piexif
import json
import concurrent.futures
from tqdm import tqdm  # Regular tqdm instead of notebook version
import numpy as np
import time
from functools import partial

def extract_frames_optimized(video_path, output_folder, frame_interval, master_csv_path):
    """
    Optimized version of extract_frames that uses more efficient frame seeking
    and better I/O handling
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get video file information
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    # Check if this video has already been processed in the master CSV
    if os.path.exists(master_csv_path):
        try:
            existing_df = pd.read_csv(master_csv_path)
            if 'video_name' in existing_df.columns and video_name in existing_df['video_name'].values:
                return pd.DataFrame(), False
        except Exception as e:
            print(f"Warning: Error reading master CSV: {str(e)}. Will proceed with processing.")
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
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
        
        # Create filename with time information
        filename = f"{video_name}_frame{frame_num}_t{time_str_filename}.jpg"
        filepath = os.path.join(output_folder, filename)
        
        # Save the frame with optimized parameters
        cv2.imwrite(filepath, frame, encode_params)
        
        # Calculate relative folder path
        relative_folder_path = os.path.relpath(output_folder, os.path.dirname(master_csv_path))
        
        # Store frame info
        frame_info = {
            'filename': filename,
            'filepath': filepath,
            'folder_name': os.path.basename(output_folder),
            'relative_folder_path': relative_folder_path,
            'frame_number': frame_idx,
            'frame_time_seconds': frame_time_seconds,
            'frame_time_hhmmss': time_str_display,
            'video_name': video_name,
            'video_fps': fps,
            'video_duration': duration,
            'source_video_path': video_path
        }
        
        # Add frame info to our list
        frame_data.append(frame_info)
    
    video.release()
    
    # Create DataFrame from frame data
    new_frame_df = pd.DataFrame(frame_data)
    
    # Update master CSV with thread/process safe approach
    if not new_frame_df.empty:
        # Use a lock file to prevent race conditions
        lock_file = f"{master_csv_path}.lock"
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Try to create a lock file
                if not os.path.exists(lock_file):
                    with open(lock_file, 'w') as f:
                        f.write(str(os.getpid()))
                    
                    try:
                        if os.path.exists(master_csv_path):
                            existing_df = pd.read_csv(master_csv_path)
                            updated_df = pd.concat([existing_df, new_frame_df], ignore_index=True)
                            updated_df.to_csv(master_csv_path, index=False)
                        else:
                            new_frame_df.to_csv(master_csv_path, index=False)
                        
                        # Success - remove lock and break
                        os.remove(lock_file)
                        break
                    except Exception as e:
                        # If an error occurred, remove the lock and retry
                        if os.path.exists(lock_file):
                            os.remove(lock_file)
                        raise e
                else:
                    # Lock exists, wait and retry
                    time.sleep(0.5)
                    retry_count += 1
            except Exception as e:
                print(f"CSV update error: {str(e)}. Retry {retry_count+1}/{max_retries}")
                retry_count += 1
                time.sleep(1)
                
        # If we couldn't update the master CSV after all retries
        if retry_count >= max_retries:
            backup_path = f"{os.path.splitext(master_csv_path)[0]}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            new_frame_df.to_csv(backup_path, index=False)
    
    return new_frame_df, True

def process_video_folder_parallel(input_folder, output_folder, frame_interval, master_csv_path, 
                                  video_extensions=['.mp4', '.avi', '.mov', '.mkv'], 
                                  max_workers=None):
    """
    Process all videos in a folder using parallel processing
    
    Args:
        input_folder (str): Root folder containing videos
        output_folder (str): Directory to save extracted frames
        frame_interval (int): Extract every Xth frame
        master_csv_path (str): Path to the master CSV file
        video_extensions (list): List of video file extensions to process
        max_workers (int): Maximum number of parallel workers (None = CPU count)
        
    Returns:
        int: Number of videos processed
        int: Number of videos skipped
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Ensure the directory for the master CSV exists
    master_csv_dir = os.path.dirname(master_csv_path)
    os.makedirs(master_csv_dir, exist_ok=True)
    
    # Create master CSV if it doesn't exist
    if not os.path.exists(master_csv_path):
        pd.DataFrame().to_csv(master_csv_path, index=False)
    
    # Find all video files in the directory and subdirectories
    all_files = []
    
    # Walk through directory structure
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Check if file extension matches any of our video extensions (case-insensitive)
            if any(file.lower().endswith(ext.lower()) for ext in video_extensions):
                all_files.append(os.path.join(root, file))
    
    # Sort files for consistent processing order
    all_files.sort()
    
    print(f"Found {len(all_files)} video files")
    
    # Prepare args for each file
    file_infos = []
    for video_path in all_files:
        # Get relative path to maintain folder structure
        rel_path = os.path.relpath(video_path, input_folder)
        rel_dir = os.path.dirname(rel_path)
        
        # Create output directory that mirrors the input structure
        video_output_folder = os.path.join(output_folder, rel_dir)
        os.makedirs(video_output_folder, exist_ok=True)
        
        file_infos.append((video_path, video_output_folder))
    
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
                print(f"Error processing {video_path}: {str(e)}")
    
    print(f"\nAll done. Processed {processed_count} videos, skipped {skipped_count} videos that were already processed.")
    return processed_count, skipped_count

def _process_single_video(video_path, output_dir, frame_interval, master_csv_path):
    """Helper function for parallel processing"""
    try:
        _, was_processed = extract_frames_optimized(
            video_path, 
            output_dir, 
            frame_interval,
            master_csv_path
        )
        return was_processed
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False

# Main function to add for command-line use
def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos in parallel')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing videos')
    parser.add_argument('--output', '-o', required=True, help='Output folder for extracted frames')
    parser.add_argument('--interval', '-n', type=int, default=30, help='Extract every Nth frame')
    parser.add_argument('--csv', '-c', required=True, help='Path for master CSV file')
    parser.add_argument('--workers', '-w', type=int, default=None, 
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--extensions', '-e', default='.mp4,.avi,.mov,.mkv',
                        help='Comma-separated list of video extensions to process')
    
    args = parser.parse_args()
    
    # Split extensions string into a list
    video_extensions = args.extensions.split(',')
    
    start_time = time.time()
    
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
    print(f"Completed in {elapsed:.2f} seconds. Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__":
    main()