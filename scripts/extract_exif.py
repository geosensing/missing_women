#!/usr/bin/env python3
"""Extract EXIF metadata from video files in parallel with modern Python patterns."""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

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
            logging.FileHandler('exif_extraction.log', mode='a')
        ]
    )


def extract_video_exif(video_path: Path, output_folder: Path, timeout: int = 180) -> Dict:
    """Extract EXIF metadata from a single video file.
    
    Args:
        video_path (Path): Path to the video file.
        output_folder (Path): Directory to save EXIF text file.
        timeout (int): Timeout in seconds for exiftool command.
        
    Returns:
        Dict: Dictionary containing video metadata and EXIF extraction status.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get basic file information
    video_name = video_path.stem
    source_folder = video_path.parent.name
    
    # Generate globally unique video identifier using path hash
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    
    file_info = {
        # Core Identity (consistent across EXIF and frame CSVs)
        'video_id': f"{source_folder}_{video_name}_{path_hash}",
        'source_folder': source_folder,
        'video_name': video_name,
        'original_video_filename': video_path.name,
        'unique_video_filename': f"{source_folder}_{video_path.name}_{path_hash}",
        
        # File Properties
        'file_size_bytes': video_path.stat().st_size,
        'file_size_mb': video_path.stat().st_size / (1024 * 1024),
        'file_created_at': datetime.fromtimestamp(video_path.stat().st_ctime),
        'file_modified_at': datetime.fromtimestamp(video_path.stat().st_mtime),
        
        # Video Properties (will be filled from EXIF)
        'video_duration_sec': None,
        'video_fps': None,
        'video_resolution_wxh': None,
        'video_codec': None,
        
        # Camera
        'camera_model': None,
        'recording_datetime': None,
        
        # EXIF References
        'exif_filename': None,
        'exif_file_path': None
    }
    
    try:
        # Run exiftool with enhanced GPS extraction
        cmd = ["exiftool", "-G", "-ee", "-api", "LargeFileSupport=1", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            logger.warning(f"exiftool returned non-zero exit code for {video_path}")
            return file_info
        
        # Generate globally unique filename using path hash
        path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        exif_filename = f"{source_folder}_{video_name}_{path_hash}_exif.txt"
        exif_file_path = output_folder / exif_filename
        
        with open(exif_file_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        
        # Update file info with consistent column names
        file_info['exif_filename'] = exif_filename
        # Use relative path from current working directory
        try:
            file_info['exif_file_path'] = str(exif_file_path.relative_to(Path.cwd()))
        except ValueError:
            # Fallback to relative to output folder
            file_info['exif_file_path'] = str(Path(output_folder.name) / exif_filename)
        
        # Parse metadata
        metadata = _parse_exif_output(result.stdout)
        file_info.update(metadata)
        
        logger.debug(f"Successfully extracted EXIF from {video_path}")
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout ({timeout}s) while processing {video_path}")
    except FileNotFoundError:
        logger.error("exiftool not found. Please install exiftool to extract EXIF data")
    except Exception as e:
        logger.error(f"Error extracting EXIF from {video_path}: {str(e)}")
    
    return file_info


def _parse_exif_output(exif_output: str) -> Dict:
    """Parse exiftool output and extract only essential metadata for clean CSV structure.
    
    Args:
        exif_output (str): Raw exiftool output.
        
    Returns:
        Dict: Parsed metadata with only essential video properties.
    """
    metadata = {
        'video_duration_sec': None,
        'video_fps': None,
        'video_resolution_wxh': None,
        'video_codec': None,
        'camera_model': None,
        'recording_datetime': None
    }
    
    lines = exif_output.strip().split('\n')
    for line in lines:
        if ': ' not in line:
            continue
            
        tag_part, value = line.split(': ', 1)
        tag = tag_part.strip().lower()
        
        # Extract video properties
        if 'duration' in tag and ('quicktime' in tag or 'composite' in tag):
            # Parse duration like "0:10:45" to seconds
            try:
                if ':' in value:
                    parts = value.split(':')
                    if len(parts) == 3:
                        h, m, s = parts
                        metadata['video_duration_sec'] = int(h) * 3600 + int(m) * 60 + float(s)
            except:
                pass
                
        elif 'frame rate' in tag or 'video frame rate' in tag:
            try:
                metadata['video_fps'] = float(value.split()[0])
            except:
                pass
                
        elif 'image size' in tag or 'video frame size' in tag:
            if 'x' in value:
                metadata['video_resolution_wxh'] = value.strip()
                
        elif 'compressor id' in tag or 'codec' in tag:
            metadata['video_codec'] = value.strip()
            
        # Extract camera info
        elif 'camera model' in tag or 'device name' in tag:
            metadata['camera_model'] = value.strip()
            
        elif 'create date' in tag and 'quicktime' in tag:
            metadata['recording_datetime'] = value.strip()
    
    return metadata


def extract_gps_data(video_path: Path, timeout: int = 180) -> List[Dict]:
    """Extract GPS data points from a video file.
    
    Args:
        video_path (Path): Path to the video file.
        timeout (int): Timeout in seconds for exiftool command.
        
    Returns:
        List[Dict]: List of GPS data points.
    """
    try:
        # Extract GPS data specifically
        cmd = ["exiftool", "-ee", "-api", "LargeFileSupport=1", "-csv", 
               "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-GPSSpeed", "-GPSDateTime", 
               str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            return []
        
        # Parse CSV output
        gps_points = []
        lines = result.stdout.strip().split('\n')
        
        if len(lines) < 2:  # No data beyond header
            return []
            
        header = lines[0].split(',')
        for line in lines[1:]:
            if not line.strip():
                continue
                
            values = line.split(',')
            if len(values) != len(header):
                continue
                
            point = dict(zip(header, values))
            point['filename'] = video_path.name
            point['source_folder'] = video_path.parent.name
            
            # Only add if we have valid GPS coordinates
            if point.get('GPSLatitude') and point.get('GPSLongitude'):
                gps_points.append(point)
        
        return gps_points
        
    except Exception as e:
        logger.error(f"Error extracting GPS data from {video_path}: {str(e)}")
        return []


def process_video_file(video_info: Tuple[Path, Path], timeout: int = 180) -> Dict:
    """Process a single video file for EXIF extraction.
    
    Args:
        video_info (Tuple[Path, Path]): Tuple of (video_path, output_folder).
        timeout (int): Timeout for exiftool commands.
        
    Returns:
        Dict: File metadata.
    """
    video_path, output_folder = video_info
    
    # Extract EXIF metadata
    file_metadata = extract_video_exif(video_path, output_folder, timeout)
    
    return file_metadata


def process_videos_parallel(input_folder: Path, exif_output_folder: Path, video_extensions: List[str],
                           max_workers: Optional[int] = None, timeout: int = 180) -> pd.DataFrame:
    """Process all videos in a folder using parallel processing.
    
    Args:
        input_folder (Path): Root folder containing videos.
        exif_output_folder (Path): Directory to save EXIF text files.
        video_extensions (List[str]): List of video file extensions to process.
        max_workers (Optional[int]): Maximum number of parallel workers.
        timeout (int): Timeout for exiftool commands.
        
    Returns:
        pd.DataFrame: Video metadata DataFrame.
    """
    # Convert to Path objects
    input_folder = Path(input_folder)
    exif_output_folder = Path(exif_output_folder)
    
    # Ensure output folder exists
    exif_output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    all_files = []
    
    def process_extension(ext: str) -> List[Path]:
        """Process a single file extension and return matching files."""
        ext_lower = ext.lower()
        match ext_lower:
            case '.mp4' | '.mov' | '.avi' | '.mkv' | '.m4v':
                files = list(input_folder.rglob(f'*{ext}'))
                files.extend(input_folder.rglob(f'*{ext.upper()}'))
                return files
            case _:
                files = list(input_folder.rglob(f'*{ext}'))
                files.extend(input_folder.rglob(f'*{ext.upper()}'))
                return files
    
    # Process each extension
    for ext in video_extensions:
        all_files.extend(process_extension(ext))
    
    # Filter out macOS metadata files and remove duplicates
    all_files = list(set([f for f in all_files if not f.name.startswith('._')]))
    all_files.sort()
    
    print(f"Found {len(all_files)} video files")
    
    if not all_files:
        logger.warning("No video files found to process")
        return pd.DataFrame()
    
    # Prepare file info for parallel processing - use flat output structure
    file_infos = []
    for video_path in all_files:
        # Use flat structure for consistency with frame extraction
        file_infos.append((video_path, exif_output_folder))
    
    # Process videos in parallel
    all_metadata = []
    
    process_func = partial(process_video_file, timeout=timeout)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(process_func, file_info): file_info[0] 
            for file_info in file_infos
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_video), 
                          total=len(file_infos), desc="Processing Videos"):
            video_path = future_to_video[future]
            try:
                metadata = future.result()
                all_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
    
    # Create DataFrame
    metadata_df = pd.DataFrame(all_metadata) if all_metadata else pd.DataFrame()
    
    return metadata_df


def save_results(metadata_df: pd.DataFrame, exif_output_folder: Path, metadata_csv_path: Path) -> None:
    """Save extraction results to CSV file with relative paths.
    
    Args:
        metadata_df (pd.DataFrame): Video metadata DataFrame.
        exif_output_folder (Path): Output folder for EXIF text files.
        metadata_csv_path (Path): Path for metadata CSV file.
    """
    # Ensure output directories exist
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add relative paths to metadata
    if not metadata_df.empty:
        # Add relative path to EXIF files (handle sibling directories)
        csv_dir = metadata_csv_path.parent
        common_parent = csv_dir.parent  # samsung_usb_output
        exif_relative_to_common = exif_output_folder.relative_to(common_parent)  # exif_files
        metadata_df['exif_relative_path'] = metadata_df.apply(
            lambda row: str(Path('..') / exif_relative_to_common / row['exif_filename']) 
            if row['exif_filename'] else None, axis=1
        )
        
        # Save metadata CSV
        metadata_df.to_csv(metadata_csv_path, index=False)
        logger.info(f"Saved metadata for {len(metadata_df)} videos to {metadata_csv_path}")


def main() -> int:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract EXIF metadata from videos in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/videos -o /path/to/exif -m metadata.csv
  %(prog)s -i videos/ -o exif_out/ -m data/metadata.csv --workers 4 --verbose
  %(prog)s -i vids/ -o exif/ -m meta.csv --timeout 300
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input folder containing videos')
    parser.add_argument('--exif-output', '-o', required=True, help='Output folder for EXIF text files')
    parser.add_argument('--metadata-csv', '-m', required=True, help='Path for video metadata CSV file')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--extensions', '-e', default='.mp4,.mov,.avi,.mkv',
                        help='Comma-separated list of video extensions to process')
    parser.add_argument('--timeout', '-t', type=int, default=180,
                        help='Timeout in seconds for exiftool commands')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Starting EXIF extraction with {args.workers or 'CPU count'} workers")
    logger.info(f"Input: {args.input}, EXIF output: {args.exif_output}")
    logger.debug(f"Extensions: {args.extensions}")
    
    # Check for exiftool
    try:
        subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
    except FileNotFoundError:
        logger.error("exiftool not found. Please install exiftool first:")
        logger.error("  macOS: brew install exiftool")
        logger.error("  Ubuntu: sudo apt-get install exiftool")
        return 1
    except subprocess.CalledProcessError:
        logger.error("exiftool found but not working properly")
        return 1
    
    # Parse extensions
    video_extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    start_time = time.time()
    
    try:
        # Process videos
        metadata_df = process_videos_parallel(
            input_folder=Path(args.input),
            exif_output_folder=Path(args.exif_output),
            video_extensions=video_extensions,
            max_workers=args.workers,
            timeout=args.timeout
        )
        
        # Save results
        save_results(metadata_df, Path(args.exif_output), Path(args.metadata_csv))
        
        elapsed = time.time() - start_time
        
        # Final summary
        print("\n" + "=" * 60)
        print("EXIF EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Videos processed: {len(metadata_df)}")
        print(f"EXIF files created: {(metadata_df['exif_filename'].notna()).sum() if not metadata_df.empty else 0}")
        print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        if len(metadata_df) > 0:
            print(f"Average time per video: {elapsed/len(metadata_df):.1f} seconds")
        print("=" * 60)
        
        logger.info(f"Extraction completed successfully. Videos: {len(metadata_df)}")
        
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
                logger.error("Permission denied accessing files")
                logger.error("Please check file/directory permissions")
                return 1
            case 'KeyboardInterrupt':
                logger.info(f"Process interrupted by user after {elapsed:.1f} seconds")
                return 130
            case 'MemoryError':
                logger.error("Insufficient memory to process videos")
                logger.error("Try reducing the number of workers")
                return 1
            case _:
                logger.error(f"Unexpected error ({error_type}): {str(e)}")
                logger.debug("Full traceback:", exc_info=True)
                return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())