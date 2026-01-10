#!/usr/bin/env python3
"""Extract detailed GPS timeseries data from EXIF text files."""

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
            logging.FileHandler('gps_extraction.log', mode='a')
        ]
    )


def extract_video_id_from_filename(exif_filename: str) -> str:
    """Extract video_id from EXIF filename.
    
    Args:
        exif_filename (str): EXIF filename like "10_itinerary_11_7c12e80b_exif.txt"
        
    Returns:
        str: Video ID like "10_itinerary_11_7c12e80b"
    """
    # Remove _exif.txt suffix
    if exif_filename.endswith('_exif.txt'):
        return exif_filename[:-9]
    return exif_filename.replace('.txt', '')


def parse_gps_from_exif(exif_file: Path) -> List[Dict]:
    """Parse GPS data from a single EXIF text file.
    
    Args:
        exif_file (Path): Path to EXIF text file.
        
    Returns:
        List[Dict]: List of GPS data points with timestamps.
    """
    gps_data = []
    video_id = extract_video_id_from_filename(exif_file.name)
    
    try:
        with open(exif_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match GoPro GPS blocks
        gps_pattern = r'\[GoPro\]\s+GPS Measure Mode\s+:\s+(.+?)\n\[GoPro\]\s+GPS Latitude\s+:\s+(.+?)\n\[GoPro\]\s+GPS Longitude\s+:\s+(.+?)\n\[GoPro\]\s+GPS Altitude\s+:\s+(.+?)\n\[GoPro\]\s+GPS Speed\s+:\s+(.+?)\n\[GoPro\]\s+GPS Speed 3D\s+:\s+(.+?)\n\[GoPro\]\s+GPS Date Time\s+:\s+(.+?)\n(?:\[GoPro\]\s+GPSDOP\s+:\s+(.+?)\n)?'
        
        matches = re.findall(gps_pattern, content)
        
        for match in matches:
            measure_mode, latitude, longitude, altitude, speed, speed_3d, date_time = match[:7]
            gpsdop = match[7] if len(match) > 7 else None
            
            # Clean up values
            latitude = latitude.strip()
            longitude = longitude.strip()
            altitude = altitude.strip()
            speed = speed.strip()
            speed_3d = speed_3d.strip()
            date_time = date_time.strip()
            
            # Skip if no valid coordinates
            if not latitude or not longitude or latitude == '-' or longitude == '-':
                continue
                
            gps_point = {
                'video_id': video_id,
                'gps_datetime': date_time,
                'gps_latitude': latitude,
                'gps_longitude': longitude,
                'gps_altitude': altitude,
                'gps_speed': speed,
                'gps_speed_3d': speed_3d,
                'gps_measure_mode': measure_mode.strip(),
                'gpsdop': gpsdop.strip() if gpsdop else None
            }
            
            gps_data.append(gps_point)
            
    except Exception as e:
        logger.error(f"Error processing {exif_file}: {str(e)}")
    
    return gps_data


def process_exif_directory(exif_dir: Path, file_pattern: str = "*_exif.txt") -> pd.DataFrame:
    """Process all EXIF files in a directory to extract GPS data.
    
    Args:
        exif_dir (Path): Directory containing EXIF text files.
        file_pattern (str): File pattern to match EXIF files.
        
    Returns:
        pd.DataFrame: DataFrame with GPS timeseries data.
    """
    exif_files = list(exif_dir.glob(file_pattern))
    
    if not exif_files:
        logger.warning(f"No EXIF files found matching pattern '{file_pattern}' in {exif_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(exif_files)} EXIF files to process")
    
    all_gps_data = []
    
    for exif_file in tqdm(exif_files, desc="Processing EXIF files"):
        gps_points = parse_gps_from_exif(exif_file)
        all_gps_data.extend(gps_points)
        
        if gps_points:
            logger.debug(f"Extracted {len(gps_points)} GPS points from {exif_file.name}")
        else:
            logger.debug(f"No GPS data found in {exif_file.name}")
    
    if all_gps_data:
        df = pd.DataFrame(all_gps_data)
        
        # Sort by video_id and datetime for consistent output
        df = df.sort_values(['video_id', 'gps_datetime'])
        
        return df
    else:
        return pd.DataFrame()


def main() -> int:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract detailed GPS timeseries from EXIF text files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i exif_files/ -o gps_timeseries.csv
  %(prog)s -i exif/ -o detailed_gps.csv --pattern "*_exif.txt" --verbose
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Directory containing EXIF text files')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file for GPS timeseries data')
    parser.add_argument('--pattern', '-p', default='*_exif.txt',
                        help='File pattern to match EXIF files (default: *_exif.txt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logger.info(f"Starting GPS timeseries extraction")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.debug(f"File pattern: {args.pattern}")
    
    # Validate input directory
    exif_dir = Path(args.input)
    if not exif_dir.exists():
        logger.error(f"Input directory does not exist: {exif_dir}")
        return 1
    
    if not exif_dir.is_dir():
        logger.error(f"Input path is not a directory: {exif_dir}")
        return 1
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Process EXIF files
        gps_df = process_exif_directory(exif_dir, args.pattern)
        
        if gps_df.empty:
            logger.warning("No GPS data found in any EXIF files")
            # Create empty CSV with headers
            empty_df = pd.DataFrame(columns=[
                'video_id', 'gps_datetime', 'gps_latitude', 'gps_longitude', 
                'gps_altitude', 'gps_speed', 'gps_speed_3d', 'gps_measure_mode', 'gpsdop'
            ])
            empty_df.to_csv(output_path, index=False)
        else:
            # Save GPS data
            gps_df.to_csv(output_path, index=False)
        
        elapsed = time.time() - start_time
        
        # Final summary
        print("\n" + "=" * 60)
        print("GPS TIMESERIES EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"EXIF files processed: {len(list(exif_dir.glob(args.pattern)))}")
        print(f"GPS data points extracted: {len(gps_df)}")
        print(f"Unique videos with GPS: {gps_df['video_id'].nunique() if not gps_df.empty else 0}")
        print(f"Output file: {output_path}")
        print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        print("=" * 60)
        
        logger.info(f"Extraction completed successfully. GPS points: {len(gps_df)}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        
        # Handle different error types
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
                logger.error("Insufficient memory to process EXIF files")
                logger.error("Try processing fewer files at once")
                return 1
            case _:
                logger.error(f"Unexpected error ({error_type}): {str(e)}")
                logger.debug("Full traceback:", exc_info=True)
                return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())