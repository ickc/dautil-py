#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import subprocess
import sys


def setup_logging(debug_mode):
    """
    Configures the logging format and level.

    Args:
        debug_mode (bool): If True, sets the logging level to DEBUG,
                           otherwise sets it to INFO.
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def is_video_corrupted(filepath):
    """
    Checks if a video file is corrupted using a fast, heuristic-based method.

    This method uses ffmpeg to remux the video and audio streams without
    fully decoding them ('stream copying'). This is much faster than a full
    decode as it is primarily limited by disk I/O, not CPU. It's effective
    at catching container-level errors, broken streams, or major packet
    corruption that prevents a simple remux.

    Args:
        filepath (str): The path to the video file.

    Returns:
        bool: True if the file is likely corrupted, False otherwise.
    """
    command = [
        "ffmpeg",
        "-v",
        "error",  # Only print error messages
        "-i",
        filepath,  # Input file
        "-c",
        "copy",  # Copy all streams without re-encoding
        "-f",
        "null",  # Use the null muxer (discard output)
        "-",  # Output to stdout (which is discarded)
    ]

    try:
        # We run the command and capture stderr. A non-zero return code
        # suggests that ffmpeg encountered a fatal error during the copy.
        result = subprocess.run(
            command,
            check=False,  # Don't raise an exception on non-zero exit codes
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.debug(f"ffmpeg stderr for {filepath}:\n{result.stderr}")
            return True
        return False
    except FileNotFoundError:
        logging.error(
            "ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH."
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while checking {filepath}: {e}")
        return True  # Treat unexpected errors as a sign of corruption


def main():
    """Main function to parse arguments and process video files."""
    parser = argparse.ArgumentParser(
        description="Find and optionally delete corrupted video files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Find corrupted MP4 files in the current directory and subdirectories
  python check_videos.py "**/*.mp4"

  # Find and delete corrupted MKV files, with verbose logging
  python check_videos.py --delete --verbose "**/*.mkv"
""",
    )
    parser.add_argument(
        "glob_pattern",
        help="Glob pattern to find video files (e.g., '**/*.mp4').\n"
        "Use quotes to prevent your shell from expanding it.",
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete files identified as corrupted."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging to see all files checked.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    logging.info(f"Searching for files matching pattern: {args.glob_pattern}")

    # Use recursive=True to support the "**" glob pattern for subdirectories
    try:
        video_files = glob.glob(args.glob_pattern, recursive=True)
    except Exception as e:
        logging.error(f"Invalid glob pattern provided: {e}")
        sys.exit(1)

    if not video_files:
        logging.warning("No files found matching the specified pattern.")
        return

    corrupted_count = 0
    ok_count = 0

    for filepath in video_files:
        if not os.path.isfile(filepath):
            continue

        if is_video_corrupted(filepath):
            logging.info(f"CORRUPTED: {filepath}")
            corrupted_count += 1
            if args.delete:
                try:
                    os.remove(filepath)
                    logging.info(f"DELETED: {filepath}")
                except OSError as e:
                    logging.error(f"Failed to delete {filepath}: {e}")
        else:
            logging.debug(f"OK: {filepath}")
            ok_count += 1

    logging.info("-" * 20)
    logging.info("Scan complete.")
    logging.info(f"Total files checked: {len(video_files)}")
    logging.info(f"Healthy files: {ok_count}")
    logging.info(f"Corrupted files found: {corrupted_count}")
    if args.delete and corrupted_count > 0:
        logging.info(f"Successfully deleted {corrupted_count} corrupted files.")
    elif args.delete:
        logging.info("No corrupted files to delete.")


if __name__ == "__main__":
    main()
