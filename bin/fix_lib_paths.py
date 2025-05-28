#!/usr/bin/env python3
# filepath: fix_lib_paths.py
# Desc: Finds dylib and so files in a conda environment,
#       removes duplicate RPATHs, and re-signs them.
#       Uses concurrent.futures.ThreadPoolExecutor for parallelism.
#       Inspired by: https://github.com/conda-forge/numpy-feedstock/issues/347#issuecomment-2746317575

import logging  # Added
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Initialize logger instance. Configuration will be done in main().
logger = logging.getLogger("fix_lib_paths")


def run_command(command, check=True, capture_output=True, text=True, shell=False):
    """Helper function to run a shell command."""
    try:
        # Ensure all parts of the command are strings
        cmd_str_list = [str(c) for c in command]
        process = subprocess.run(
            cmd_str_list,
            check=check,
            capture_output=capture_output,
            text=text,
            shell=shell,  # Be cautious with shell=True
            errors="replace",  # Handle potential encoding issues in output
        )
        return process
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Command '{' '.join(map(str,e.cmd))}' failed with exit code {e.returncode}"
        )
        if e.stdout:
            logger.error(f"Stdout: {e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr.strip()}")
        return None
    except FileNotFoundError as e:
        logger.error(
            f"Command not found: {e.filename}. Please ensure it's in your PATH."
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while running command: {command}. Error: {e}"
        )
        return None


def process_library(library_path: Path):
    """
    Processes a single library file:
    1. Extracts RPATHs.
    2. Removes duplicate RPATHs.
    3. Re-signs the library.
    Returns True if successful (or no action needed), False if a critical error occurred.
    """
    logger.debug(f"Processing {library_path}...")
    processed_successfully = True  # Assume success unless an error occurs

    # 1. Extract all LC_RPATH entries
    otool_cmd = ["otool", "-l", str(library_path)]
    otool_process = run_command(otool_cmd)

    rpaths: list[str] = []
    if not otool_process or otool_process.returncode != 0:
        logger.warning(
            f"Could not extract RPATHs for {library_path}. Skipping RPATH processing."
        )
    else:
        lines = otool_process.stdout.splitlines()
        for i, line in enumerate(lines):
            if "cmd LC_RPATH" in line:
                # Path is typically on the line after "cmd LC_RPATH" and "cmdsize", specifically "path "
                if i + 2 < len(lines) and "path " in lines[i + 2]:
                    try:
                        rpath_line_content = lines[i + 2].split("path ", 1)[1]
                        rpath = rpath_line_content.split(" (offset")[0].strip()
                        if rpath:  # Ensure rpath is not empty
                            rpaths.append(rpath)
                    except IndexError:
                        logger.warning(
                            f"Could not parse RPATH from line: '{lines[i+2]}' for {library_path}"
                        )
                elif (
                    i + 1 < len(lines) and "path " in lines[i + 1]
                ):  # Sometimes it's i+1
                    try:
                        rpath_line_content = lines[i + 1].split("path ", 1)[1]
                        rpath = rpath_line_content.split(" (offset")[0].strip()
                        if rpath:
                            rpaths.append(rpath)
                    except IndexError:
                        logger.warning(
                            f"Could not parse RPATH from line: '{lines[i+1]}' for {library_path}"
                        )

    if not rpaths:
        logger.debug(f"  No RPATHs found in {library_path}.")

    # 2. Check for duplicates and remove them
    rpaths_to_remove_successfully = True  # Track success of rpath removal

    # Deduplicate RPATHs before processing removal to avoid multiple attempts on the same path
    rpaths_to_actually_delete = []

    temp_seen_for_deletion_logic = set()
    for rpath in rpaths:  # Iterate through original rpaths to find duplicates
        if rpath in temp_seen_for_deletion_logic:
            if rpath not in rpaths_to_actually_delete:  # Add only once for deletion
                rpaths_to_actually_delete.append(rpath)
            logger.debug(
                f"  Marking duplicate RPATH for removal: {rpath} in {library_path}"
            )
        else:
            temp_seen_for_deletion_logic.add(rpath)
            logger.debug(
                f"  Keeping RPATH (or first instance): {rpath} in {library_path}"
            )
    for rpath_to_delete in rpaths_to_actually_delete:
        logger.debug(
            f"  Attempting to remove duplicate RPATH: {rpath_to_delete} from {library_path}"
        )
        delete_cmd = [
            "install_name_tool",
            "-delete_rpath",
            rpath_to_delete,
            str(library_path),
        ]
        delete_process = run_command(delete_cmd, check=False)
        if delete_process and delete_process.returncode != 0:
            logger.warning(
                f"Failed to delete RPATH '{rpath_to_delete}' from {library_path}."
            )
            rpaths_to_remove_successfully = False
        elif delete_process:
            logger.debug(
                f"  Successfully deleted RPATH: {rpath_to_delete} from {library_path}"
            )
        else:  # run_command itself failed
            rpaths_to_remove_successfully = False

    # 3. Re-sign the library
    logger.debug(f"  Re-signing {library_path}")
    codesign_cmd = ["codesign", "--force", "--sign", "-", str(library_path)]
    codesign_process = run_command(codesign_cmd, check=False)

    if codesign_process and codesign_process.returncode != 0:
        logger.warning(
            f"Could not re-sign {library_path}. Stderr: {codesign_process.stderr.strip() if codesign_process.stderr else 'N/A'}"
        )
        processed_successfully = (
            False  # Consider signing failure as a processing failure
        )
    elif codesign_process:
        logger.debug(f"  Successfully re-signed {library_path}.")  # Was log_info
    else:  # run_command itself failed for codesign
        processed_successfully = False

    if not rpaths_to_remove_successfully:  # If any rpath deletion failed
        processed_successfully = False

    logger.debug(f"Done with {library_path}")
    return processed_successfully


def main():
    # --- Logging Setup ---
    # Set the overall logger level. To see DEBUG messages, set this to logging.DEBUG
    # For example, by checking an environment variable:
    # log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    # current_log_level = getattr(logging, log_level_str, logging.INFO)
    # logger.setLevel(current_log_level)
    logger.setLevel(
        logging.INFO
    )  # Default: INFO, WARNING, ERROR. Set to logging.DEBUG to see debug messages.

    # Handler for DEBUG and INFO messages to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)  # Process messages from DEBUG level
    # Filter to ensure only DEBUG and INFO messages are handled by this handler
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    stdout_formatter = logging.Formatter("%(message)s")  # Simple format for debug/info
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

    # Handler for WARNING, ERROR, CRITICAL messages to stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)  # Process messages from WARNING level
    stderr_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )  # Prefix with level name
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)

    logger.propagate = False  # Prevent messages from being passed to the root logger
    # --- End Logging Setup ---

    conda_prefix_env = os.environ.get("CONDA_PREFIX")
    if not conda_prefix_env:
        logger.error(
            "CONDA_PREFIX environment variable is not set. Please activate a conda environment."
        )
        sys.exit(1)

    lib_path = Path(conda_prefix_env) / "lib"
    if not lib_path.is_dir():
        logger.error(f"Library directory not found: {lib_path}")
        sys.exit(1)

    logger.info(f"Scanning for libraries in: {lib_path}")

    libraries_to_process = []
    for ext in ("*.dylib", "*.so"):
        libraries_to_process.extend(lib_path.rglob(ext))

    if not libraries_to_process:
        logger.info("No .dylib or .so files found to process.")
        sys.exit(0)

    logger.info(f"Found {len(libraries_to_process)} libraries to process.")

    results = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_library, libraries_to_process))

    successful_processing = sum(1 for r in results if r is True)
    failed_processing = len(libraries_to_process) - successful_processing

    logger.info(f"\n--- Summary ---")  # Using info for summary
    logger.info(f"Total libraries attempted: {len(libraries_to_process)}")
    logger.info(f"Successfully processed: {successful_processing}")
    if failed_processing > 0:
        logger.warning(  # Warning for issues
            f"Encountered issues with {failed_processing} libraries (check logs above)."
        )
    else:
        logger.info("All libraries processed without reported issues.")

    if failed_processing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
