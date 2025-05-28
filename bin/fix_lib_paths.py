#!/usr/bin/env python3
# filepath: fix_lib_paths.py
# Desc: Finds dylib and so files in a conda environment,
#       removes duplicate RPATHs, and re-signs them.
#       Uses concurrent.futures.ThreadPoolExecutor for parallelism.
#       Inspired by: https://github.com/conda-forge/numpy-feedstock/issues/347#issuecomment-2746317575

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERBOSE = False


def log_info(message):
    """Logs an informational message to stdout if VERBOSE is True."""
    if VERBOSE:
        print(message, file=sys.stdout)


def log_error(message):
    """Logs an error message to stderr."""
    print(f"ERROR: {message}", file=sys.stderr)


def log_warning(message):
    """Logs a warning message to stderr."""
    print(f"WARNING: {message}", file=sys.stderr)


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
        log_error(
            f"Command '{' '.join(map(str,e.cmd))}' failed with exit code {e.returncode}"
        )
        if e.stdout:
            log_error(f"Stdout: {e.stdout.strip()}")
        if e.stderr:
            log_error(f"Stderr: {e.stderr.strip()}")
        return None
    except FileNotFoundError as e:
        log_error(f"Command not found: {e.filename}. Please ensure it's in your PATH.")
        return None
    except Exception as e:
        log_error(
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
    log_info(f"Processing {library_path}...")
    processed_successfully = True  # Assume success unless an error occurs

    # 1. Extract all LC_RPATH entries
    otool_cmd = ["otool", "-l", str(library_path)]
    otool_process = run_command(otool_cmd)

    rpaths: list[str] = []
    if not otool_process or otool_process.returncode != 0:
        log_warning(
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
                        log_warning(
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
                        log_warning(
                            f"Could not parse RPATH from line: '{lines[i+1]}' for {library_path}"
                        )

    if not rpaths and VERBOSE:
        log_info(f"  No RPATHs found in {library_path}.")

    # 2. Check for duplicates and remove them
    rpaths_to_remove_successfully = True  # Track success of rpath removal

    # Deduplicate RPATHs before processing removal to avoid multiple attempts on the same path
    unique_rpaths_from_file = list(
        dict.fromkeys(rpaths)
    )  # Preserves order, gets unique
    rpaths_to_actually_delete = []

    temp_seen_for_deletion_logic = set()
    for rpath in rpaths:  # Iterate through original rpaths to find duplicates
        if rpath in temp_seen_for_deletion_logic:
            if rpath not in rpaths_to_actually_delete:  # Add only once for deletion
                rpaths_to_actually_delete.append(rpath)
            log_info(
                f"  Marking duplicate RPATH for removal: {rpath} in {library_path}"
            )
        else:
            temp_seen_for_deletion_logic.add(rpath)
            log_info(f"  Keeping RPATH (or first instance): {rpath} in {library_path}")

    for rpath_to_delete in rpaths_to_actually_delete:
        log_info(
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
            log_warning(
                f"Failed to delete RPATH '{rpath_to_delete}' from {library_path}."
            )
            rpaths_to_remove_successfully = False
        elif delete_process:
            log_info(
                f"  Successfully deleted RPATH: {rpath_to_delete} from {library_path}"
            )
        else:  # run_command itself failed
            rpaths_to_remove_successfully = False

    # 3. Re-sign the library (only if RPATHs were modified or always if desired)
    # The original script re-signs regardless of whether RPATHs were changed.
    log_info(f"  Re-signing {library_path}")
    codesign_cmd = ["codesign", "--force", "--sign", "-", str(library_path)]
    codesign_process = run_command(codesign_cmd, check=False)

    if codesign_process and codesign_process.returncode != 0:
        log_warning(
            f"Could not re-sign {library_path}. Stderr: {codesign_process.stderr.strip() if codesign_process.stderr else 'N/A'}"
        )
        processed_successfully = (
            False  # Consider signing failure as a processing failure
        )
    elif codesign_process:
        log_info(f"  Successfully re-signed {library_path}.")
    else:  # run_command itself failed for codesign
        processed_successfully = False

    if not rpaths_to_remove_successfully:  # If any rpath deletion failed
        processed_successfully = False

    log_info(f"Done with {library_path}")
    return processed_successfully


def main():
    conda_prefix_env = os.environ.get("CONDA_PREFIX")
    if not conda_prefix_env:
        log_error(
            "CONDA_PREFIX environment variable is not set. Please activate a conda environment."
        )
        sys.exit(1)

    lib_path = Path(conda_prefix_env) / "lib"
    if not lib_path.is_dir():
        log_error(f"Library directory not found: {lib_path}")
        sys.exit(1)

    log_info(f"Scanning for libraries in: {lib_path}")

    libraries_to_process = []
    for ext in ("*.dylib", "*.so"):
        # Using rglob for recursive search as in the original find command
        libraries_to_process.extend(lib_path.rglob(ext))

    if not libraries_to_process:
        log_info("No .dylib or .so files found to process.")
        sys.exit(0)

    log_info(f"Found {len(libraries_to_process)} libraries to process.")

    results = []
    # Using ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor() as executor:
        # The map function will block until all tasks are complete
        # It returns an iterator, so we convert it to a list to get all results
        results = list(executor.map(process_library, libraries_to_process))

    successful_processing = sum(
        1 for r in results if r is True
    )  # process_library returns True or False
    failed_processing = len(libraries_to_process) - successful_processing

    log_info(f"\n--- Summary ---")
    log_info(f"Total libraries attempted: {len(libraries_to_process)}")
    log_info(f"Successfully processed: {successful_processing}")
    if failed_processing > 0:
        log_warning(
            f"Encountered issues with {failed_processing} libraries (check logs above)."
        )
    else:
        log_info("All libraries processed without reported issues.")

    if failed_processing > 0:
        sys.exit(1)  # Exit with error code if there were failures


if __name__ == "__main__":
    main()
