#!/usr/bin/env python3
"""
GitHub Actions Artifacts Cleanup Script

Removes old GitHub Actions artifacts from a repository.
Supports dry-run mode and interactive confirmation.

Inspired by a bash version in
https://www.eliostruyf.com/clean-github-actions-artifacts-script/
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with appropriate format and level."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("gh-cleanup")
    logger.setLevel(log_level)

    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def check_gh_auth(logger: logging.Logger) -> bool:
    """Check if GitHub CLI is authenticated."""
    logger.debug("Checking GitHub CLI authentication status")
    try:
        result = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, check=True
        )
        logger.debug("GitHub CLI is authenticated")
        return True
    except subprocess.CalledProcessError:
        logger.error("GitHub CLI is not authenticated")
        logger.error("Please authenticate using: gh auth login")
        return False
    except FileNotFoundError:
        logger.error("GitHub CLI (gh) is not installed")
        logger.error("Please install it from: https://cli.github.com/")
        return False


def call_gh_api(
    endpoint: str, method: str = "GET", logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[Dict]]:
    """
    Call GitHub API using gh CLI.

    Returns:
        Tuple of (success: bool, response: dict or None)
    """
    if logger:
        logger.debug(f"Calling GitHub API: {method} {endpoint}")

    cmd = ["gh", "api", "-H", "Accept: application/vnd.github+json"]

    if method != "GET":
        cmd.extend(["-X", method])

    cmd.append(endpoint)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if method == "DELETE":
            return True, None

        try:
            data = json.loads(result.stdout)
            return True, data
        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"Failed to parse JSON response: {e}")
            return False, None

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"API call failed: {e.stderr}")
        return False, None


def calculate_age_days(created_at: str, logger: logging.Logger) -> Optional[int]:
    """Calculate age of artifact in days from ISO 8601 timestamp."""
    try:
        # Parse ISO 8601 timestamp
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        current_dt = datetime.now(timezone.utc)
        age = current_dt - created_dt
        return age.days
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse timestamp '{created_at}': {e}")
        return None


def fetch_artifacts(
    repo: str, page: int, logger: logging.Logger
) -> Tuple[bool, List[Dict]]:
    """Fetch a page of artifacts from the repository."""
    endpoint = f"/repos/{repo}/actions/artifacts?per_page=100&page={page}"
    logger.info(f"Fetching artifacts page {page} for repository: {repo}")

    success, response = call_gh_api(endpoint, logger=logger)

    if not success or response is None:
        logger.error(f"Failed to fetch artifacts for page {page}")
        return False, []

    artifacts = response.get("artifacts", [])
    logger.debug(f"Found {len(artifacts)} artifacts on page {page}")

    return True, artifacts


def delete_artifact(
    repo: str,
    artifact_id: int,
    artifact_name: str,
    dry_run: bool,
    logger: logging.Logger,
) -> bool:
    """Delete a single artifact."""
    if dry_run:
        logger.info(
            f"[DRY-RUN] Would delete artifact: {artifact_name} (ID: {artifact_id})"
        )
        return True

    logger.info(f"Deleting artifact: {artifact_name} (ID: {artifact_id})")
    endpoint = f"/repos/{repo}/actions/artifacts/{artifact_id}"

    success, _ = call_gh_api(endpoint, method="DELETE", logger=logger)

    if success:
        logger.debug(f"Successfully deleted artifact {artifact_id}")
    else:
        logger.error(f"Failed to delete artifact {artifact_id}")

    return success


def confirm_deletion(total_count: int) -> bool:
    """Ask user for confirmation before deletion."""
    print(f"\nFound {total_count} artifact(s) to delete.")
    response = (
        input("Do you want to proceed with deletion? (y/yes to confirm): ")
        .strip()
        .lower()
    )
    return response in ["y", "yes"]


def process_artifacts(
    repo: str, days_old: int, dry_run: bool, interactive: bool, logger: logging.Logger
) -> Tuple[int, int]:
    """
    Process all artifacts in the repository.

    Returns:
        Tuple of (deleted_count, skipped_count)
    """
    logger.info(f"Starting artifact cleanup for repository: {repo}")
    logger.info(f"Will delete artifacts older than {days_old} days")

    if dry_run:
        logger.info("Running in DRY-RUN mode - no artifacts will be deleted")

    page = 1
    total_deleted = 0
    total_skipped = 0
    artifacts_to_delete = []

    while True:
        success, artifacts = fetch_artifacts(repo, page, logger)

        if not success:
            logger.error("Failed to fetch artifacts, aborting")
            break

        if not artifacts:
            logger.info("No more artifacts found")
            break

        for artifact in artifacts:
            artifact_id = artifact.get("id")
            artifact_name = artifact.get("name")
            created_at = artifact.get("created_at")

            # Validate artifact data
            if not all([artifact_id, artifact_name, created_at]):
                logger.warning(f"Skipping artifact with incomplete data: {artifact}")
                total_skipped += 1
                continue

            # Calculate age
            age_days = calculate_age_days(created_at, logger)
            if age_days is None:
                logger.warning(
                    f"Skipping artifact {artifact_name} - could not determine age"
                )
                total_skipped += 1
                continue

            # Check if artifact should be deleted
            if age_days > days_old:
                logger.debug(
                    f"Artifact {artifact_name} is {age_days} days old - marking for deletion"
                )
                artifacts_to_delete.append(
                    {"id": artifact_id, "name": artifact_name, "age_days": age_days}
                )
            else:
                logger.debug(
                    f"Artifact {artifact_name} is {age_days} days old - keeping"
                )
                total_skipped += 1

        # Check if there are more pages
        if len(artifacts) < 100:
            logger.debug("Last page reached (fewer than 100 artifacts)")
            break

        page += 1

    # Handle deletion with optional confirmation
    if artifacts_to_delete:
        if interactive and not dry_run:
            if not confirm_deletion(len(artifacts_to_delete)):
                logger.info("Deletion cancelled by user")
                return 0, total_skipped + len(artifacts_to_delete)

        for artifact in artifacts_to_delete:
            success = delete_artifact(
                repo, artifact["id"], artifact["name"], dry_run, logger
            )

            if success:
                total_deleted += 1
                logger.info(
                    f"Processed: {artifact['name']} (Age: {artifact['age_days']} days)"
                )
            else:
                total_skipped += 1
    else:
        logger.info("No artifacts found that meet deletion criteria")

    return total_deleted, total_skipped


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up old GitHub Actions artifacts from a repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s owner/repo                    # Delete artifacts older than 30 days
  %(prog)s owner/repo --days 7           # Delete artifacts older than 7 days
  %(prog)s owner/repo --dry-run          # Preview what would be deleted
  %(prog)s owner/repo --interactive      # Ask for confirmation before deletion
  %(prog)s owner/repo -v                 # Verbose output with debug logging
        """,
    )

    parser.add_argument("repo", help="Repository in format owner/repo")

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Delete artifacts older than this many days (default: 30)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Ask for confirmation before deleting artifacts",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Validate repository format
    if "/" not in args.repo:
        logger.error(f"Invalid repository format: {args.repo}")
        logger.error("Expected format: owner/repo")
        sys.exit(1)

    # Check GitHub CLI authentication
    if not check_gh_auth(logger):
        sys.exit(1)

    # Process artifacts
    try:
        deleted, skipped = process_artifacts(
            args.repo, args.days, args.dry_run, args.interactive, logger
        )

        # Print summary
        logger.info("=" * 50)
        logger.info("Cleanup Summary:")
        logger.info(f"  Repository: {args.repo}")
        logger.info(f"  Age threshold: {args.days} days")
        if args.dry_run:
            logger.info(f"  Mode: DRY-RUN")
            logger.info(f"  Would delete: {deleted} artifact(s)")
        else:
            logger.info(f"  Deleted: {deleted} artifact(s)")
        logger.info(f"  Skipped: {skipped} artifact(s)")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
