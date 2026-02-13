"""
Upload release artifacts for a specific target to GitHub releases.

This script uploads files from build/{target}/release_artifacts
to a GitHub release, comparing manifests with existing assets.

Usage:
    python main.py --tag v1.0.0 --target linux-x64-cpu
    python main.py --tag v1.0.0 --target linux-x64-cpu --repo owner/repo
    python main.py --tag v1.0.0 --target linux-x64-cpu --dry-run

Credentials:
    Create a .creds.json file in the current working directory with:
    {
        "github_token": "your_github_token_here"
    }

    Alternatively, set the GITHUB_TOKEN environment variable.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from scripts.upload.hash_utils import hash_url_quick, hash_file_quick

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Install it with: pip install requests")
    sys.exit(1)


class GitHubReleaseUploader:
    def __init__(self, repo: str, token: str, dry_run: bool = False):
        self.repo = repo
        self.token = token
        self.dry_run = dry_run
        self.api_base = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_release_by_tag(self, tag: str) -> Optional[Dict]:
        """Get release information by tag name."""
        url = f"{self.api_base}/repos/{self.repo}/releases/tags/{tag}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 404:
            print(f"Release with tag '{tag}' not found.")
            return None

        response.raise_for_status()
        return response.json()

    def get_release_assets(self, release_id: int) -> Dict[str, Dict]:
        """Get all assets for a release, indexed by filename."""
        assets = []
        page = 1
        per_page = 100  # Max allowed by GitHub API

        while True:
            url = f"{self.api_base}/repos/{self.repo}/releases/{release_id}/assets"
            response = requests.get(url, headers=self.headers, params={"page": page, "per_page": per_page})
            response.raise_for_status()

            page_assets = response.json()
            if not page_assets:
                break

            assets.extend(page_assets)
            page += 1

        return {asset["name"]: asset for asset in assets}

    def delete_asset(self, asset_id: int, asset_name: str):
        """Delete an existing release asset."""
        if self.dry_run:
            print(f"  [DRY-RUN] Would delete existing asset: {asset_name}")
            return

        url = f"{self.api_base}/repos/{self.repo}/releases/assets/{asset_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        print(f"  Deleted existing asset: {asset_name}")

    def upload_asset(self, release_id: int, upload_url: str, file_path: Path, asset_name: str):
        """Upload a file as a release asset."""
        if self.dry_run:
            print(f"  [DRY-RUN] Would upload: {asset_name} ({file_path.stat().st_size} bytes)")
            return

        # Remove the template part from upload_url
        upload_url = upload_url.split("{")[0]

        with open(file_path, "rb") as f:
            file_data = f.read()

        headers = self.headers.copy()
        if asset_name.endswith(".json"):
            headers["Content-Type"] = "application/json"
        else:
            headers["Content-Type"] = "application/gzip"

        params = {"name": asset_name}
        response = requests.post(upload_url, headers=headers, params=params, data=file_data)
        response.raise_for_status()

        print(f"  Uploaded: {asset_name} ({len(file_data)} bytes)")

    def download_manifest(self, asset: Dict) -> Optional[Dict]:
        """Download a manifest file and parse its JSON content."""
        try:
            response = requests.get(asset["browser_download_url"])
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"    Warning: Could not download manifest {asset['name']}: {e}")
            return None


def load_manifest(artifacts_dir: Path) -> Optional[Dict]:
    """Load the manifest JSON file from a release_artifacts directory."""
    # Find the manifest file (pattern: {platform}-manifest.json)
    manifest_files = list(artifacts_dir.glob("*-manifest.json"))

    if not manifest_files:
        print(f"  Warning: No manifest file found in {artifacts_dir}")
        return None

    if len(manifest_files) > 1:
        print(f"  Warning: Multiple manifest files found in {artifacts_dir}, using first")

    manifest_file = manifest_files[0]
    try:
        with open(manifest_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error reading manifest {manifest_file}: {e}")
        return None


def load_github_token() -> Optional[str]:
    """Load GitHub token from .creds.json file in pwd or environment variable."""
    creds_file = Path(".creds.json")

    # Try to load from .creds.json first
    if creds_file.exists():
        try:
            with open(creds_file, "r") as f:
                creds = json.load(f)
                token = creds.get("github_token")
                if token:
                    return token
        except Exception as e:
            print(f"Warning: Could not read .creds.json: {e}")

    # Fall back to environment variable
    return os.environ.get("GITHUB_TOKEN")


def compare_and_upload(
    uploader: GitHubReleaseUploader,
    release_id: int,
    upload_url: str,
    artifacts_dir: Path,
    manifest: Dict,
    manifest_path: Path,
    existing_assets: Dict[str, Dict],
    force: bool = False,
) -> tuple[int, int]:
    """
    Compare local artifacts with remote assets and upload if different.

    Returns:
        Tuple of (uploaded_count, skipped_count)
    """
    uploaded = 0
    skipped = 0

    files = manifest.get("files", {})
    platform_name = artifacts_dir.parent.name
    remote_manifest = None

    print(f"\nProcessing platform: {platform_name}")
    print(f"  Found {len(files)} files in manifest")

    # First, handle the manifest file itself
    manifest_filename = manifest_path.name
    print(f"  Checking manifest: {manifest_filename}")

    if manifest_filename in existing_assets and not force:
        existing_asset = existing_assets[manifest_filename]
        remote_manifest = uploader.download_manifest(existing_asset)

        # Compare manifest content (as JSON) to determine if upload is needed
        if remote_manifest and remote_manifest == manifest:
            print(f"    Manifest content matches, skipping")
            skipped += 1
        else:
            print(f"    Manifest content differs, uploading")
            uploader.delete_asset(existing_asset["id"], manifest_filename)
            uploader.upload_asset(release_id, upload_url, manifest_path, manifest_filename)
            uploaded += 1
    else:
        print(f"    Uploading manifest")
        uploader.upload_asset(release_id, upload_url, manifest_path, manifest_filename)
        uploaded += 1

    # Note: remote_manifest is already downloaded above during manifest check
    # We'll reuse it for hash comparison

    for filename, file_info in files.items():
        asset_name = file_info["uri"]
        local_hash = file_info["sha256"]
        local_path = artifacts_dir / asset_name

        if not local_path.exists():
            print(f"  Warning: File not found: {local_path}")
            continue

        # Check if asset already exists
        if asset_name in existing_assets and not force:
            existing_asset = existing_assets[asset_name]

            # Compare hashes from manifests (not actual file data)
            print(f"  Checking: {asset_name}")
            remote_hash = None
            if remote_manifest and "files" in remote_manifest:
                # Find the file in remote manifest
                for remote_filename, remote_file_info in remote_manifest["files"].items():
                    if remote_file_info.get("uri") == asset_name:
                        remote_hash = remote_file_info.get("sha256")
                        break

            if remote_hash and remote_hash == local_hash:
                print(f"    Manifest hash matches, skipping")
                skipped += 1
                continue
            else:
                if remote_hash:
                    print(f"    Manifest hash differs (local: {local_hash[:16]}..., remote: {remote_hash[:16]}...)")
                else:
                    print(f"    Remote hash not found in manifest, checking actual file hash")

                    remote_path = existing_asset["browser_download_url"]

                    actual_local_hash = hash_file_quick(local_path)
                    actual_remote_hash = None
                    try:
                        actual_remote_hash = hash_url_quick(remote_path)
                    except Exception as e:
                        print(f"    Error hashing remote file: {e}")

                    print(f"    Local file: {local_path} (hash: {actual_local_hash}...)")
                    print(f"    Remote file: {remote_path} (hash: {actual_remote_hash}...)")

                    if actual_local_hash == actual_remote_hash:
                        print(f"    Actual file hash matches, skipping")
                        skipped += 1
                        continue

                    print(f"    Actual file hash differs, uploading new version")

                # Delete old asset before uploading new one
                uploader.delete_asset(existing_asset["id"], asset_name)
        else:
            print(f"  Uploading asset: {asset_name}")

        # Upload the file
        uploader.upload_asset(release_id, upload_url, local_path, asset_name)
        uploaded += 1

    return uploaded, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Upload release artifacts for a specific target to GitHub releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag", required=True, help="GitHub release tag (e.g., v1.0.0)")
    parser.add_argument("--target", required=True, help="Target (e.g., linux-x64-cpu)")
    parser.add_argument(
        "--repo",
        default="easydiffusion/sdkit",
        help="GitHub repository in format owner/repo (default: easydiffusion/sdkit)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    parser.add_argument(
        "--force", action="store_true", help="Force upload all files, even if they already exist with matching hashes"
    )

    args = parser.parse_args()

    # Get GitHub token
    token = load_github_token()
    if not token:
        print("Error: GitHub token not provided.")
        print("Create a .creds.json file in the current working directory with:")
        print('  {"github_token": "your_github_token_here"}')
        print("Or set the GITHUB_TOKEN environment variable.")
        sys.exit(1)

    # Find the artifacts directory
    artifacts_dir = Path("build") / args.target / "release_artifacts"
    if not artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)

    # Find the manifest file
    manifest_files = list(artifacts_dir.glob("*-manifest.json"))
    if not manifest_files:
        print(f"Error: No manifest file found in {artifacts_dir}")
        sys.exit(1)

    manifest_path = manifest_files[0]
    manifest = load_manifest(artifacts_dir)
    if not manifest:
        sys.exit(1)

    # Initialize uploader
    uploader = GitHubReleaseUploader(args.repo, token, dry_run=args.dry_run)

    # Get release information
    print(f"Fetching release information for tag: {args.tag}")
    release = uploader.get_release_by_tag(args.tag)
    if not release:
        print(f"Error: Release '{args.tag}' not found in {args.repo}")
        print("Create the release first using GitHub web interface or API.")
        sys.exit(1)

    release_id = release["id"]
    upload_url = release["upload_url"]
    print(f"Release found: {release['name']} (ID: {release_id})")

    # Get existing assets
    existing_assets = uploader.get_release_assets(release_id)
    print(f"Existing assets: {len(existing_assets)}")

    # Upload
    uploaded, skipped = compare_and_upload(
        uploader, release_id, upload_url, artifacts_dir, manifest, manifest_path, existing_assets, args.force
    )

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Files uploaded: {uploaded}")
    print(f"  Files skipped (identical): {skipped}")
    if args.dry_run:
        print("\n  This was a DRY RUN - no files were actually uploaded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
