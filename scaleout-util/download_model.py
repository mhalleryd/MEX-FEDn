#!/usr/bin/env python3
"""Script to download a model using the Scaleout API client.

Usage:
    python download_model.py <model_id> <token>

Example:
    python download_model.py abc123 your_refresh_token
"""

import sys

from scaleoututil.api.client import APIClient


def main():
    """Download a model with the specified ID and token."""
    if len(sys.argv) != 3:
        print("Usage: python download_model.py <model_id> <token>")
        print("Example: python download_model.py abc123 your_refresh_token")
        sys.exit(1)

    model_id = sys.argv[1]
    token = sys.argv[2]
    output_path = f"model_{model_id}.bin"

    # Initialize API client and download
    print(f"Downloading model '{model_id}'...")
    client = APIClient(host="localhost", token=token)
    result = client.download_model(id=model_id, path=output_path)

    if result.get("success"):
        print(f"✓ Model saved to: {output_path}")
    else:
        print(f"✗ Failed to download model", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
