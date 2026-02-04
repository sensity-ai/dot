#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

Model downloader utilities for diffusion-based face swap models.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dot", "diffusion")


def get_cache_dir() -> str:
    """Get the cache directory for diffusion models."""
    cache_dir = os.environ.get("DOT_CACHE_DIR", DEFAULT_CACHE_DIR)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model_file(
    repo_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download a single file from HuggingFace Hub.

    Args:
        repo_id: The HuggingFace repository ID (e.g., "InstantX/InstantID")
        filename: The filename to download
        subfolder: Optional subfolder within the repository
        cache_dir: Optional cache directory (defaults to ~/.cache/dot/diffusion)
        token: Optional HuggingFace token for private repos

    Returns:
        str: Path to the downloaded file
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for model downloads. "
            "Install with: pip install huggingface-hub"
        )

    if cache_dir is None:
        cache_dir = get_cache_dir()

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir=cache_dir,
        token=token,
    )


def download_model_repo(
    repo_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[list] = None,
    ignore_patterns: Optional[list] = None,
) -> str:
    """
    Download an entire model repository from HuggingFace Hub.

    Args:
        repo_id: The HuggingFace repository ID
        cache_dir: Optional cache directory (defaults to ~/.cache/dot/diffusion)
        token: Optional HuggingFace token for private repos
        allow_patterns: Optional list of patterns to include
        ignore_patterns: Optional list of patterns to exclude

    Returns:
        str: Path to the downloaded repository
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for model downloads. "
            "Install with: pip install huggingface-hub"
        )

    if cache_dir is None:
        cache_dir = get_cache_dir()

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def check_model_exists(model_path: str) -> bool:
    """Check if a model file or directory exists."""
    return os.path.exists(model_path)
