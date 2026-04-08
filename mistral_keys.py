"""
Mistral API key helper.

Supports multiple environment key names such as:
- MISTRAL_API_KEY
- MISTRAL_API_KEY_1
- MISTRAL_API_KEY_2
- ...

This allows the project to use a primary key and keep fallback keys available.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

_API_KEY_PREFIX = "MISTRAL_API_KEY"


def _env_key_names() -> List[str]:
    names: List[str] = []
    if _API_KEY_PREFIX in os.environ:
        names.append(_API_KEY_PREFIX)

    for index in range(1, 11):
        env_name = f"{_API_KEY_PREFIX}_{index}"
        if env_name in os.environ:
            names.append(env_name)

    return names


def get_mistral_api_keys() -> List[str]:
    """Return all configured Mistral API keys from the environment."""
    keys = [os.environ[name].strip() for name in _env_key_names() if os.environ[name].strip()]
    return keys


def get_mistral_api_key() -> Optional[str]:
    """Return the first available Mistral API key."""
    keys = get_mistral_api_keys()
    return keys[0] if keys else None
