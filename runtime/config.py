"""Pinned runtime versions, launch defaults, and their checks for both harnesses.

Also mounted into each worker container, so it imports only the standard library.
"""

import json

DRIVER_REVISION = "413d02e44d0cc0514b9f5bd3fcefea156b047a49"
CLI_VERSIONS = {"codex": "0.153.4", "claude-code": "2.1.280"}
DEFAULT_MODELS = {"codex": "gpt-6-astra", "claude-code": "claude-opus-5-5"}
DEFAULT_COUNT = 2
EFFORT = "xhigh"
AUTHENTICATION = {"codex": ("api-key", "chatgpt"), "claude-code": ("api-key", "claude-oauth")}
HARNESS_NAMES = {"codex": "Codex", "claude-code": "Claude"}


def check_authentication(harness, authentication):
    if authentication not in AUTHENTICATION[harness]:
        raise ValueError(f"Unsupported {HARNESS_NAMES[harness]} authentication method")
    return authentication


def check_driver():
    from importlib.metadata import distribution
    direct = json.loads(distribution("hmz").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != DRIVER_REVISION:
        raise ValueError("Humanize does not match the pinned driver revision")
