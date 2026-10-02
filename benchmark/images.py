"""Bind locally built task images to the verified task source."""

import hashlib
import json
import subprocess

SOURCE_LABEL = "delm.task.source"


def source_digest(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def source_label(source):
    return f"{SOURCE_LABEL}={source_digest(source)}"


def inspect_image(image):
    return json.loads(subprocess.check_output(
        ["docker", "image", "inspect", image], timeout=30))[0]


def validate_source(image, source):
    if (image["Config"].get("Labels") or {}).get(SOURCE_LABEL) != source_digest(source):
        raise ValueError("Task image source is missing or outdated; rebuild with build_task.py")
