"""Pinned Codex bootstrap with explicit API-key or ChatGPT authentication."""

import json
import os
from pathlib import Path
import subprocess

from runtime.config import CLI_VERSIONS, DRIVER_REVISION, EFFORT, check_authentication, check_driver

VERSION = CLI_VERSIONS["codex"]
KEY_PATH = "/run/secrets/openai_api_key"
AUTH_PATH = "/run/secrets/codex_auth.json"
CONFIG = ('forced_login_method = "api"\ncli_auth_credentials_store = "file"\n'
          f'model_provider = "openai"\nmodel_reasoning_effort = "{EFFORT}"\n')


def clean_environment(environment):
    for name in list(environment):
        if name.startswith(("OPENAI_", "CODEX_")) or name == "LD_LIBRARY_PATH":
            environment.pop(name)
    environment["HOME"] = "/root"
    environment["CODEX_CA_CERTIFICATE"] = "/opt/native/share/ca-certificates.crt"


def prepare_auth(home=Path("/root"), key_path=Path(KEY_PATH), authentication="api-key"):
    """Prepare private worker credentials without importing host configuration.

    Secrets never appear in argv or runtime records.
    """
    if check_authentication("codex", authentication) == "api-key":
        key = key_path.read_text().strip()
        if not key or any(c.isspace() for c in key) or key.startswith(("{", "[")):
            raise ValueError("Supply a raw single-line OpenAI API key, not an auth.json")
        expected = {"auth_mode": "apikey", "OPENAI_API_KEY": key}
        config = CONFIG
    else:
        expected = json.loads(key_path.read_text())
        tokens = expected.get("tokens") if isinstance(expected, dict) else None
        if (not isinstance(tokens, dict) or expected.get("auth_mode") != "chatgpt"
                or expected.get("OPENAI_API_KEY")
                or any(not isinstance(tokens.get(k), str) or not tokens[k]
                       for k in ("access_token", "refresh_token", "id_token", "account_id"))):
            raise ValueError("Supply a ChatGPT login auth.json containing subscription tokens")
        config = CONFIG.replace('forced_login_method = "api"', 'forced_login_method = "chatgpt"')
    directory = home / ".codex"
    if directory.is_symlink():
        raise ValueError("Codex home must not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    for path, content in ((directory / "auth.json", json.dumps(expected)),
                          (directory / "config.toml", config)):
        if path.is_symlink():
            raise ValueError("Unexpected symlink in Codex authentication/configuration")
        if path.exists():
            if path.read_text() != content or path.stat().st_mode & 0o077:
                raise ValueError("Codex home contains inherited or non-private credentials/settings")
        else:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as output:
                output.write(content)


def preflight(environment, home=Path("/root")):
    if not environment.get("AGENT_MODEL") or environment.get("AGENT_EFFORT") != EFFORT:
        raise ValueError("Codex requires an explicit model and xhigh effort")
    version = subprocess.check_output(["codex", "--version"], text=True, timeout=20).strip()
    if version != f"codex-cli {VERSION}":
        raise ValueError(f"Expected Codex {VERSION}, found {version}")
    check_driver()
    authentication = environment.get("AGENT_AUTHENTICATION", "api-key")
    prepare_auth(home, Path(AUTH_PATH if authentication == "chatgpt" else KEY_PATH), authentication)
    return dict(codex_version=VERSION, driver_revision=DRIVER_REVISION,
                authentication=authentication, subscription_credentials=authentication == "chatgpt",
                model=environment["AGENT_MODEL"], effort=EFFORT)


def validate_record(path, model):
    """Verify actual native session/version/model/effort, not just launch flags."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metas = [r["payload"] for r in records if r.get("type") == "session_meta"]
    turns = [r["payload"] for r in records if r.get("type") == "turn_context"]
    if (len(metas) != 1 or not metas[0].get("id") or metas[0].get("parent_thread_id")
            or metas[0].get("cli_version") != VERSION
            or metas[0].get("model_provider") != "openai"):
        raise ValueError("Expected one native root session with pinned Codex/OpenAI provider")
    if not turns or any((t.get("model"), t.get("effort")) != (model, EFFORT) for t in turns):
        raise ValueError("Native model/reasoning settings differ from the experiment")
    return metas[0]["id"]


def main():
    clean_environment(os.environ)
    preflight(os.environ)
    import native_worker
    native_worker.main()


if __name__ == "__main__":
    main()
