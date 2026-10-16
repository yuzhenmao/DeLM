"""HTTPS CONNECT proxy for workers on isolated, internal Docker networks."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import select
import socket

ALLOWED_HOSTS = frozenset({"chatgpt.com", "auth.openai.com", "api.openai.com"})


def destination(authority, allowed_hosts=ALLOWED_HOSTS):
    host, separator, port = authority.rpartition(":")
    if separator != ":" or port != "443" or host.lower() not in allowed_hosts:
        raise ValueError("Only the configured model/authentication hosts on port 443 are allowed")
    return host.lower(), 443


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_CONNECT(self):
        try:
            target = destination(self.path, self.server.allowed_hosts)
        except ValueError:
            self.send_error(403, "Destination is not allowed")
            return
        try:
            upstream = socket.create_connection(target, timeout=20)
        except OSError:
            self.send_error(502, "Model endpoint connection failed")
            return
        with upstream:
            self.send_response(200, "Connection Established")
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            connections = (self.connection, upstream)
            for connection in connections:
                connection.settimeout(30)
            try:
                while not self.server.stopping.is_set():
                    readable, _, _ = select.select(connections, [], [], 1)
                    for incoming in readable:
                        data = incoming.recv(65536)
                        if not data:
                            return
                        outgoing = upstream if incoming is self.connection else self.connection
                        outgoing.sendall(data)
            except (OSError, ValueError):
                return

    def do_GET(self):
        self.send_error(403, "Only HTTPS CONNECT is supported")


def proxy(address, allowed_hosts=ALLOWED_HOSTS):
    import threading
    server = ThreadingHTTPServer((address, 0), ProxyHandler)
    server.stopping = threading.Event()
    server.allowed_hosts = frozenset(allowed_hosts)
    return server


class RestrictedNetwork:
    """Model-only egress for a trial; the runner owns startup and cleanup."""

    def __init__(self, harness):
        import uuid
        self.name = "delm-restricted-" + uuid.uuid4().hex[:12]
        self.hosts = (ALLOWED_HOSTS if harness == "codex"
                      else frozenset({"api.anthropic.com"}))
        self.server = None

    def start(self):
        import json
        import subprocess
        import threading
        subprocess.run(["docker", "network", "create", "--internal", "--label",
                        "delm.workspace.research=true", self.name],
                       check=True, capture_output=True, timeout=30)
        info = json.loads(subprocess.check_output(
            ["docker", "network", "inspect", self.name], timeout=30))[0]
        self.gateway = info["IPAM"]["Config"][0]["Gateway"]
        self.server = proxy(self.gateway, self.hosts)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.gateway

    def environment(self):
        url = f"http://{self.gateway}:{self.server.server_port}"
        return dict(HTTPS_PROXY=url, HTTP_PROXY=url,
                    NO_PROXY=f"{self.gateway},localhost,127.0.0.1")

    def close(self):
        import subprocess
        try:
            if self.server:
                self.server.stopping.set()
                self.server.shutdown()
                self.server.server_close()
        finally:
            subprocess.run(["docker", "network", "rm", self.name],
                           check=True, capture_output=True, timeout=30)
