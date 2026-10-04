#!/usr/bin/env python3
"""Serve a bounded CPU tokenization stream to the single router coordinator."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--port', type=int, default=19190)
    p.add_argument('--log', required=True)
    p.add_argument('--exclude-groups')
    a = p.parse_args()
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200); self.end_headers(); self.wfile.write(b'ready'); return
            if self.path != '/stream':
                self.send_error(404); return
            if not lock.acquire(blocking=False):
                self.send_error(409, 'one active training consumer supported'); return
            process = None
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                with Path(a.log).open('a') as log:
                    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('stream_router_raw.py')),
                        '--raw-root', a.raw_root, '--tokenizer', a.tokenizer] +
                        (['--exclude-groups', a.exclude_groups] if a.exclude_groups else []), stdout=subprocess.PIPE, stderr=log)
                    for line in process.stdout:
                        self.wfile.write(line); self.wfile.flush()
                    process.wait()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if process is not None and process.poll() is None:
                    process.terminate(); process.wait(timeout=15)
                lock.release()
    ThreadingHTTPServer(('0.0.0.0', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
