"""Loopback-only GET/HEAD media server with three fixed, isolated roots.

Run --software-check without opening a socket. Run without flags only after
the final brake media directory exists; no worker or simulation is launched.
"""
from __future__ import annotations

import argparse
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
HOST, PORT = '127.0.0.1', 8771
ROUTES = {
    '/': ROOT/'ff_prior_balanced_delivery/final_brake_001',
    '/failed/': ROOT/'ff_prior_balanced_delivery/final_001',
    '/static/': ROOT/'artifacts/v7_1/balanced_restore_001/units/u00/visuals',
}


class InvalidMediaPath(ValueError):
    pass


class InvalidRange(ValueError):
    pass


def resolve_media_path(target: str, routes=None):
    """Return (resolved file path, optional slash redirect) within one root."""
    if len(target) > 8192:
        raise InvalidMediaPath('request target too long')
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith('/'):
        raise InvalidMediaPath('only local relative request paths are accepted')
    try:
        path = unquote(parsed.path, encoding='utf-8', errors='strict')
    except UnicodeError as exc:
        raise InvalidMediaPath('invalid path encoding') from exc
    if any(char in path for char in ('\\', '\x00', ':')):
        raise InvalidMediaPath('invalid path separator or alternate stream')
    if any(part in ('.', '..') for part in path.split('/')):
        raise InvalidMediaPath('dot path components are not accepted')
    roots = ROUTES if routes is None else routes
    for prefix in sorted(roots, key=len, reverse=True):
        if prefix != '/' and path == prefix[:-1]:
            suffix = ('?'+parsed.query) if parsed.query else ''
            return roots[prefix].resolve()/'index.html', prefix+suffix
        if not path.startswith(prefix):
            continue
        root = roots[prefix].resolve()
        relative = path[len(prefix):]
        candidate = (root/relative).resolve()
        if not candidate.is_relative_to(root):
            raise InvalidMediaPath('resolved path escaped its media root')
        if not relative or path.endswith('/') or candidate.is_dir():
            candidate = (candidate/'index.html').resolve()
            if not candidate.is_relative_to(root):
                raise InvalidMediaPath('index path escaped its media root')
        return candidate, None
    raise InvalidMediaPath('no media route')


def byte_range(header: str | None, size: int):
    """One standard byte range supports browser video seeking."""
    if header is None:
        return 0, size-1
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or size <= 0:
        raise InvalidRange('unsupported or empty range')
    begin, end = match.groups()
    if not begin:
        if not end or int(end) <= 0:
            raise InvalidRange('invalid suffix range')
        return max(0, size-int(end)), size-1
    first = int(begin)
    last = min(size-1, int(end)) if end else size-1
    if first >= size or last < first:
        raise InvalidRange('range not satisfiable')
    return first, last


class MediaHandler(BaseHTTPRequestHandler):
    server_version = 'BalancedMedia/1.0'

    def do_GET(self):
        self._send_file(head=False)

    def do_HEAD(self):
        self._send_file(head=True)

    def _method_not_allowed(self):
        self.send_response(405)
        self.send_header('Allow', 'GET, HEAD')
        self.send_header('Content-Length', '0')
        self.end_headers()

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_CONNECT = _method_not_allowed
    do_TRACE = _method_not_allowed

    def _send_file(self, *, head):
        try:
            path, redirect = resolve_media_path(self.path)
        except (InvalidMediaPath, OSError, ValueError):
            self.send_error(403, 'Media path is outside the allowed routes')
            return
        if redirect:
            self.send_response(308)
            self.send_header('Location', redirect)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        try:
            if not path.is_file():
                self.send_error(404, 'Media file not found')
                return
            stat = path.stat()
            stream = path.open('rb')
        except OSError:
            self.send_error(404, 'Media file not found')
            return
        with stream:
            etag = f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
            requested = self.headers.get('Range')
            if self.headers.get('If-Range') not in (None, etag):
                requested = None
            try:
                first, last = byte_range(requested, stat.st_size)
            except InvalidRange:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{stat.st_size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            size = max(0, last-first+1)
            self.send_response(206 if requested else 200)
            mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(size))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('ETag', etag)
            self.send_header('Last-Modified', formatdate(stat.st_mtime, usegmt=True))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if requested:
                self.send_header('Content-Range', f'bytes {first}-{last}/{stat.st_size}')
            self.end_headers()
            if head:
                return
            try:
                stream.seek(first)
                remaining = size
                while remaining:
                    block = stream.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass


def software_check():
    checks = []
    for url, prefix, name in (
        ('/', '/', 'index.html'),
        ('/index.html', '/', 'index.html'),
        ('/failed/overview.mp4', '/failed/', 'overview.mp4'),
        ('/static/index.html?unused=1', '/static/', 'index.html'),
        ('/failed', '/failed/', 'index.html'),
    ):
        path, redirect = resolve_media_path(url)
        assert path == ROUTES[prefix].resolve()/name
        if url == '/failed':
            assert redirect == '/failed/'
    checks.append('three_fixed_routes_and_slash_redirect')
    bad = ('../index.html', '/../../control.json', '/%2e%2e/control.json',
        '/static/%2E%2E/index.html', '/failed/../index.html',
        '/static/..%5cindex.html', '/%5c%5cserver/share',
        '/index.html%00.png', '/index.html:secret',
        'http://example.org/index.html', '//example.org/index.html', '/%FF')
    for url in bad:
        try:
            resolve_media_path(url)
        except (InvalidMediaPath, ValueError):
            continue
        raise AssertionError('unsafe path accepted: '+url)
    checks.append('encoded_traversal_windows_separators_streams_and_absolute_urls_rejected')
    for header, size, wanted in ((None, 100, (0,99)), ('bytes=10-19',100,(10,19)),
        ('bytes=90-',100,(90,99)), ('bytes=-12',100,(88,99)),
        ('bytes=90-999',100,(90,99)), (None,0,(0,-1))):
        assert byte_range(header, size) == wanted
    for header in ('bytes=100-', 'bytes=10-9', 'bytes=-0', 'bytes=-',
        'bytes=0-1,5-6', 'items=0-1'):
        try:
            byte_range(header, 100)
        except InvalidRange:
            continue
        raise AssertionError('invalid range accepted: '+header)
    checks.append('video_seek_range_boundaries_and_invalid_ranges')
    for method in ('POST','PUT','PATCH','DELETE','OPTIONS','CONNECT','TRACE'):
        assert getattr(MediaHandler, 'do_'+method) is MediaHandler._method_not_allowed
    checks.append('no_write_or_proxy_handlers')
    return dict(status='PASS', checks=checks, socket_opened=False,
        files_written=False, physics=False, host=HOST, port=PORT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--software-check', action='store_true')
    args = parser.parse_args()
    if args.software_check:
        print(json.dumps(software_check(), indent=2))
        return
    for root in ROUTES.values():
        if not root.is_dir() or not (root/'index.html').is_file():
            raise SystemExit('Media route has no completed index: '+str(root))
    print(json.dumps(dict(status='SERVING_READ_ONLY', host=HOST, port=PORT,
        routes={k:str(v) for k,v in ROUTES.items()})), flush=True)
    with ThreadingHTTPServer((HOST, PORT), MediaHandler) as server:
        server.daemon_threads = True
        server.serve_forever()


if __name__ == '__main__':
    main()
