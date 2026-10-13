"""
Publish a scheduled Reel to Instagram through the Instagram Graph API.

    uv run python -m tools.post_reel --check              # token works? which account?
    uv run python -m tools.post_reel 2026-10-14 --dry-run # show caption + video checks only
    uv run python -m tools.post_reel 2026-10-14           # upload, publish, mark row `posted`
    uv run python -m tools.post_reel --refresh-token      # extend the 60-day token in .env

The row's date picks everything: the script `lab/<YYYY>/<MMDD>.py` (its module docstring is
the caption), the video `outputs/<YYYYMMDD>/<YYYYMMDD>.mp4`, and the `SCHEDULE.md` row whose
status becomes `posted`.

The Instagram-login API only takes a `video_url`, so while a post is processing the MP4 is
served from this PC through a Cloudflare quick tunnel (`cloudflared`, no account needed) at a
random, unguessable https URL. The tunnel closes as soon as Instagram has fetched the video.

Credentials live in `.env` at the repo root (gitignored):

    IG_ACCESS_TOKEN=...      # long-lived token, scope instagram_business_content_publish
    IG_USER_ID=...           # optional; looked up from the token when missing
    IG_API_VERSION=v23.0     # optional
"""

import argparse
import ast
import datetime as dt
import json
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
SCHEDULE_PATH = ROOT / "SCHEDULE.md"
GRAPH = "https://graph.instagram.com"

READY = "music ok"
MAX_CAPTION = 2200
MAX_HASHTAGS = 30
MAX_BYTES = 300 * 1024**2


class GraphError(RuntimeError):
    pass


# --- config -------------------------------------------------------------------------------


def read_env():
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip("'\"")
    return env


def write_env_value(key, value):
    lines = (
        ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    )
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- HTTP ---------------------------------------------------------------------------------


def request(method, url, params=None, data=None, headers=None, timeout=60):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            err = json.loads(body).get("error", {})
            body = f"{err.get('message')} (code {err.get('code')}, {err.get('type')})"
        except (json.JSONDecodeError, AttributeError):
            pass
        raise GraphError(
            f"{method} {url.split('?')[0]} -> HTTP {e.code}: {body}"
        ) from None


class Instagram:
    def __init__(self, token, version, user_id=None):
        self.token = token
        self.base = f"{GRAPH}/{version}"
        self._user_id = user_id

    def get(self, path, **params):
        return request(
            "GET", f"{self.base}/{path}", {**params, "access_token": self.token}
        )

    def post(self, path, **params):
        body = urllib.parse.urlencode({**params, "access_token": self.token})
        return request("POST", f"{self.base}/{path}", data=body.encode())

    def me(self):
        return self.get("me", fields="user_id,username,account_type")

    @property
    def user_id(self):
        if not self._user_id:
            self._user_id = self.me()["user_id"]
        return self._user_id

    def publish_reel(self, video: Path, caption: str):
        with tunnel(video) as video_url:
            container = self.post(
                f"{self.user_id}/media",
                media_type="REELS",
                video_url=video_url,
                caption=caption,
                share_to_feed="true",
            )
            cid = container["id"]
            logger.info(f"Container {cid} created, Instagram is fetching the video")
            self.wait_until_finished(cid)

        media_id = self.post(f"{self.user_id}/media_publish", creation_id=cid)["id"]
        permalink = self.get(media_id, fields="permalink").get("permalink")
        return media_id, permalink

    def wait_until_finished(self, cid):
        deadline = time.monotonic() + 15 * 60
        while True:
            state = self.get(cid, fields="status_code,status")
            code = state.get("status_code")
            if code == "FINISHED":
                break
            if code in ("ERROR", "EXPIRED"):
                raise GraphError(f"Instagram could not process the video: {state}")
            if time.monotonic() > deadline:
                raise GraphError(f"Processing timed out, last status: {state}")
            logger.info(f"Processing on Instagram's side: {code}")
            time.sleep(10)

    def refresh_token(self):
        return request(
            "GET",
            f"{GRAPH}/refresh_access_token",
            {"grant_type": "ig_refresh_token", "access_token": self.token},
        )


# --- temporary public URL for the video -------------------------------------------------


def file_handler(video: Path, secret: str):
    """Serve only `video`, only at /<secret>/<name>, with Range support."""
    size = video.stat().st_size
    route = f"/{secret}/{video.name}"

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(body=False)

        def do_GET(self):
            self.respond(body=True)

        def respond(self, body):
            if self.path.split("?")[0] != route:
                self.send_error(404)
                return
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m[1] or m[2]):
                if m[1]:
                    start, end = int(m[1]), int(m[2]) if m[2] else size - 1
                else:
                    start = size - int(m[2])
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if body:
                with video.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0 and (chunk := f.read(min(1 << 20, left))):
                        self.wfile.write(chunk)
                        left -= len(chunk)

        def log_message(self, fmt, *args):
            logger.debug(f"video server: {fmt % args}")

    return Handler, route


def public_head(host, path):
    """HEAD https://host/path, resolving `host` with Cloudflare DNS-over-HTTPS."""
    answer = request(
        "GET",
        "https://cloudflare-dns.com/dns-query",
        {"name": host, "type": "A"},
        headers={"accept": "application/dns-json"},
        timeout=10,
    )
    ips = [a["data"] for a in answer.get("Answer", []) if a.get("type") == 1]
    if not ips:
        raise ValueError(f"{host} not in public DNS yet")
    ip = ips[0]
    ctx = ssl.create_default_context()
    with socket.create_connection((ip, 443), timeout=10) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as tls:
            tls.sendall(
                f"HEAD {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
            )
            status_line = tls.recv(4096).split(b"\r\n", 1)[0].decode()
    return int(status_line.split()[1])


class tunnel:
    """Context manager: a public https URL for `video`, closed on exit."""

    def __init__(self, video: Path):
        self.video = video

    def __enter__(self):
        try:
            return self._open()
        except BaseException:
            self.__exit__()
            raise

    def _open(self):
        exe = shutil.which("cloudflared") or shutil.which(
            r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
        )
        if not exe:
            raise SystemExit(
                "cloudflared not found: winget install --id Cloudflare.cloudflared -e"
            )
        handler, route = file_handler(self.video, secrets.token_urlsafe(24))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        port = self.server.server_address[1]

        self.proc = subprocess.Popen(
            [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        found = {}

        def scan():  # cloudflared prints the URL to stderr; keep draining it after
            for line in self.proc.stderr:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and "url" not in found:
                    found["url"] = m[0]

        threading.Thread(target=scan, daemon=True).start()
        deadline = time.monotonic() + 60
        while "url" not in found:
            if self.proc.poll() is not None or time.monotonic() > deadline:
                raise SystemExit("cloudflared did not produce a tunnel URL")
            time.sleep(0.5)
        url = found["url"] + route

        # The hostname takes a few seconds to exist. Check through public DNS: Windows
        # caches the "not found" from its first lookup, but Instagram's servers won't.
        host = urllib.parse.urlsplit(url).hostname
        deadline = time.monotonic() + 90
        while True:
            try:
                if public_head(host, urllib.parse.urlsplit(url).path) == 200:
                    break
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
            if time.monotonic() > deadline:
                raise SystemExit("Tunnel URL never became reachable")
            time.sleep(2)
        logger.info(f"Serving {self.video.name} at {found['url']}/...")
        return url

    def __exit__(self, *exc):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "server", None):
            self.server.shutdown()
            self.server.server_close()
        logger.info("Tunnel closed")


# --- schedule, caption, video -------------------------------------------------------------


def schedule_row(date: dt.date):
    """Return (line index, cells) of the SCHEDULE.md row for `date`."""
    lines = SCHEDULE_PATH.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"| {date.isoformat()} |"):
            return i, [c.strip() for c in line.strip().strip("|").split("|")]
    raise SystemExit(f"No row for {date} in SCHEDULE.md")


def set_status(date: dt.date, status: str):
    i, cells = schedule_row(date)
    cells[-1] = status
    lines = SCHEDULE_PATH.read_text(encoding="utf-8").splitlines()
    lines[i] = "| " + " | ".join(cells) + " |"
    SCHEDULE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def caption_for(script: Path) -> str:
    """The module docstring with hard line wraps joined, paragraph breaks kept."""
    doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
    if not doc:
        raise SystemExit(f"{script} has no module docstring (the caption)")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", doc.strip())]
    # Keep the "Made with" line and the hashtags as separate lines in the last block.
    tail = paragraphs[-1]
    tail = re.sub(r"\s+(?=#)", " ", tail)
    tail = re.sub(r"(\.)\s+(#)", r"\1\n\2", tail, count=1)
    paragraphs[-1] = tail
    return "\n\n".join(paragraphs)


def check_caption(caption: str):
    problems = []
    if len(caption) > MAX_CAPTION:
        problems.append(f"caption is {len(caption)} chars (max {MAX_CAPTION})")
    tags = re.findall(r"#\w+", caption)
    if len(tags) > MAX_HASHTAGS:
        problems.append(f"{len(tags)} hashtags (max {MAX_HASHTAGS})")
    return problems


def probe(video: Path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-of", "json",
            "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate",
            "-show_entries", "format=duration,size",
            str(video),
        ],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    info = json.loads(out.stdout)
    streams = {s["codec_type"]: s for s in info["streams"]}
    v, a = streams.get("video", {}), streams.get("audio")
    duration = float(info["format"]["duration"])
    size = int(info["format"]["size"])
    summary = (
        f"{v.get('codec_name')} {v.get('width')}x{v.get('height')} "
        f"@ {v.get('r_frame_rate')}, {duration:.2f} s, {size / 1e6:.1f} MB, "
        f"audio: {a['codec_name'] if a else 'none'}"
    )
    problems = []
    if v.get("codec_name") not in ("h264", "hevc"):
        problems.append(f"video codec {v.get('codec_name')} (needs h264/hevc)")
    if (v.get("width"), v.get("height")) != (1080, 1920):
        problems.append("not 1080x1920")
    if not 3 <= duration <= 900:
        problems.append(f"duration {duration:.1f} s outside 3 s - 15 min")
    if size > MAX_BYTES:
        problems.append(f"file is {size / 1e6:.0f} MB (max {MAX_BYTES / 1e6:.0f})")
    if a and a["codec_name"] != "aac":
        problems.append(f"audio codec {a['codec_name']} (needs aac)")
    return summary, problems


# --- main ---------------------------------------------------------------------------------


def client(env):
    token = env.get("IG_ACCESS_TOKEN")
    if not token:
        raise SystemExit(f"IG_ACCESS_TOKEN missing from {ENV_PATH}")
    return Instagram(token, env.get("IG_API_VERSION", "v23.0"), env.get("IG_USER_ID"))


@logger.catch(reraise=False, onerror=lambda _: sys.exit(1))
def main():
    parser = argparse.ArgumentParser(
        description="Publish a scheduled Reel to Instagram."
    )
    parser.add_argument(
        "date", nargs="?", help="posting date YYYY-MM-DD (default today)"
    )
    parser.add_argument("--dry-run", action="store_true", help="checks only, no upload")
    parser.add_argument(
        "--force", action="store_true", help=f"post even if not '{READY}'"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify the token and exit"
    )
    parser.add_argument("--refresh-token", action="store_true", help="extend the token")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    env = read_env()

    if args.check:
        me = client(env).me()
        logger.success(f"Token OK: @{me['username']} ({me.get('account_type')}), "
                       f"user id {me['user_id']}")  # fmt: skip
        return
    if args.refresh_token:
        result = client(env).refresh_token()
        write_env_value("IG_ACCESS_TOKEN", result["access_token"])
        days = int(result.get("expires_in", 0)) / 86400
        logger.success(f"Token refreshed, valid for {days:.0f} more days")
        return

    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    _, cells = schedule_row(date)
    title, status = cells[-2], cells[-1]
    script = ROOT / cells[-3].strip("`")
    video = ROOT / "outputs" / date.strftime("%Y%m%d") / f"{date:%Y%m%d}.mp4"
    logger.info(f"{date} {title or '(untitled)'}: status '{status}'")

    if status == "posted":
        raise SystemExit(f"{date} is already marked posted")
    if status != READY and not args.force:
        raise SystemExit(
            f"{date} is '{status}', not '{READY}' (use --force to post anyway)"
        )
    if not video.exists():
        raise SystemExit(f"Missing {video}: run the full render first")

    caption = caption_for(script)
    summary, problems = probe(video)
    problems += check_caption(caption)
    print(f"\n----- caption ({len(caption)} chars) -----\n{caption}\n{'-' * 30}")
    logger.info(f"Video: {video.relative_to(ROOT)} | {summary}")
    if problems:
        raise SystemExit("Not postable: " + "; ".join(problems))
    if args.dry_run:
        logger.success("Dry run OK, nothing uploaded")
        return

    media_id, permalink = client(env).publish_reel(video, caption)
    set_status(date, "posted")
    logger.success(
        f"Published {media_id}: {permalink}  (SCHEDULE.md row set to posted)"
    )


if __name__ == "__main__":
    main()
