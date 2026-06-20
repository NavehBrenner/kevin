"""Serve a webcam as an MJPEG-over-HTTP stream — the WSL2 camera bridge.

WSL2's kernel has no UVC driver, so a host webcam never appears as a device
inside WSL. Run THIS script on **Windows** (where the webcam lives) and have the
WSL-side vision teleop open the stream by URL instead of a device index:

    # on Windows (needs: pip install opencv-python):
    python scripts/stream_webcam.py                 # serves http://0.0.0.0:8080/video

    # in WSL:
    uv run kvn episode --input vision --camera http://<windows-host>:8080/video

Finding ``<windows-host>`` from WSL:
- WSL "mirrored" networking (``networkingMode=mirrored`` in ``.wslconfig``):
  just use ``localhost``.
- Default NAT networking: it's the default-route gateway —
  ``ip route show default | awk '{print $3}'``.

First run pops a Windows Firewall prompt — allow it on private networks, or the
WSL side can't connect. Single viewer at a time is plenty for the demo.

Stdlib + OpenCV only (no project imports): it runs outside the repo venv on the
Windows host, so it logs with ``print`` rather than the project logger.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_BOUNDARY = "frame"


def _make_handler(camera: int, jpeg_quality: int) -> type[BaseHTTPRequestHandler]:
    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            # Reachability/firewall pre-flight check (`curl -I`): headers only, no
            # body, no camera open — answers "can the client reach me?" with 200.
            if self.path not in ("/", "/video"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path not in ("/", "/video"):
                self.send_error(404)
                return

            capture = cv2.VideoCapture(camera)
            if not capture.isOpened():
                self.send_error(500, f"could not open camera {camera}")
                return

            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
            self.end_headers()
            print(f"client connected: {self.client_address[0]}")
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
                    if not ok:
                        continue
                    payload = jpeg.tobytes()
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                print(f"client disconnected: {self.client_address[0]}")
            finally:
                capture.release()

        def log_message(self, *args: object) -> None:
            pass  # quiet the default per-request access log

    return MJPEGHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Windows camera index (default 0).")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default 8080).")
    parser.add_argument(
        "--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default 80)."
    )
    args = parser.parse_args()

    handler = _make_handler(args.camera, args.jpeg_quality)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"serving camera {args.camera} at http://0.0.0.0:{args.port}/video  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
