"""Long-lived local backend service that sits behind the ngrok tunnel.

Runs the Flask app on the threaded Werkzeug server with debug OFF (important:
debug exposes a remote code-execution console, which must never be reachable
through a public tunnel). Local development still uses backend/app.py.
"""

import os

from backend.app import create_app

app = create_app()

# Load the face model once at startup so the first request isn't slow.
try:
    from backend.vision.face_engine import warm_up
    warm_up()
    print("[serve_local] face engine warmed up", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[serve_local] warm-up skipped: {exc}", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"[serve_local] serving on 127.0.0.1:{port}", flush=True)
    # threaded=True so several students can be recognized at once; onnxruntime
    # releases the GIL during inference, so threads give real concurrency.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)
