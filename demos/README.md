# Demos

Small apps that showcase the mathnote_ocr public API. Each demo uses only
`from mathnote_ocr import ...` — no internal imports. If a demo needs
something the public API doesn't expose, that might be a signal to extend the API.

## Demos

- [`web/`](web/) — FastAPI + WebSocket: draw math in the browser, get LaTeX.
