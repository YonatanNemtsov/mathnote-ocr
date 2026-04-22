/**
 * Canvas drawing module for stroke input.
 *
 * Usage:
 *   const drawer = new StrokeCanvas('drawCanvas', 'overlayCanvas');
 *   drawer.onStrokeEnd = () => sendDetect(drawer.strokes);
 */

class StrokeCanvas {
  constructor(canvasId, overlayId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.overlay = document.getElementById(overlayId);
    this.overlayCtx = this.overlay.getContext('2d');

    this.strokes = [];
    this.strokeIds = [];
    this._nextStrokeId = 0;
    this.currentStroke = [];
    this.isDrawing = false;
    this.strokeWidth = 2.0;
    this.onStrokeEnd = null;

    this._initStyle();
    this._bindEvents();
    this.clear();
  }

  _initStyle() {
    this.ctx.strokeStyle = 'black';
    this.ctx.lineWidth = this.strokeWidth;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
  }

  setStrokeWidth(w) {
    this.strokeWidth = w;
    this.ctx.lineWidth = w;
  }

  _getPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
      t: Date.now(),
    };
  }

  _bindEvents() {
    const start = (e) => {
      e.preventDefault();
      this.isDrawing = true;
      this.currentStroke = [this._getPoint(e)];
    };
    const move = (e) => {
      if (!this.isDrawing) return;
      e.preventDefault();
      const pt = this._getPoint(e);
      this.currentStroke.push(pt);
      const prev = this.currentStroke[this.currentStroke.length - 2];
      this.ctx.beginPath();
      this.ctx.moveTo(prev.x, prev.y);
      this.ctx.lineTo(pt.x, pt.y);
      this.ctx.stroke();
    };
    const end = (e) => {
      if (!this.isDrawing) return;
      e.preventDefault();
      this.isDrawing = false;
      if (this.currentStroke.length > 0) {
        const id = this._nextStrokeId++;
        this.strokes.push(this.currentStroke);
        this.strokeIds.push(id);
        this.currentStroke = [];
        if (this.onStrokeEnd) this.onStrokeEnd(id);
      }
    };

    this.canvas.addEventListener('mousedown', start);
    this.canvas.addEventListener('mousemove', move);
    this.canvas.addEventListener('mouseup', end);
    this.canvas.addEventListener('mouseleave', end);
    this.canvas.addEventListener('touchstart', start, { passive: false });
    this.canvas.addEventListener('touchmove', move, { passive: false });
    this.canvas.addEventListener('touchend', end, { passive: false });
  }

  clear() {
    this.ctx.fillStyle = 'white';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this._initStyle();
    this.strokes = [];
    this.strokeIds = [];
    this._nextStrokeId = 0;
    this.currentStroke = [];
    this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);
  }

  undo() {
    if (this.strokes.length === 0) return;
    this.strokes.pop();
    const removedId = this.strokeIds.pop();
    this.redraw();
    this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    if (this.onStrokeRemove) this.onStrokeRemove(removedId);
  }

  redraw() {
    this.ctx.fillStyle = 'white';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this._initStyle();
    for (const stroke of this.strokes) {
      this.ctx.beginPath();
      for (let i = 0; i < stroke.length; i++) {
        if (i === 0) this.ctx.moveTo(stroke[i].x, stroke[i].y);
        else this.ctx.lineTo(stroke[i].x, stroke[i].y);
      }
      this.ctx.stroke();
    }
  }

  drawBboxes(symbols) {
    const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
    this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    for (let i = 0; i < symbols.length; i++) {
      const sym = symbols[i];
      const color = COLORS[i % COLORS.length];
      const b = sym.bbox;

      this.overlayCtx.strokeStyle = color;
      this.overlayCtx.lineWidth = 3;
      this.overlayCtx.setLineDash([6, 3]);
      this.overlayCtx.strokeRect(b.x - 4, b.y - 4, b.w + 8, b.h + 8);
      this.overlayCtx.setLineDash([]);

      this.overlayCtx.font = 'bold 18px monospace';
      this.overlayCtx.fillStyle = color;
      this.overlayCtx.fillText(sym.name || sym.symbol, b.x - 4, b.y - 10);
    }
  }
}
