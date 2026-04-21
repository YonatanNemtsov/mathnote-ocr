/**
 * Relation graph renderer.
 */

const ET_COLORS = {
  root: '#888', num: '#f59e0b', den: '#f97316', sup: '#ef4444',
  sub: '#10b981', sqrt: '#8b5cf6', upper: '#3b82f6', lower: '#06b6d4', match: '#ec4899',
};
const SYM_COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

function drawRelationGraph(canvasId, symbols, relations) {
  const gc = document.getElementById(canvasId).getContext('2d');
  const graphCanvas = document.getElementById(canvasId);
  const n = symbols.length;
  if (n === 0) { gc.clearRect(0, 0, graphCanvas.width, graphCanvas.height); return; }

  const childrenOf = new Map();
  childrenOf.set(-1, []);
  for (let i = 0; i < n; i++) childrenOf.set(i, []);
  for (const rel of relations) {
    if (childrenOf.has(rel.to)) {
      childrenOf.get(rel.to).push({ child: rel.from, type: rel.type, prob: rel.prob });
    }
  }

  const depth = new Array(n).fill(-1);
  const queue = [];
  const rootChildren = childrenOf.get(-1) || [];
  for (const rc of rootChildren) { depth[rc.child] = 1; queue.push(rc.child); }
  let qi = 0;
  while (qi < queue.length) {
    const cur = queue[qi++];
    for (const ch of childrenOf.get(cur) || []) {
      if (depth[ch.child] === -1) { depth[ch.child] = depth[cur] + 1; queue.push(ch.child); }
    }
  }
  for (let i = 0; i < n; i++) if (depth[i] === -1) depth[i] = 1;

  const maxDepth = Math.max(...depth);
  const nodeR = 16, levelH = 60, padX = 30, padTop = 25, padBot = 20;
  const H = padTop + (maxDepth + 1) * levelH + padBot;
  graphCanvas.height = Math.max(H, 100);
  const W = graphCanvas.width;
  gc.clearRect(0, 0, W, graphCanvas.height);

  const subtreeW = new Array(n).fill(1);
  for (let d = maxDepth; d >= 0; d--) {
    for (let i = 0; i < n; i++) {
      if (depth[i] !== d) continue;
      let w = 0;
      for (const ch of childrenOf.get(i) || []) w += subtreeW[ch.child];
      subtreeW[i] = Math.max(w, 1);
    }
  }
  let totalW = 0;
  for (const rc of rootChildren) totalW += subtreeW[rc.child];
  totalW = Math.max(totalW, 1);

  const nodeX = new Array(n).fill(0);
  const nodeY = new Array(n).fill(0);

  function layoutChildren(parentCh, xStart, xEnd) {
    let totalSW = 0;
    for (const ch of parentCh) totalSW += subtreeW[ch.child];
    if (totalSW === 0) return;
    let x = xStart;
    for (const ch of parentCh) {
      const frac = subtreeW[ch.child] / totalSW;
      const cxS = x, cxE = x + frac * (xEnd - xStart);
      nodeX[ch.child] = (cxS + cxE) / 2;
      nodeY[ch.child] = padTop + depth[ch.child] * levelH;
      layoutChildren(childrenOf.get(ch.child) || [], cxS, cxE);
      x = cxE;
    }
  }
  layoutChildren(rootChildren, padX, W - padX);

  const rootX = W / 2, rootY = padTop;

  // Draw edges
  for (const rel of relations) {
    const ci = rel.from, pi = rel.to;
    const color = ET_COLORS[rel.type] || '#888';
    const px = pi === -1 ? rootX : nodeX[pi];
    const py = pi === -1 ? rootY : nodeY[pi];
    const cx = nodeX[ci], cy = nodeY[ci];

    gc.strokeStyle = color;
    gc.lineWidth = 2;
    gc.globalAlpha = 0.9;
    gc.beginPath();
    gc.moveTo(px, py + nodeR);
    gc.lineTo(cx, cy - nodeR);
    gc.stroke();

    const ax = cx, ay = cy - nodeR;
    const ang = Math.atan2((cy - nodeR) - (py + nodeR), cx - px);
    const hl = 7, ha = 0.45;
    gc.beginPath();
    gc.moveTo(ax, ay);
    gc.lineTo(ax - hl * Math.cos(ang - ha), ay - hl * Math.sin(ang - ha));
    gc.moveTo(ax, ay);
    gc.lineTo(ax - hl * Math.cos(ang + ha), ay - hl * Math.sin(ang + ha));
    gc.stroke();
    gc.globalAlpha = 1.0;

    const lx = (px + cx) / 2 + (pi === -1 ? 0 : 8);
    const ly = (py + nodeR + cy - nodeR) / 2;
    const pct = rel.prob != null ? (rel.prob * 100).toFixed(0) + '%' : '';
    const label = (rel.type === 'root' ? '' : rel.type) + (pct ? (rel.type === 'root' ? '' : ' ') + pct : '');
    gc.font = 'bold 10px -apple-system, sans-serif';
    gc.textAlign = 'center';
    gc.textBaseline = 'middle';
    const tw = gc.measureText(label).width + 6;
    gc.fillStyle = '#111827';
    gc.fillRect(lx - tw / 2, ly - 7, tw, 14);
    gc.fillStyle = color;
    gc.fillText(label, lx, ly);
  }

  // ROOT node
  gc.fillStyle = '#374151';
  gc.strokeStyle = '#888';
  gc.lineWidth = 2;
  gc.beginPath();
  gc.arc(rootX, rootY, nodeR, 0, Math.PI * 2);
  gc.fill(); gc.stroke();
  gc.fillStyle = '#aaa';
  gc.font = 'bold 11px -apple-system, sans-serif';
  gc.textAlign = 'center';
  gc.textBaseline = 'middle';
  gc.fillText('ROOT', rootX, rootY);

  // Symbol nodes
  for (let i = 0; i < n; i++) {
    const x = nodeX[i], y = nodeY[i];
    gc.fillStyle = '#1e293b';
    gc.strokeStyle = SYM_COLORS[i % SYM_COLORS.length];
    gc.lineWidth = 2;
    gc.beginPath();
    gc.arc(x, y, nodeR, 0, Math.PI * 2);
    gc.fill(); gc.stroke();
    gc.fillStyle = '#fff';
    gc.font = 'bold 13px Courier New';
    gc.textAlign = 'center';
    gc.textBaseline = 'middle';
    gc.fillText((symbols[i].name || symbols[i].symbol).slice(0, 4), x, y);
  }
}
