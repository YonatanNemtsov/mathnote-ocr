/**
 * Parse page logic — connects canvas, WebSocket, and UI.
 */

const drawer = new StrokeCanvas('drawCanvas', 'overlayCanvas');
let ws = null;
let allPartitions = [];
let currentPartitionIdx = 0;
let autoDetectTimer = null;

// ── WebSocket ───────────────────────────────────────────────────────

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('status').textContent = 'Connected';
    document.getElementById('status').className = 'status connected';
  };
  ws.onclose = () => {
    document.getElementById('status').textContent = 'Disconnected';
    document.getElementById('status').className = 'status disconnected';
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'result') displayResult(data);
    else if (data.type === 'error') console.error('Server error:', data.message);
  };
}

// ── Actions ─────────────────────────────────────────────────────────

function detect() {
  if (drawer.strokes.length === 0) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  ws.send(JSON.stringify({
    type: 'detect',
    strokes: drawer.strokes,
    debug: document.getElementById('debugCheck').checked,
    canvas_width: drawer.canvas.width,
    canvas_height: drawer.canvas.height,
    stroke_width: drawer.strokeWidth,
  }));
}

function clearAll() {
  drawer.clear();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'clear' }));
  }
  document.getElementById('rendered').innerHTML = '<span class="placeholder">Draw and press Enter</span>';
  document.getElementById('latexSource').innerHTML = '&mdash;';
  document.getElementById('symbolCards').innerHTML = '<div class="no-result">Draw and press Detect</div>';
  document.getElementById('relationsPanel').innerHTML = '<div class="no-result">No relations yet</div>';
  document.getElementById('timingInfo').textContent = '';
  const gc = document.getElementById('graphCanvas').getContext('2d');
  gc.clearRect(0, 0, 300, 200);
  allPartitions = [];
}

// ── Auto-detect on stroke end ───────────────────────────────────────

drawer.onStrokeEnd = () => {
  clearTimeout(autoDetectTimer);
  autoDetectTimer = setTimeout(detect, 50);
};

// ── Display results ─────────────────────────────────────────────────

function escHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function displayResult(data) {
  allPartitions = data.partitions || [];
  currentPartitionIdx = 0;

  const nav = document.getElementById('partitionNav');
  nav.style.display = allPartitions.length > 1 ? 'flex' : 'none';

  // Timing
  if (data.timing) {
    document.getElementById('timingInfo').textContent =
      `grouper: ${data.timing.grouper_ms}ms | tree: ${data.timing.tree_ms}ms`;
  }

  if (allPartitions.length === 0) showEmptyResult();
  else showPartition(0);
}

function showEmptyResult() {
  document.getElementById('latexSource').textContent = '(empty)';
  document.getElementById('rendered').innerHTML = '<span class="placeholder">No expression detected</span>';
  document.getElementById('symbolCards').innerHTML = '<div class="no-result">No symbols detected</div>';
  document.getElementById('relationsPanel').innerHTML = '<div class="no-result">No relations</div>';
  drawer.overlayCtx.clearRect(0, 0, drawer.overlay.width, drawer.overlay.height);
  const gc = document.getElementById('graphCanvas').getContext('2d');
  gc.clearRect(0, 0, 300, 200);
}

function showPartition(idx) {
  if (idx < 0 || idx >= allPartitions.length) return;
  currentPartitionIdx = idx;

  const part = allPartitions[idx];
  const latex = part.latex || '';
  const symbols = part.symbols || [];
  const relations = part.relations || [];

  // Nav
  document.getElementById('partitionLabel').textContent = (idx + 1) + ' / ' + allPartitions.length;
  document.getElementById('prevBtn').disabled = idx === 0;
  document.getElementById('nextBtn').disabled = idx === allPartitions.length - 1;

  // LaTeX source
  const confStr = part.score != null ? '  (' + (part.score * 100).toFixed(0) + '%)' : '';
  document.getElementById('latexSource').textContent = (latex || '(empty)') + confStr;

  // Rendered
  const renderedEl = document.getElementById('rendered');
  if (latex) {
    try {
      const wrapper = document.createElement('div');
      wrapper.className = 'katex-wrapper';
      katex.render(latex, wrapper, { throwOnError: false, displayMode: true });
      renderedEl.innerHTML = '';
      renderedEl.appendChild(wrapper);
      requestAnimationFrame(() => {
        const cW = renderedEl.clientWidth - 40;
        const cH = renderedEl.clientHeight - 40;
        const kW = wrapper.offsetWidth;
        const kH = wrapper.offsetHeight;
        const scale = Math.min(kW > cW ? cW / kW : 1, kH > cH ? cH / kH : 1);
        if (scale < 1) {
          wrapper.style.transform = 'scale(' + scale + ')';
          wrapper.style.transformOrigin = 'center center';
        }
      });
    } catch (e) {
      renderedEl.innerHTML = '<span class="error">' + escHtml(e.message) + '</span>';
    }
  } else {
    renderedEl.innerHTML = '<span class="placeholder">No expression detected</span>';
  }

  // Bounding boxes
  drawer.drawBboxes(symbols);

  // Symbol cards
  const cardsEl = document.getElementById('symbolCards');
  if (symbols.length === 0) {
    cardsEl.innerHTML = '<div class="no-result">No symbols detected</div>';
  } else {
    cardsEl.innerHTML = symbols.map((sym, i) => {
      const color = SYM_COLORS[i % SYM_COLORS.length];
      return `<div class="symbol-card" style="border-color:${color}">
        <div class="sym" style="color:${color}">${escHtml(sym.symbol)}</div>
        <div class="conf">${(sym.confidence * 100).toFixed(1)}%</div>
      </div>`;
    }).join('');
  }

  // Relations
  const relEl = document.getElementById('relationsPanel');
  if (relations.length === 0) {
    relEl.innerHTML = '<div class="no-result">No relations</div>';
  } else {
    relEl.innerHTML = relations.map(rel => {
      const childSym = symbols[rel.from] ? symbols[rel.from].symbol : '?';
      const parentSym = rel.to >= 0 && symbols[rel.to] ? symbols[rel.to].symbol : 'ROOT';
      const pct = (rel.prob * 100).toFixed(0) + '%';
      return `<div class="rel-row">
        <span class="rel-tag rel-${rel.type}">${rel.type}</span>
        ${escHtml(parentSym)} &rarr; ${escHtml(childSym)}
        <span style="color:#888;font-size:11px">${pct}</span>
      </div>`;
    }).join('');
  }

  drawRelationGraph('graphCanvas', symbols, relations);

  // Subsets (debug mode only)
  const subsetsSection = document.getElementById('subsetsSection');
  const subsetsPanel = document.getElementById('subsetsPanel');
  const subsets = part.subsets || [];
  if (subsets.length > 0) {
    subsetsSection.style.display = '';
    subsetsPanel.innerHTML =
      `<div class="subset-count">${subsets.length} subsets processed</div>` +
      `<div class="subset-list">` +
      subsets.map((sub, si) => {
        const EDGE_NAMES = ['num', 'den', 'sup', 'sub', 'sqrt', 'upper', 'lower', 'match'];
        const predsHtml = sub.predictions.map(p => {
          const edgeName = p.pred_edge >= 0 ? EDGE_NAMES[p.pred_edge] || '?' : 'root';
          const edgeColor = ET_COLORS[edgeName] || '#888';
          const seqSym = p.pred_seq >= 0 ? (symbols[p.pred_seq] ? symbols[p.pred_seq].symbol : '?') : null;
          const seqHtml = seqSym ? `<span class="seq">${escHtml(seqSym)}&rarr;${escHtml(p.symbol)}</span>` : '';
          return `<div class="subset-pred">
            <span class="sym">${escHtml(p.symbol)}</span>
            <span class="arrow">&rarr;</span>
            <span class="parent">${escHtml(p.pred_parent_sym)}</span>
            <span class="edge" style="background:${edgeColor}">${edgeName}</span>
            <span class="conf">${(p.parent_conf * 100).toFixed(0)}%</span>
            ${seqHtml}
          </div>`;
        }).join('');

        return `<div class="subset-card" data-subset-idx="${si}"
                     onmouseenter="highlightSubset(${si})"
                     onmouseleave="unhighlightSubset()">
          <div class="subset-header">Subset ${si + 1}: [${sub.symbols.join(', ')}]</div>
          <div class="subset-latex" id="subset-latex-${si}"></div>
          <div class="subset-preds">${predsHtml}</div>
        </div>`;
      }).join('') +
      `</div>`;
    // Render KaTeX in each subset card
    subsets.forEach((sub, si) => {
      const el = document.getElementById(`subset-latex-${si}`);
      if (el && sub.latex) {
        try {
          katex.render(sub.latex, el, { throwOnError: false, displayMode: false });
        } catch (e) {
          el.textContent = sub.latex;
        }
      } else if (el) {
        el.textContent = '(empty)';
        el.style.color = '#888';
      }
    });
  } else {
    subsetsSection.style.display = 'none';
    subsetsPanel.innerHTML = '';
  }
}

function highlightSubset(subsetIdx) {
  if (!allPartitions[currentPartitionIdx]) return;
  const part = allPartitions[currentPartitionIdx];
  const sub = (part.subsets || [])[subsetIdx];
  const symbols = part.symbols || [];
  if (!sub) return;

  // Redraw normal bboxes then highlight subset members
  drawer.drawBboxes(symbols);
  const oc = drawer.overlayCtx;
  for (const globalIdx of sub.indices) {
    const sym = symbols[globalIdx];
    if (!sym) continue;
    const b = sym.bbox;
    oc.strokeStyle = '#8b5cf6';
    oc.lineWidth = 4;
    oc.setLineDash([]);
    oc.strokeRect(b.x - 6, b.y - 6, b.w + 12, b.h + 12);

    // Fill with translucent purple
    oc.fillStyle = 'rgba(139, 92, 246, 0.15)';
    oc.fillRect(b.x - 6, b.y - 6, b.w + 12, b.h + 12);
  }
}

function unhighlightSubset() {
  if (!allPartitions[currentPartitionIdx]) return;
  drawer.drawBboxes(allPartitions[currentPartitionIdx].symbols || []);
}

function toggleSection(contentId, headerEl) {
  const content = document.getElementById(contentId);
  const toggle = headerEl.querySelector('.toggle');
  content.classList.toggle('collapsed');
  toggle.classList.toggle('collapsed');
}

// ── Config ──────────────────────────────────────────────────────────

async function loadConfigs() {
  try {
    const resp = await fetch('/api/configs');
    const data = await resp.json();
    const select = document.getElementById('configSelect');
    select.innerHTML = data.configs.map(name =>
      `<option value="${name}" ${name === data.current ? 'selected' : ''}>${name}</option>`
    ).join('');
  } catch (e) {
    console.error('Failed to load configs:', e);
  }
}

async function switchConfig(name) {
  try {
    const resp = await fetch(`/api/config/${name}`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      console.log('Switched to config:', data.config);
      // Re-detect with new config if there are strokes
      if (drawer.strokes.length > 0) detect();
    } else {
      alert('Failed to switch config: ' + data.message);
    }
  } catch (e) {
    alert('Failed to switch config: ' + e);
  }
}

// ── Event bindings ──────────────────────────────────────────────────

document.getElementById('detectBtn').addEventListener('click', detect);
document.getElementById('undoBtn').addEventListener('click', () => drawer.undo());
document.getElementById('clearBtn').addEventListener('click', clearAll);
document.getElementById('prevBtn').addEventListener('click', () => showPartition(currentPartitionIdx - 1));
document.getElementById('nextBtn').addEventListener('click', () => showPartition(currentPartitionIdx + 1));

document.getElementById('strokeSlider').addEventListener('input', (e) => {
  drawer.setStrokeWidth(parseFloat(e.target.value));
  document.getElementById('strokeVal').textContent = parseFloat(e.target.value).toFixed(1);
});

document.getElementById('configSelect').addEventListener('change', (e) => {
  switchConfig(e.target.value);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') detect();
  if (e.key === 'Escape') clearAll();
  if (e.key === 'z' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); drawer.undo(); }
  if (e.key === 'ArrowLeft' && allPartitions.length > 1) showPartition(currentPartitionIdx - 1);
  if (e.key === 'ArrowRight' && allPartitions.length > 1) showPartition(currentPartitionIdx + 1);
});

// ── Init ────────────────────────────────────────────────────────────

connect();
loadConfigs();
