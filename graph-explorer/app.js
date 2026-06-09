/* ===========================================================================
   GANDALF Explorer — frontend logic (vanilla JS, no build step).

   Mental model
   ------------
   A traversal is a `path`: a list of "steps". Each step has a root node and,
   once the user drills in, a chosen relationship (predicate + direction) and a
   chosen neighbour. The chosen neighbour becomes the root of the next step.

   The screen shows ONE step at a time as a panel of three columns:
       [ root node ] [ relationships ] [ connected nodes ]
   Panels live side by side in a track; navigating slides the track left/right.
   The top filmstrip shows the whole path with a frame around the current triple.
   ========================================================================= */

'use strict';

const CHUNK = 40;        // neighbours rendered per scroll chunk
const DEGREE_CAP = 300;  // max neighbours we fetch degrees for per relationship

const EXAMPLES = [
  { name: 'Metformin',         id: 'CHEBI:6801',      kind: 'Drug' },
  { name: 'Aspirin',           id: 'CHEBI:15365',     kind: 'Drug' },
  { name: 'Caffeine',          id: 'CHEBI:27732',     kind: 'Chemical' },
  { name: 'TP53',              id: 'NCBIGene:7157',   kind: 'Gene' },
  { name: 'EGFR',              id: 'NCBIGene:1956',   kind: 'Gene' },
  { name: 'Type 2 diabetes',   id: 'MONDO:0005148',   kind: 'Disease' },
  { name: 'Asthma',            id: 'MONDO:0004979',   kind: 'Disease' },
  { name: 'Seizure',           id: 'HP:0001250',      kind: 'Phenotype' },
];

// ---- state ---------------------------------------------------------------
/** @type {Array<Step>} */
let path = [];
let current = 0;

const degreeCache = new Map();   // curie -> degree|null
const expandCache = new Map();   // curie -> raw expand result

const panelEls = [];             // DOM panel per step index

// ---- dom refs ------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const track = $('#track');
const stage = $('#stage');
const pathEl = $('#path');
const tripleFrame = $('#tripleFrame');
const overlay = $('#overlay');
const prevBtn = $('#prevBtn');
const nextBtn = $('#nextBtn');
const stepReadout = $('#stepReadout');

// ---- small helpers -------------------------------------------------------
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const predLabel = (p) => String(p || '').replace(/^biolink:/, '').replace(/_/g, ' ');
const catLabel = (c) => String(c || '').replace(/^biolink:/, '');
const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString());
const arrow = (dir) => (dir === 'in' ? '←' : '→');

// =========================================================================
// Start screen
// =========================================================================
function buildExamples() {
  const grid = $('#examples');
  grid.innerHTML = '';
  for (const ex of EXAMPLES) {
    const b = el('button', 'example');
    b.innerHTML =
      `<div class="ex-name">${esc(ex.name)}</div>` +
      `<div class="ex-meta"><span class="ex-id">${esc(ex.id)}</span>` +
      `<span class="ex-kind">${esc(ex.kind)}</span></div>`;
    b.addEventListener('click', () => startFrom(ex.id, ex.name, [`biolink:${ex.kind}`]));
    grid.appendChild(b);
  }
}

function validCurie(s) { return /^[A-Za-z0-9.\-]+:[A-Za-z0-9.\-_:]+$/.test(s); }

function wireStart() {
  const input = $('#curieInput');
  const err = $('#curieError');
  const go = () => {
    const v = input.value.trim();
    if (!v) { err.textContent = 'Enter a CURIE to begin.'; return; }
    if (!validCurie(v)) { err.textContent = 'That does not look like a CURIE (expected PREFIX:identifier).'; return; }
    err.textContent = '';
    startFrom(v, v, []);
  };
  $('#curieGo').addEventListener('click', go);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  $('#newStart').addEventListener('click', () => { overlay.hidden = false; input.value = ''; err.textContent = ''; input.focus(); });
}

// =========================================================================
// Traversal control
// =========================================================================
function startFrom(curie, name, categories) {
  overlay.hidden = true;
  path = [makeStep(curie, name, categories)];
  current = 0;
  // wipe old panels
  for (const p of panelEls) p && p.remove();
  panelEls.length = 0;
  reconcilePanels();
  setCurrent(0);
  loadStep(0);
}

function makeStep(curie, name, categories) {
  return {
    curie,
    name: name || curie,
    categories: categories || [],
    status: 'idle',        // idle | loading | loaded | error
    error: null,
    groups: null,          // [{key,predicate,direction,count,neighbors,_rendered,_degreesLoaded,_maxDegree,_note}]
    selectedKey: null,
  };
}

async function loadStep(i) {
  const step = path[i];
  if (!step || step.status === 'loaded' || step.status === 'loading') {
    if (step && step.status === 'loaded') updatePanelPreds(i);
    return;
  }
  step.status = 'loading';
  updatePanelPreds(i);

  if (expandCache.has(step.curie)) {
    applyExpand(step, expandCache.get(step.curie));
    finishLoad(i);
    return;
  }
  try {
    const r = await fetch(`/api/expand?curie=${encodeURIComponent(step.curie)}`);
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    expandCache.set(step.curie, data);
    applyExpand(step, data);
    finishLoad(i);
  } catch (e) {
    step.status = 'error';
    step.error = String(e.message || e);
    updatePanelNode(i);
    updatePanelPreds(i);
  }
}

function applyExpand(step, data) {
  if (data.name) step.name = data.name;
  if (data.categories && data.categories.length) step.categories = data.categories;
  step.serverErrors = data.errors || [];
  step.groups = (data.groups || []).map((g) => ({
    ...g,
    _rendered: 0,
    _degreesLoaded: false,
    _degreeLoading: false,
    _maxDegree: 0,
    _note: '',
  }));
  step.status = 'loaded';
}

function finishLoad(i) {
  updatePanelNode(i);
  updatePanelPreds(i);
  if (i === current) drawConnectors(i);
}

function selectPredicate(i, key) {
  const step = path[i];
  if (!step || !step.groups) return;
  step.selectedKey = key;
  // truncate any forward path that was built from a different choice
  if (i < path.length - 1) {
    truncatePathTo(i);
  }
  updatePanelPreds(i);
  updatePanelNeighbors(i);
  drawConnectors(i);
  loadDegrees(i, key);
}

function stepInto(i, neighbor) {
  // record forward path; if diverging, truncate
  truncatePathTo(i);
  const next = makeStep(neighbor.id, neighbor.name, neighbor.categories);
  path.push(next);
  reconcilePanels();
  renderBreadcrumb();
  setCurrent(i + 1);
  loadStep(i + 1);
}

function truncatePathTo(i) {
  if (path.length > i + 1) {
    const removed = panelEls.splice(i + 1);
    for (const p of removed) p && p.remove();
    path.length = i + 1;
  }
}

// =========================================================================
// Degrees
// =========================================================================
async function loadDegrees(i, key) {
  const step = path[i];
  const group = step.groups.find((g) => g.key === key);
  if (!group || group._degreesLoaded || group._degreeLoading) {
    if (group && group._degreesLoaded) renderNeighborStatus(i, group);
    return;
  }
  group._degreeLoading = true;
  renderNeighborStatus(i, group);

  const slice = group.neighbors.slice(0, DEGREE_CAP);
  const need = slice.map((n) => n.id).filter((id) => !degreeCache.has(id));

  try {
    if (need.length) {
      const r = await fetch('/api/degrees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ curies: need }),
      });
      const data = await r.json();
      const map = (data && data.degrees) || {};
      for (const id of need) degreeCache.set(id, map[id] != null ? map[id] : null);
    }
  } catch (e) {
    // degrees are best-effort; fall back to unsorted
  }

  // attach degrees + sort (known degrees descending, unknown last)
  for (const n of group.neighbors) n.degree = degreeCache.has(n.id) ? degreeCache.get(n.id) : undefined;
  group.neighbors.sort((a, b) => {
    const da = (a.degree == null) ? -1 : a.degree;
    const db = (b.degree == null) ? -1 : b.degree;
    if (db !== da) return db - da;
    return (a.name || a.id).localeCompare(b.name || b.id);
  });
  group._maxDegree = group.neighbors.reduce((m, n) => Math.max(m, n.degree || 0), 0);
  group._degreesLoaded = true;
  group._degreeLoading = false;
  group._note = group.count > DEGREE_CAP
    ? `sorted by degree · sampled first ${DEGREE_CAP.toLocaleString()} of ${group.count.toLocaleString()}`
    : `sorted by degree · ${group.count.toLocaleString()} node${group.count === 1 ? '' : 's'}`;

  // re-render the list from the top in sorted order
  group._rendered = 0;
  if (step.selectedKey === key) {
    updatePanelNeighbors(i);
    if (i === current) drawConnectors(i);
  }
}

// =========================================================================
// Panels — create skeletons, then fill in place
// =========================================================================
function reconcilePanels() {
  // create any missing panels
  for (let i = 0; i < path.length; i++) {
    if (!panelEls[i]) {
      const p = buildPanelSkeleton(i);
      panelEls[i] = p;
      track.appendChild(p);
      updatePanelNode(i);
      updatePanelPreds(i);
      updatePanelNeighbors(i);
    }
  }
  // remove extras
  while (panelEls.length > path.length) {
    const p = panelEls.pop();
    p && p.remove();
  }
}

function buildPanelSkeleton(i) {
  const panel = el('div', 'panel');
  panel.dataset.index = String(i);
  panel.innerHTML =
    `<svg class="connector-layer" data-role="svg"></svg>` +
    `<div class="col col-node"><div class="col-head">Node</div>` +
      `<div class="col-scroll" data-role="node"></div></div>` +
    `<div class="col col-pred"><div class="col-head">Relationships</div>` +
      `<div class="col-scroll" data-role="pred"></div></div>` +
    `<div class="col col-nb"><div class="col-head">Connected nodes</div>` +
      `<div class="nb-toolbar" data-role="nbtool"></div>` +
      `<div class="col-scroll" data-role="nb"></div></div>`;

  // redraw connectors when relationship / neighbour columns scroll
  const predScroll = panel.querySelector('[data-role="pred"]');
  const nbScroll = panel.querySelector('[data-role="nb"]');
  predScroll.addEventListener('scroll', () => { if (i === current) drawConnectors(i); }, { passive: true });
  nbScroll.addEventListener('scroll', () => {
    if (i === current) drawConnectors(i);
    maybeLoadMoreNeighbors(i, nbScroll);
  }, { passive: true });
  return panel;
}

function updatePanelNode(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="node"]');
  const cats = (step.categories || []).map((c) => `<span class="cat-chip">${esc(catLabel(c))}</span>`).join('');
  let meta = '';
  if (step.status === 'loaded') {
    const total = (step.groups || []).reduce((s, g) => s + g.count, 0);
    const rels = (step.groups || []).length;
    meta = `<div class="node-meta"><b>${fmt(rels)}</b> relationship type${rels === 1 ? '' : 's'} · ` +
           `<b>${fmt(total)}</b> connected node slot${total === 1 ? '' : 's'}</div>`;
  } else if (step.status === 'loading') {
    meta = `<div class="node-meta">Loading neighbours…</div>`;
  } else if (step.status === 'error') {
    meta = `<div class="node-meta" style="color:#ffb9a3">Could not load neighbours.</div>`;
  }
  host.innerHTML =
    `<div class="node-card">` +
      `<div class="nc-eyebrow">${i === 0 ? 'Start node' : `Step ${i + 1}`}</div>` +
      `<div class="nc-name">${esc(step.name)}</div>` +
      `<div class="nc-id">${esc(step.curie)}</div>` +
      `<div class="cats">${cats}</div>` +
      meta +
    `</div>`;
}

function updatePanelPreds(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="pred"]');

  if (step.status === 'loading' || step.status === 'idle') {
    host.innerHTML = `<div class="loading-block"><span class="spin"></span> Querying TRAPI for neighbours…</div>`;
    return;
  }
  if (step.status === 'error') {
    host.innerHTML =
      `<div class="error-block">Couldn’t reach the knowledge graph.<br>` +
      `<small style="font-family:var(--font-mono)">${esc(step.error)}</small><br>` +
      `<button class="retry">Try again</button></div>`;
    host.querySelector('.retry').addEventListener('click', () => { step.status = 'idle'; loadStep(i); });
    return;
  }
  if (!step.groups || step.groups.length === 0) {
    if (step.serverErrors && step.serverErrors.length) {
      host.innerHTML =
        `<div class="error-block">The knowledge graph returned no usable response.<br>` +
        `<small style="font-family:var(--font-mono)">${esc(step.serverErrors.join('; '))}</small><br>` +
        `<button class="retry">Try again</button></div>`;
      host.querySelector('.retry').addEventListener('click', () => {
        expandCache.delete(step.curie);
        step.status = 'idle';
        loadStep(i);
      });
      return;
    }
    host.innerHTML = `<div class="empty">No relationships found for this node in the graph.</div>`;
    return;
  }

  const list = el('div', 'pred-list');
  for (const g of step.groups) {
    const b = el('button', `pred-pill dir-${g.direction}${step.selectedKey === g.key ? ' is-selected' : ''}`);
    b.dataset.key = g.key;
    b.dataset.dir = g.direction;
    b.innerHTML =
      `<span class="pp-dir">${arrow(g.direction)}</span>` +
      `<span class="pp-text">` +
        `<span class="pp-name">${esc(predLabel(g.predicate))}</span>` +
        `<span class="pp-full">${g.direction === 'in' ? 'incoming' : 'outgoing'} · ${esc(g.predicate)}</span>` +
      `</span>` +
      `<span class="pp-count">${fmt(g.count)}</span>`;
    b.addEventListener('click', () => selectPredicate(i, g.key));
    list.appendChild(b);
  }
  host.innerHTML = '';
  host.appendChild(list);
}

function renderNeighborStatus(i, group) {
  const panel = panelEls[i]; if (!panel) return;
  const tool = panel.querySelector('[data-role="nbtool"]');
  if (!group) { tool.innerHTML = ''; return; }
  let txt;
  if (group._degreeLoading && !group._degreesLoaded) {
    txt = `<span class="spin"></span>loading node degrees…`;
  } else if (group._degreesLoaded) {
    txt = esc(group._note);
  } else {
    txt = `${fmt(group.count)} node${group.count === 1 ? '' : 's'}`;
  }
  tool.innerHTML = `<span class="nb-status">${txt}</span>`;
}

function updatePanelNeighbors(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="nb"]');
  const tool = panel.querySelector('[data-role="nbtool"]');

  if (!step.selectedKey || !step.groups) {
    tool.innerHTML = '';
    host.innerHTML = `<div class="empty">Pick a relationship on the left to list the nodes connected through it.</div>`;
    return;
  }
  const group = step.groups.find((g) => g.key === step.selectedKey);
  if (!group) { host.innerHTML = ''; return; }

  renderNeighborStatus(i, group);

  if (group.count === 0) {
    host.innerHTML = `<div class="empty">No nodes connected via this relationship.</div>`;
    return;
  }

  host.innerHTML = '';
  const list = el('div', 'nb-list');
  list.dataset.role = 'nblist';
  host.appendChild(list);
  group._rendered = 0;
  appendNeighborChunk(i, list, group);

  const sentinel = el('div', 'sentinel');
  host.appendChild(sentinel);

  if (group.count > DEGREE_CAP && group._degreesLoaded) {
    host.appendChild(el('div', 'muted-hint',
      `Only the first ${DEGREE_CAP.toLocaleString()} of ${group.count.toLocaleString()} nodes are degree-ranked. ` +
      `Narrow the path or pick a more specific relationship to see the rest.`));
  }
}

function appendNeighborChunk(i, list, group) {
  const step = path[i];
  const end = Math.min(group._rendered + CHUNK, group.neighbors.length);
  for (let k = group._rendered; k < end; k++) {
    const n = group.neighbors[k];
    const item = el('button', 'nb-item');
    const cat = (n.categories && n.categories[0]) ? catLabel(n.categories[0]) : '';
    let deg;
    if (n.degree === undefined) {
      deg = `<span class="deg-val unknown">·</span>`;
    } else if (n.degree == null) {
      deg = `<span class="deg-val unknown">n/a</span>`;
    } else {
      const pct = group._maxDegree ? Math.max(4, Math.round((n.degree / group._maxDegree) * 100)) : 0;
      deg = `<span class="deg-val">${fmt(n.degree)}</span>` +
            `<span class="deg-bar"><i style="width:${pct}%"></i></span>`;
    }
    item.innerHTML =
      `<span class="nb-text">` +
        `<span class="nb-name">${esc(n.name || n.id)}</span>` +
        `<span class="nb-id">${esc(n.id)}${cat ? ' · ' + esc(cat) : ''}</span>` +
      `</span>` +
      `<span class="nb-degree" title="node degree">${deg}</span>` +
      `<span class="nb-go">›</span>`;
    item.addEventListener('click', () => stepInto(i, n));
    list.appendChild(item);
  }
  group._rendered = end;
}

function maybeLoadMoreNeighbors(i, nbScroll) {
  const step = path[i];
  if (!step.selectedKey) return;
  const group = step.groups && step.groups.find((g) => g.key === step.selectedKey);
  if (!group || group._rendered >= group.neighbors.length) return;
  if (nbScroll.scrollTop + nbScroll.clientHeight >= nbScroll.scrollHeight - 240) {
    const list = nbScroll.querySelector('[data-role="nblist"]');
    if (list) appendNeighborChunk(i, list, group);
  }
}

// =========================================================================
// Breadcrumb filmstrip + the moving "viewing" frame
// =========================================================================
function renderBreadcrumb() {
  pathEl.innerHTML = '';
  for (let i = 0; i < path.length; i++) {
    const step = path[i];
    const node = el('button', `crumb-node${i === current ? ' is-root' : ''}`);
    node.dataset.nodeIndex = String(i);
    node.innerHTML =
      `<div class="cn-name">${esc(step.name)}</div>` +
      `<div class="cn-id">${esc(step.curie)}</div>`;
    node.addEventListener('click', () => setCurrent(i));
    pathEl.appendChild(node);

    // predicate connector between node i and node i+1
    if (i < path.length - 1) {
      const g = step.groups && step.selectedKey
        ? step.groups.find((x) => x.key === step.selectedKey) : null;
      const dir = g ? g.direction : 'out';
      const label = g ? predLabel(g.predicate) : 'related to';
      const cp = el('div', `crumb-pred dir-${dir}`);
      cp.dataset.predIndex = String(i);
      cp.innerHTML = `<div class="cp-label">${esc(label)}</div><div class="cp-arrow">${arrow(dir)}</div>`;
      pathEl.appendChild(cp);
    }
  }
  positionTripleFrame();
}

function positionTripleFrame() {
  const wrapRect = pathEl.parentElement.getBoundingClientRect();
  const startNode = pathEl.querySelector(`.crumb-node[data-node-index="${current}"]`);
  if (!startNode) { tripleFrame.style.opacity = '0'; return; }

  const endNode = pathEl.querySelector(`.crumb-node[data-node-index="${current + 1}"]`);
  const els = [startNode];
  const predEl = pathEl.querySelector(`.crumb-pred[data-pred-index="${current}"]`);
  if (endNode) { if (predEl) els.push(predEl); els.push(endNode); }

  let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
  for (const e of els) {
    const r = e.getBoundingClientRect();
    left = Math.min(left, r.left); right = Math.max(right, r.right);
    top = Math.min(top, r.top);   bottom = Math.max(bottom, r.bottom);
  }
  const padX = 8, padY = 6;
  tripleFrame.style.opacity = '1';
  tripleFrame.style.left = (left - wrapRect.left - padX) + 'px';
  tripleFrame.style.top = (top - wrapRect.top - padY) + 'px';
  tripleFrame.style.width = (right - left + padX * 2) + 'px';
  tripleFrame.style.height = (bottom - top + padY * 2) + 'px';
}

function scrollPathToCurrent() {
  const startNode = pathEl.querySelector(`.crumb-node[data-node-index="${current}"]`);
  if (!startNode) return;
  const r = startNode.getBoundingClientRect();
  const pr = pathEl.getBoundingClientRect();
  if (r.left < pr.left + 40 || r.right > pr.right - 40) {
    startNode.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
  }
}

// =========================================================================
// Connectors (curved SVG links node→predicate and predicate→neighbour)
// =========================================================================
let connectorRaf = 0;
function drawConnectors(i) {
  if (i !== current) return;
  cancelAnimationFrame(connectorRaf);
  connectorRaf = requestAnimationFrame(() => paintConnectors(i));
}

function paintConnectors(i) {
  const panel = panelEls[i]; if (!panel) return;
  const svg = panel.querySelector('[data-role="svg"]');
  const pr = panel.getBoundingClientRect();
  const W = panel.clientWidth, H = panel.clientHeight;
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  const nodeCard = panel.querySelector('.node-card');
  const predScroll = panel.querySelector('[data-role="pred"]');
  if (!nodeCard) return;

  const nr = nodeCard.getBoundingClientRect();
  const x1 = nr.right - pr.left;
  const y1 = nr.top - pr.top + nr.height / 2;

  // only connect to relationship pills currently visible in their column
  const psRect = predScroll.getBoundingClientRect();
  const pills = panel.querySelectorAll('.pred-pill');
  pills.forEach((pill) => {
    const r = pill.getBoundingClientRect();
    if (r.bottom < psRect.top + 2 || r.top > psRect.bottom - 2) return;
    const x2 = r.left - pr.left;
    const y2 = r.top - pr.top + r.height / 2;
    const selected = pill.classList.contains('is-selected');
    const dir = pill.dataset.dir;
    svg.appendChild(curve(x1, y1, x2, y2, dir, selected ? 0.95 : 0.4, selected ? 2 : 1.25));
  });

  // selected predicate → neighbour list
  const sel = panel.querySelector('.pred-pill.is-selected');
  const nbList = panel.querySelector('[data-role="nblist"]');
  const nbCol = panel.querySelector('.col-nb');
  if (sel && nbCol) {
    const sr = sel.getBoundingClientRect();
    const target = nbList && nbList.firstChild ? nbList.firstChild : nbCol;
    const tr = target.getBoundingClientRect();
    const nbRect = nbCol.getBoundingClientRect();
    const sx = sr.right - pr.left;
    const sy = sr.top - pr.top + sr.height / 2;
    const tx = nbRect.left - pr.left;
    const ty = Math.min(Math.max(tr.top - pr.top + 16, sy - 60), sy + 60);
    svg.appendChild(curve(sx, sy, tx, ty, sel.dataset.dir, 0.85, 2));
  }
}

function curve(x1, y1, x2, y2, dir, opacity, width) {
  const dx = Math.max(28, (x2 - x1) * 0.55);
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`);
  p.setAttribute('fill', 'none');
  const color = dir === 'in' ? 'var(--in)' : 'var(--out)';
  p.setAttribute('stroke', color);
  p.setAttribute('stroke-width', String(width));
  p.setAttribute('stroke-opacity', String(opacity));
  p.setAttribute('stroke-linecap', 'round');
  return p;
}

// =========================================================================
// Navigation
// =========================================================================
function setCurrent(i) {
  current = Math.max(0, Math.min(path.length - 1, i));
  track.style.transform = `translateX(${-current * 100}%)`;
  panelEls.forEach((p, idx) => p && p.classList.toggle('is-current', idx === current));
  renderBreadcrumb();
  scrollPathToCurrent();
  updateControls();
  // ensure current step's data is present
  if (path[current] && path[current].status === 'idle') loadStep(current);
  drawConnectors(current);
}

function updateControls() {
  prevBtn.disabled = current <= 0;
  nextBtn.disabled = current >= path.length - 1;
  const step = path[current];
  const where = step ? `${esc(step.name)}` : '';
  stepReadout.innerHTML = `triple <b>${current + 1}</b> of <b>${path.length}</b> · viewing <b>${where}</b>`;
}

// =========================================================================
// Wire-up
// =========================================================================
function init() {
  buildExamples();
  wireStart();

  prevBtn.addEventListener('click', () => setCurrent(current - 1));
  nextBtn.addEventListener('click', () => setCurrent(current + 1));

  document.addEventListener('keydown', (e) => {
    if (!overlay.hidden) return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { setCurrent(current - 1); }
    else if (e.key === 'ArrowRight') { setCurrent(current + 1); }
  });

  let resizeRaf = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => { positionTripleFrame(); drawConnectors(current); });
  });
  pathEl.addEventListener('scroll', () => positionTripleFrame(), { passive: true });

  $('#curieInput').focus();
}

document.addEventListener('DOMContentLoaded', init);
