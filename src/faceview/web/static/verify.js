// Verify: runs the Python-vs-JS comparisons (verify_core.js) live in the browser against values the Python code
// produced (scripts/make_face_fixture.py), and puts the desktop renderer's PNGs next to the canvas port with a
// per-pixel difference.
import { route, app, api, esc, list } from './core.js';
import { applyExpression, applyPersona, faceState, faceStateToParams, setAU, VISEMES } from './face_engine.js';
import { drawToCanvas } from './face_render.js';
import { faceData } from './face_stage.js';
import { colors, compareReference, TOL } from './verify_core.js';

function refParams(stem, D) {
  const s = faceState();
  if (stem.startsWith('speaking_')) for (const [au, v] of Object.entries(VISEMES[stem.split('_')[1]])) setAU(s, au, v);
  else applyExpression(s, D.expr[stem]);
  return applyPersona(faceStateToParams(s), D.byName.default);
}

function loadImage(src) {
  return new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error('no ' + src)); im.src = src; });
}

// Mean |difference| per channel (0-255) and the share of pixels whose largest channel difference is <= 16.
function diff(a, b) {
  let sum = 0, close = 0;
  const n = a.length / 4;
  for (let i = 0; i < a.length; i += 4) {
    const d = [0, 1, 2].map(k => Math.abs(a[i + k] - b[i + k]));
    sum += d[0] + d[1] + d[2];
    if (Math.max(...d) <= 16) close++;
  }
  return { mad: sum / (3 * n), close: close / n };
}

route('verify', async () => {
  const [REF, D] = await Promise.all([api('/reference'), faceData()]);
  app().innerHTML = `
  <p class="kicker">Verify</p><h1 class="glow">Python vs JavaScript</h1>
  <p class="lead">The browser runs ports of faceView's Python modules. These checks compare them, in your browser,
  with numbers the Python code wrote to <span class="mono">tests/fixtures/</span> (the same checks run in the node test
  and in CI).</p>
  <div class="scroll"><table id="checks"><thead><tr><th>Ported module</th><th>Checks</th><th>Max |error|</th><th>Result</th></tr></thead><tbody></tbody></table></div>
  <h2>Renderer: QPainter vs canvas</h2>
  <p>Left: the desktop app's <span class="mono">render_face</span> (PySide6 QPainter, 320x240, committed PNG). Right:
  the canvas port drawing the same parameters now. Geometry and colours follow the Python code (colours match
  QColor exactly, see the table); antialiasing and gradient interpolation differ slightly between the two
  rasterisers, so the pixel difference is small but not zero.</p>
  <div class="cmp" id="cmp"></div>`;

  const tbody = app().querySelector('#checks tbody');
  const sections = compareReference(REF, { expr: D.expr, dict: D.dict,
    personas: Object.fromEntries(D.personas.map(p => [p.name, p])) });
  if (REF.colors) sections.push(colors(REF.colors));
  tbody.innerHTML = list(sections, s => `<tr><td>${esc(s.name)}</td><td class="mono">${s.checks}</td>
    <td class="mono">${s.maxErr.toExponential(1)}</td>
    <td>${s.fails ? `<span class="fail">${s.fails} differ</span><br><span class="small">${esc(s.firstFail)}</span>` : `<span class="pass">all within ${TOL}</span>`}</td></tr>`);

  const cmp = document.getElementById('cmp');
  for (const stem of REF.renders) {
    const box = document.createElement('div');
    box.className = 'card';
    box.innerHTML = `<div class="pair"><img src="static/ref/${esc(stem)}.png" alt="Python render: ${esc(stem)}" width="320" height="240">
      <canvas width="320" height="240" role="img" aria-label="Canvas render: ${esc(stem)}"></canvas></div>
      <b>${esc(stem.replace('_', ' '))}</b><small class="mono">...</small>`;
    cmp.appendChild(box);
    const cv = box.querySelector('canvas');
    drawToCanvas(cv, refParams(stem, D), 320, 240);
    try {
      const im = await loadImage(`static/ref/${stem}.png`);
      const off = document.createElement('canvas'); off.width = 320; off.height = 240;
      const oc = off.getContext('2d'); oc.drawImage(im, 0, 0);
      const d = diff(oc.getImageData(0, 0, 320, 240).data, cv.getContext('2d').getImageData(0, 0, 320, 240).data);
      box.querySelector('small').textContent = `mean |diff| ${d.mad.toFixed(2)} / 255 · ${(d.close * 100).toFixed(1)}% of pixels within 16 levels`;
    } catch (e) { box.querySelector('small').textContent = e.message; }
  }
});
