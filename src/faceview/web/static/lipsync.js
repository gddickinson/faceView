// Lip-sync: text -> ARPAbet phonemes (bundled CMU dictionary, letter rules as fallback) -> timed visemes ->
// coarticulated mouth-AU blend, played on the face. Same pipeline as vision/speech.py + avatar.py. On the local
// app the Python SpeechEngine's timeline (POST /api/speech) is compared with the browser's.
import { route, app, api, esc, list, slider, isStatic } from './core.js';
import { SpeechEngine, visemeBlendAt } from './face_engine.js';
import { FaceStage, faceData, reducedMotion, runStages } from './face_stage.js';

const CURVES = [['AU25', 'lips part', '#3fe0ff'], ['AU26', 'jaw drop', '#ff5fd2'], ['AU22', 'funneler', '#ffb35c'],
  ['AU20', 'stretch', '#3dff8f'], ['AU12', 'corner pull', '#b18cff']];

function drawChart(cv, tl, dur, playhead) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  if (!tl.length) return;
  const T = dur + 0.15, X = t => 8 + (w - 16) * t / T, Y = v => h - 14 - (h - 26) * v / 0.7;
  c.strokeStyle = 'rgba(148,163,198,.25)'; c.lineWidth = 1;
  for (const v of [0, 0.2, 0.4, 0.6]) { c.beginPath(); c.moveTo(8, Y(v)); c.lineTo(w - 8, Y(v)); c.stroke(); }
  c.fillStyle = 'rgba(148,163,198,.8)'; c.font = '10px ui-monospace, monospace';
  for (const v of [0.2, 0.4, 0.6]) c.fillText(v.toFixed(1), 10, Y(v) - 2);
  const n = Math.max(60, Math.round(w));
  for (const [au, , col] of CURVES) {
    c.strokeStyle = col; c.lineWidth = 1.6; c.beginPath();
    for (let i = 0; i <= n; i++) { const t = T * i / n; const v = visemeBlendAt(tl, t)[au] || 0; i ? c.lineTo(X(t), Y(v)) : c.moveTo(X(t), Y(v)); }
    c.stroke();
  }
  if (playhead !== null) { c.strokeStyle = '#fff'; c.beginPath(); c.moveTo(X(playhead), 4); c.lineTo(X(playhead), h - 4); c.stroke(); }
}

route('lipsync', async () => {
  const [D, lines] = await Promise.all([faceData(), api('/demo_lines')]);
  const eng = new SpeechEngine(D.dict);
  const scripted = [...lines.seed_prompts.slice(0, 6), ...lines.catchphrases.map(c => c.line)];
  app().innerHTML = `
  <p class="kicker">Face</p><h1 class="glow">Lip-sync</h1>
  <p class="lead">The avatar's mouth is driven from text, not audio: each word becomes ARPAbet phonemes, each phoneme
  one of 15 visemes on an 85 ms slot (40 ms pauses between words), and overlapping viseme envelopes are blended into
  the mouth Action Units while the rest of the face keeps its expression.</p>
  <div class="studio">
    <div class="sticky">
      <canvas id="face" class="face"></canvas>
      <p class="caption" id="cap" aria-live="polite"></p>
    </div>
    <div class="panel">
      <label for="txt">Line to speak</label>
      <textarea id="txt" maxlength="600">Hello world. What does hope look like for you?</textarea>
      <div id="spd"></div>
      <label for="emo">Expression while speaking</label>
      <select id="emo">${list(Object.keys(D.expr), n => `<option${n === 'happy' ? ' selected' : ''}>${esc(n)}</option>`)}</select>
      <label for="per">Persona</label>
      <select id="per">${list(D.stylised, n => `<option${n === 'claude' ? ' selected' : ''}>${esc(n)}</option>`)}</select>
      <button id="go">Speak</button>
      <h3>Scripted lines <span class="label-demo">demo</span></h3>
      <p class="small mute">From the desktop app's canned test mode and the character sheets; no model writes them.</p>
      <div class="chips" id="lines">${list(scripted, (l, i) => `<button class="chip" data-i="${i}">${esc(l.length > 46 ? l.slice(0, 44) + '…' : l)}</button>`)}</div>
    </div>
  </div>
  <h2>Timeline</h2>
  <div class="strip" id="strip" aria-label="Viseme timeline"></div>
  <canvas class="chart" id="chart" aria-label="Mouth Action Unit targets over time"></canvas>
  <div class="legend">${list(CURVES, ([au, n, col]) => `<span><i style="background:${col}"></i>${au} ${n}</span>`)}</div>
  <h2>Words and phonemes</h2>
  <p class="small mute">Bordered amber: not in the bundled ${Object.keys(D.dict).length}-word dictionary, so spelled
  out by letter rules (the desktop app does the same).</p>
  <div class="words" id="words"></div>
  <p class="small" id="pycheck"></p>`;

  const stage = new FaceStage(document.getElementById('face'), D, { emotion: 'happy', persona: 'claude', seed: 5,
    label: 'Stylised face lip-syncing the line' });
  const txt = document.getElementById('txt'), cap = document.getElementById('cap');
  let speed = 1, tl = [], dur = 0;
  slider(document.getElementById('spd'), { id: 'speed', label: 'Speed', min: 0.5, max: 2, step: 0.05, value: 1,
    fmt: v => v.toFixed(2), unit: 'x' }, v => { speed = v; });
  const chart = document.getElementById('chart'), strip = document.getElementById('strip');
  let cells = [];

  async function speak(play = true) {
    const text = txt.value.trim();
    if (!text) return;
    stage.setEmotion(document.getElementById('emo').value);
    stage.setPersona(document.getElementById('per').value);
    const u = play ? stage.say(text, speed) : stage.avatar.engine.generate(text, speed);
    tl = play ? u.timeline : u; dur = tl.length ? tl[tl.length - 1].end : 0;
    cap.textContent = `${tl.length} visemes · ${dur.toFixed(2)} s`;
    strip.innerHTML = list(tl, v => `<span class="v ${v.viseme === 'REST' ? 'rest' : ''}" title="${esc(v.phoneme)} ${v.start.toFixed(3)}-${v.end.toFixed(3)} s"><b>${esc(v.viseme)}</b>${esc(v.phoneme)}</span>`);
    cells = [...strip.children];
    document.getElementById('words').innerHTML = list(eng.words(text), w =>
      `<span class="w ${w.source}">${esc(w.word)}<small>${esc(w.phonemes.join(' '))}</small></span>`);
    drawChart(chart, tl, dur, null);
    const check = document.getElementById('pycheck');
    if (isStatic()) { check.innerHTML = 'Computed in your browser (<a href="#/verify">checked against Python</a>).'; return; }
    try {
      const py = await api('/speech', { body: { text, speed } });
      const same = py.timeline.length === tl.length && py.timeline.every((p, i) => p.viseme === tl[i].viseme
        && Math.abs(p.start - tl[i].start) < 1e-9 && Math.abs(p.end - tl[i].end) < 1e-9);
      check.innerHTML = same ? '<span class="pass">Python SpeechEngine timeline: identical.</span>'
        : '<span class="fail">Python SpeechEngine timeline differs from the browser\'s.</span>';
    } catch (e) { check.textContent = 'Python check failed: ' + e.message; }
  }
  stage.onFrame = (p, t) => {
    const u = stage.avatar.utterance;
    if (!u) return;
    const rel = t - u.start;
    cells.forEach((c, i) => c.classList.toggle('on', tl[i] && rel >= tl[i].start && rel < tl[i].end));
    drawChart(chart, tl, dur, Math.min(rel, dur));
  };
  document.getElementById('go').addEventListener('click', () => speak());
  document.getElementById('lines').addEventListener('click', e => {
    const b = e.target.closest('[data-i]'); if (!b) return;
    txt.value = scripted[+b.dataset.i]; speak();
  });
  const run = runStages([stage]);
  if (reducedMotion()) { speak(false); run.poke(); } else speak();
});
