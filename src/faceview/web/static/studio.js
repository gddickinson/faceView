// Expression studio: the 12 FACS expression presets, every Action Unit on a slider (added on top of the preset,
// as TalkingAvatar's target), head pose and gaze, the stylised personas, and the idle systems switched on or off.
import { route, app, esc, list, slider } from './core.js';
import { AU_IDS, faceState, applyExpression } from './face_engine.js';
import { FaceStage, faceData, personaDot, runStages } from './face_stage.js';

const POSE = [['head_yaw', 'Head yaw'], ['head_pitch', 'Head pitch'], ['eye_look_x', 'Gaze x'], ['eye_look_y', 'Gaze y']];
const SHOWN = ['smile', 'jaw_open', 'brow_raise', 'eye_open', 'mouth_pucker', 'mouth_stretch', 'cheek_raise', 'lip_corner_drop'];

route('studio', async (args, query) => {
  const D = await faceData();
  const names = Object.keys(D.expr);
  const start = names.includes(query.e) ? query.e : 'happy';
  app().innerHTML = `
  <p class="kicker">Face</p><h1 class="glow">Expression studio</h1>
  <p class="lead">An expression is a set of FACS Action Unit activations (assets/config/expressions.json). The avatar
  converts them to renderer parameters exactly as <code>face_state_to_params</code> does; sliders below add to the
  preset, and blinks, breathing and saccades run on top.</p>
  <div class="studio">
    <div class="sticky">
      <canvas id="face" class="face"></canvas>
      <div class="readout" id="ro" aria-live="off"></div>
    </div>
    <div>
      <div class="panel">
        <h3>Expression preset</h3>
        <div class="chips" id="ex">${list(names, n => `<button class="chip" data-e="${esc(n)}">${esc(n)}</button>`)}</div>
        <h3>Persona</h3>
        <div class="chips" id="pe">${list(D.stylised, n => `<button class="chip" data-p="${esc(n)}">${personaDot(D.byName[n])}${esc(n)}</button>`)}</div>
        <p class="small mute">The ${D.stylised.length} presets drawn by the stylised renderer; the other
        ${D.personas.length - D.stylised.length} use the desktop app's 3D, anatomical or photo-warp renderers
        (see <a href="#/gallery">Desktop renders</a>).</p>
        <h3>Idle behaviour</h3>
        <div class="toggles">
          <label><input type="checkbox" id="i-blink" checked> blinks</label>
          <label><input type="checkbox" id="i-breath" checked> breathing</label>
          <label><input type="checkbox" id="i-saccade" checked> saccades</label>
        </div>
        <button class="alt" id="reset">Reset sliders</button>
      </div>
      <div class="panel"><h3>Action Units (added to the preset)</h3><div class="au-grid" id="aus"></div></div>
      <div class="panel"><h3>Head and gaze</h3><div class="au-grid" id="pose"></div></div>
    </div>
  </div>`;

  const stage = new FaceStage(document.getElementById('face'), D, { emotion: start, persona: 'claude', seed: 3,
    label: 'Stylised face showing the selected expression' });
  const extra = {};
  const ro = document.getElementById('ro');
  const run = runStages([stage]);

  // Effective target = preset (+ head pose) + slider offsets, clamped like FaceState.set.
  function apply() {
    const base = applyExpression(faceState(), D.expr[stage.emotion] || {});
    const ov = {};
    for (const au of AU_IDS) ov[au] = Math.max(0, Math.min(1, base[au] + (extra[au] || 0)));
    for (const [k] of POSE) ov[k] = base[k] + (extra[k] || 0);
    stage.avatar.overrides = ov;
    run.poke();
  }
  const auName = Object.fromEntries(D.aus.map(a => [a.id, a.name]));
  const sliders = [];
  for (const au of AU_IDS) {
    sliders.push(slider(document.getElementById('aus'), { id: au, label: `${au} ${auName[au] || ''}`, min: -1, max: 1,
      step: 0.05, value: 0, fmt: v => (v > 0 ? '+' : '') + v.toFixed(2) }, v => { extra[au] = v; apply(); }));
  }
  for (const [k, label] of POSE) {
    sliders.push(slider(document.getElementById('pose'), { id: k, label, min: -1, max: 1, step: 0.05, value: 0,
      fmt: v => v.toFixed(2) }, v => { extra[k] = v; apply(); }));
  }
  const mark = () => {
    document.querySelectorAll('#ex .chip').forEach(b => b.classList.toggle('on', b.dataset.e === stage.emotion));
    document.querySelectorAll('#pe .chip').forEach(b => b.classList.toggle('on', b.dataset.p === stage.personaName));
  };
  document.getElementById('ex').addEventListener('click', e => {
    const b = e.target.closest('[data-e]'); if (!b) return;
    stage.setEmotion(b.dataset.e); apply(); mark();
    history.replaceState(null, '', `#/studio?e=${encodeURIComponent(b.dataset.e)}`);
  });
  document.getElementById('pe').addEventListener('click', e => {
    const b = e.target.closest('[data-p]'); if (!b) return;
    stage.setPersona(b.dataset.p); run.poke(); mark();
  });
  for (const k of ['blink', 'breath', 'saccade']) {
    document.getElementById('i-' + k).addEventListener('change', e => { stage.avatar.idle[k] = e.target.checked; run.poke(); });
  }
  document.getElementById('reset').addEventListener('click', () => {
    for (const s of sliders) { s.value = 0; s.dispatchEvent(new Event('input')); }
  });
  stage.onFrame = p => {
    ro.innerHTML = list(SHOWN, k => `${k} <b>${p[k].toFixed(2)}</b>`, ' · ');
  };
  apply(); mark();
});
