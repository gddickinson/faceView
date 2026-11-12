// Visemes: the 15 mouth shapes of vision/visemes.py, each drawn by the canvas renderer from its AU targets on a
// neutral face, with the phonemes that map to it, its AU values and the tongue pose the 3D heads use.
import { route, app, api, esc, list } from './core.js';
import { applyPersona, faceState, faceStateToParams, setAU } from './face_engine.js';
import { drawToCanvas } from './face_render.js';
import { faceData } from './face_stage.js';

const AU_ORDER = ['AU25', 'AU26', 'AU22', 'AU20', 'AU12'];
const AU_NAME = { AU25: 'lips part', AU26: 'jaw drop', AU22: 'funneler', AU20: 'stretch', AU12: 'corner pull' };

route('visemes', async () => {
  const [V, D] = await Promise.all([api('/visemes'), faceData()]);
  app().innerHTML = `
  <p class="kicker">Face</p><h1 class="glow">Visemes</h1>
  <p class="lead">A viseme is what a speech sound looks like from outside: P, B and M all close the lips, so they share
  one. faceView's 15-class alphabet (from the FaceForge animation system) sets five mouth Action Units per viseme
  and leaves the rest of the face alone, so a smiling speaker keeps smiling between syllables.</p>
  <p class="small mute">Each tile is the browser renderer drawing that viseme's AU targets on the neutral face (default
  persona), cropped to the lower face. Slots are ${Math.round(V.phoneme_duration * 1000)} ms per phoneme and ${Math.round(V.word_gap * 1000)} ms
  between words at speed 1.</p>
  <div class="vgrid">${list(V.visemes, v => `
    <div class="card">
      <canvas width="320" height="240" data-v="${esc(v.name)}" role="img" aria-label="Viseme ${esc(v.name)}: ${esc(v.info)}"></canvas>
      <b>${esc(v.name)}</b><small>${esc(v.info)}</small>
      <span class="mono small">${list(AU_ORDER.filter(a => v.au[a]), a => `${a} ${v.au[a].toFixed(2)}`, ' · ') || 'all mouth AUs 0'}</span>
    </div>`)}</div>
  <h2>Phonemes to visemes</h2>
  <div class="scroll"><table>
    <thead><tr><th>Viseme</th><th>ARPAbet phonemes</th>${list(AU_ORDER, a => `<th>${a}<br><span class="small">${AU_NAME[a]}</span></th>`)}<th>Tongue (extend, vertical, lateral, taper)</th></tr></thead>
    <tbody>${list(V.visemes, v => `<tr><td><b>${esc(v.name)}</b></td><td class="mono">${esc(v.phonemes.join(' ')) || '(silence)'}</td>
      ${list(AU_ORDER, a => `<td class="mono">${(v.au[a] || 0).toFixed(2)}</td>`)}<td class="mono">${v.tongue.map(x => x.toFixed(2)).join(', ')}</td></tr>`)}</tbody>
  </table></div>
  <p class="small mute">The tongue poses feed the desktop app's optional talking-tongue effect; the stylised face drawn
  here has no tongue.</p>`;
  const full = document.createElement('canvas');
  for (const cv of app().querySelectorAll('canvas[data-v]')) {
    const v = V.visemes.find(x => x.name === cv.dataset.v);
    const s = faceState();
    for (const [au, val] of Object.entries(v.au)) setAU(s, au, val);
    drawToCanvas(full, applyPersona(faceStateToParams(s), D.byName.default), 640, 480);
    cv.getContext('2d').drawImage(full, 200, 215, 240, 180, 0, 0, 320, 240);  // lower face, 1.33x
  }
});
