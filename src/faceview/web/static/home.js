// Home page: hero with a live face that idles and speaks the characters' scripted catchphrases, counts from the
// Python package, and cards to the pages.
import { route, api, app, list, esc } from './core.js';
import { FaceStage, faceData, runStages, reducedMotion } from './face_stage.js';

const DOORS = [
  ['#/studio', 'Expression studio', 'Twelve FACS presets and every Action Unit on a slider, with idle blinks, breathing and saccades.', 'a'],
  ['#/lipsync', 'Lip-sync', 'Type a line: text to ARPAbet phonemes to timed visemes, blended into mouth shapes as the face speaks.', 'b'],
  ['#/visemes', 'Visemes', 'The 15 mouth shapes, the phonemes behind each, their AU targets and tongue poses.', 'c'],
  ['#/duet', 'Two-bot test mode', 'The desktop app\'s canned test mode: two faces trading its scripted seed lines.', 'a'],
  ['#/chat', 'Chat with Claude', 'In the local app, with your own API key: a character replies and the face speaks it.', 'b'],
  ['#/characters', 'Characters', 'The seven character sheets: backstory, Big Five traits, catchphrases, voice.', 'c'],
  ['#/verify', 'Python vs JavaScript', 'The browser port checked against the Python pipeline and renderer, live.', 'a'],
  ['#/gallery', 'Desktop renders', 'Screens and avatars from the desktop app, including the 3D heads not ported here.', 'b'],
];

route('home', async () => {
  const [info, D, lines] = await Promise.all([api('/info'), faceData(), api('/demo_lines')]);
  app().innerHTML = `
  <section class="hero home-hero">
    <div>
      <p class="kicker">Face · Voice · Chat</p>
      <h1 class="glow">faceView</h1>
      <p class="lead">${esc(info.summary)}</p>
      <p>This site runs the avatar's face in your browser: the FACS expression model, the text-to-viseme lip-sync and
      the stylised renderer, ported from the Python app and checked against it. Voice, webcam and live LLM chat need
      the desktop or local web app.</p>
      <a class="btn" href="#/lipsync">Make it talk</a><a class="btn alt" href="#/studio">Expression studio</a>
    </div>
    <div>
      <canvas id="hero" class="face hero-face"></canvas>
      <p class="caption" id="cap" aria-live="polite"></p>
    </div>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.action_units}</b><span>FACS Action Units</span></div>
    <div class="tile"><b>${info.expressions}</b><span>expression presets</span></div>
    <div class="tile"><b>${info.visemes}</b><span>visemes</span></div>
    <div class="tile"><b>${info.phonemes}</b><span>ARPAbet phonemes mapped</span></div>
    <div class="tile"><b>${info.personas}</b><span>appearance presets (${info.stylised_personas} drawn here)</span></div>
    <div class="tile"><b>${info.characters}</b><span>character sheets</span></div>
  </div>
  <div class="grid wide">${list(DOORS, ([href, t, d, c]) => `<a class="card door ${c}" href="${href}"><b>${t}</b><span>${d}</span></a>`)}</div>`;

  const stage = new FaceStage(document.getElementById('hero'), D, { emotion: 'happy', persona: 'claude', seed: 11,
    label: 'Playful Claude, the stylised avatar, idling and speaking scripted lines' });
  const cap = document.getElementById('cap');
  const pool = lines.catchphrases.filter(c => D.byName[c.persona]?.render_mode === 'stylised');
  let i = 0;
  const speak = () => {
    const c = pool[i++ % pool.length];
    stage.setPersona(c.persona);
    stage.say(c.line);
    cap.innerHTML = `<span class="label-demo">scripted</span>${esc(c.name)}: "${esc(c.line)}"`;
  };
  const run = runStages([stage]);
  if (reducedMotion()) {
    cap.innerHTML = '<span class="mute">Reduced motion is on: the face is still. Open Lip-sync to make it talk.</span>';
    run.poke();
    return;
  }
  setTimeout(speak, 1200);
  const timer = setInterval(() => { if (!document.getElementById('hero')) return clearInterval(timer); if (!stage.speaking()) speak(); }, 6500);
});
