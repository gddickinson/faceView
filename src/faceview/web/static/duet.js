// Two-bot test mode, canned: the desktop app's TestConversation._canned_loop replayed in the browser. Two faces
// alternate through SEED_PROMPTS (llm/canned_lines.py); each line is lip-synced on its speaker, the mood is guessed
// by the same keyword rule, and the wait between turns uses the same pacing formula. No model is involved.
import { route, app, api, esc, list } from './core.js';
import { FaceStage, faceData, runStages, reducedMotion } from './face_stage.js';

const PERIOD_S = 6.0;  // TestConversation.period_s default
// The keyword mood labels follow DeepFace's naming; the expression presets call surprise "surprised".
const MOOD_TO_PRESET = { happy: 'happy', sad: 'sad', surprise: 'surprised', neutral: 'neutral' };

export function guessEmotion(text, keywords) {
  const t = text.toLowerCase();
  for (const [label, words] of Object.entries(keywords)) for (const w of words) if (t.includes(w)) return label;
  return 'neutral';
}
// TestConversation._wait_paced: roughly proportional to talking time plus an idle gap.
export const pacedWait = text => Math.max(2.5, Math.min(PERIOD_S + 4.0, 0.05 * text.length + 2.5));

route('duet', async () => {
  const [D, lines] = await Promise.all([faceData(), api('/demo_lines')]);
  const opts = list(D.stylised, n => `<option>${esc(n)}</option>`);
  app().innerHTML = `
  <p class="kicker">Talk</p><h1 class="glow">Two-bot test mode <span class="label-demo">scripted</span></h1>
  <p class="lead">In the desktop app, test mode puts a second avatar in the camera window and lets two bots talk.
  With the <b>canned</b> engine they cycle through a fixed list of ${lines.seed_prompts.length} seed lines; that is
  what plays here. With Anthropic or Ollama selected (desktop app only) the lines come from real models instead.</p>
  <div class="row">
    <div><label for="pa">Camera-window persona</label><select id="pa">${opts}</select></div>
    <div><label for="pb">Avatar-window persona</label><select id="pb">${opts}</select></div>
  </div>
  <button id="toggle">Start</button><button class="alt" id="reset">Restart from line 1</button>
  <div class="duet" style="margin-top:16px">
    <div><canvas id="fa" class="face"></canvas><p class="caption">Camera bot</p></div>
    <div><canvas id="fb" class="face"></canvas><p class="caption">Avatar bot</p></div>
  </div>
  <div class="panel log" id="log" aria-live="polite"><p class="mute">Press Start: the camera bot speaks first.</p></div>
  <p class="small mute">Moods come from the same keyword rule the desktop app uses (it shows them in the status
  pills); here the speaker's expression is also set from the mood so you can see it.</p>`;

  const pa = document.getElementById('pa'), pb = document.getElementById('pb');
  pa.value = 'warm_dark'; pb.value = 'claude';
  const A = new FaceStage(document.getElementById('fa'), D, { emotion: 'neutral', persona: pa.value, seed: 21, label: 'Camera-window bot face' });
  const B = new FaceStage(document.getElementById('fb'), D, { emotion: 'neutral', persona: pb.value, seed: 22, label: 'Avatar-window bot face' });
  const run = runStages([A, B]);
  const log = document.getElementById('log'), btn = document.getElementById('toggle');
  let i = 0, playing = false, timer = null;

  function turn() {
    if (!document.getElementById('log')) return;
    const side = i % 2 === 0 ? A : B;
    const text = lines.seed_prompts[i % lines.seed_prompts.length];
    const mood = guessEmotion(text, lines.emotion_keywords);
    side.setEmotion(MOOD_TO_PRESET[mood] || 'neutral');
    side.say(text);
    const who = side === A ? 'Camera bot' : 'Avatar bot';
    if (i === 0) log.innerHTML = '';
    log.insertAdjacentHTML('beforeend', `<p><span class="who ${side === B ? 'b' : ''}">${who}</span>${esc(text)}<span class="mood">mood: ${esc(mood)}</span></p>`);
    log.scrollTop = log.scrollHeight;
    i++;
    timer = setTimeout(() => { if (playing) turn(); }, pacedWait(text) * 1000);
  }
  function setPlaying(on) {
    playing = on; btn.textContent = on ? 'Pause' : 'Start';
    clearTimeout(timer);
    if (on) turn();
  }
  btn.addEventListener('click', () => setPlaying(!playing));
  document.getElementById('reset').addEventListener('click', () => { i = 0; setPlaying(true); });
  pa.addEventListener('change', () => { A.setPersona(pa.value); run.poke(); });
  pb.addEventListener('change', () => { B.setPersona(pb.value); run.poke(); });
  if (!reducedMotion()) setPlaying(true);
});
