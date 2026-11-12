// Characters: the character sheets in assets/config/characters.json (name, backstory, Big Five traits, topics,
// catchphrases, goals, Kokoro voice). Faces with the stylised renderer get a live preview that says a catchphrase.
import { route, app, api, esc, list } from './core.js';
import { FaceStage, faceData, runStages } from './face_stage.js';

const TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

route('characters', async () => {
  const [chars, D] = await Promise.all([api('/characters'), faceData()]);
  app().innerHTML = `
  <p class="kicker">Library</p><h1 class="glow">Characters</h1>
  <p class="lead">Each persona can carry a character sheet. The desktop app turns it into the system prompt
  (<code>Character.narrate_identity</code>) for whichever engine is running, keeps a separate memory file per persona
  and speaks with the character's Kokoro voice.</p>
  <div class="grid wide">${list(chars, c => `
    <div class="card">
      ${c.render_mode === 'stylised' ? `<canvas class="face" data-p="${esc(c.persona)}" width="320" height="240"></canvas>
        <button class="alt small" data-say="${esc(c.persona)}">Say a catchphrase <span class="label-demo">scripted</span></button>`
        : `<p class="small mute">Drawn by the desktop app's <span class="mono">${esc(c.render_mode)}</span> renderer
           (<a href="#/gallery">see renders</a>).</p>`}
      <b>${esc(c.name)}</b><small>${esc(c.occupation)}${c.age ? ' · ' + esc(c.age) : ''} · persona <span class="mono">${esc(c.persona)}</span> · voice <span class="mono">${esc(c.voice)}</span></small>
      <span>${esc(c.backstory)}</span>
      <h3>Big Five</h3>
      <div class="traits">${list(TRAITS, t => `<span>${t}</span><span class="bar"><i style="width:${(c.traits[t] ?? 0) * 100}%"></i></span><span class="mono">${(c.traits[t] ?? 0).toFixed(2)}</span>`)}</div>
      ${c.topics.length ? `<h3>Topics</h3><p class="small">${esc(c.topics.join(', '))}</p>` : ''}
      ${c.catchphrases.length ? `<h3>Catchphrases</h3>${list(c.catchphrases, l => `<p class="small">"${esc(l)}"</p>`)}` : ''}
      ${c.goals.length ? `<h3>Goals</h3>${list(c.goals, g => `<p class="small">${esc(g)}</p>`)}` : ''}
    </div>`)}</div>`;
  const stages = [...app().querySelectorAll('canvas[data-p]')].map((cv, i) =>
    new FaceStage(cv, D, { emotion: 'neutral', persona: cv.dataset.p, seed: 40 + i, width: 320, height: 240,
      label: `Face of persona ${cv.dataset.p}` }));
  runStages(stages);
  const turn = {};
  app().addEventListener('click', e => {
    const b = e.target.closest('[data-say]'); if (!b) return;
    const c = chars.find(x => x.persona === b.dataset.say);
    const st = stages.find(s => s.personaName === c.persona);
    const k = turn[c.persona] = ((turn[c.persona] ?? -1) + 1) % Math.max(1, c.catchphrases.length);
    if (st && c.catchphrases.length) st.say(c.catchphrases[k]);
  });
});
