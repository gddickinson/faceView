// Chat with Claude (local app only): pick a character, type, and the reply from POST /api/chat is lip-synced on the
// face. The API key stays in the server's environment. On the static site, or without a key, the page says so and
// still lets you make the face say your own text.
import { route, app, api, esc, list, isStatic, needsServer } from './core.js';
import { FaceStage, faceData, runStages } from './face_stage.js';

route('chat', async () => {
  const [D, chars, status] = await Promise.all([faceData(), api('/characters'), api('/chat/status', { fresh: true })]);
  const drawable = chars.filter(c => c.render_mode === 'stylised');
  const live = !!status.available;
  app().innerHTML = `
  <p class="kicker">Talk</p><h1 class="glow">Chat with Claude</h1>
  <p class="lead">Pick a character; its sheet (backstory, catchphrases, goals) becomes the system prompt, as in the
  desktop app, and the face speaks each reply.</p>
  ${live ? `<p class="callout a">Live: replies come from <span class="mono">${esc(status.model)}</span> through the
    local server. Your API key stays on the server.</p>`
    : isStatic() ? `<div class="callout">${needsServer('Live chat with Claude')}<p class="small">It runs in the local
      web app (<span class="mono">python run_web.py</span>) or the desktop app, with <span class="mono">ANTHROPIC_API_KEY</span>
      set in that process's environment. The key is never sent to the browser. Below, the face can still say your
      own words.</p></div>`
    : `<p class="callout">No <span class="mono">ANTHROPIC_API_KEY</span> in the web server's environment, so live chat is
      off. Restart <span class="mono">run_web.py</span> with the key set. Below, the face can still say your own words.</p>`}
  <div class="studio">
    <div class="sticky"><canvas id="face" class="face"></canvas><p class="caption" id="cap"></p></div>
    <div>
      <label for="who">Character</label>
      <select id="who">${list(drawable, c => `<option value="${esc(c.persona)}">${esc(c.name)}: ${esc(c.occupation)}</option>`)}</select>
      <div class="panel log" id="chatlog" aria-live="polite"><p class="mute">${live ? 'Say hello.' : 'Type something for the face to say.'}</p></div>
      <div class="chatbox">
        <div style="flex:1"><label for="msg">${live ? 'Message' : 'Text to say'}</label><textarea id="msg" maxlength="2000"></textarea></div>
        <button id="send">${live ? 'Send' : 'Say it'}</button>
      </div>
    </div>
  </div>`;

  const sel = document.getElementById('who');
  const stage = new FaceStage(document.getElementById('face'), D, { emotion: 'neutral', persona: sel.value, label: 'Character face' });
  runStages([stage]);
  const logEl = document.getElementById('chatlog'), msg = document.getElementById('msg'), send = document.getElementById('send');
  const cap = document.getElementById('cap');
  let history = [];
  const add = (who, text, cls = '') => {
    if (!history.length && logEl.querySelector('.mute')) logEl.innerHTML = '';
    logEl.insertAdjacentHTML('beforeend', `<p><span class="who ${cls}">${esc(who)}</span>${esc(text)}</p>`);
    logEl.scrollTop = logEl.scrollHeight;
  };
  sel.addEventListener('change', () => { stage.setPersona(sel.value); history = []; logEl.innerHTML = ''; });

  async function go() {
    const text = msg.value.trim();
    if (!text) return;
    msg.value = '';
    const name = drawable.find(c => c.persona === sel.value)?.name || sel.value;
    if (!live) { add('Face', text, 'b'); history.push({ role: 'user', content: text }); stage.say(text); cap.textContent = ''; return; }
    add('You', text);
    history.push({ role: 'user', content: text });
    send.disabled = true; cap.textContent = `${name} is thinking...`;
    try {
      const r = await api('/chat', { body: { persona: sel.value, history } });
      if (r.refused || !r.reply) { cap.textContent = 'No reply (the model declined).'; history.pop(); return; }
      history.push({ role: 'assistant', content: r.reply });
      add(name, r.reply, 'b');
      stage.say(r.reply);
      cap.textContent = `reply from ${r.model}`;
    } catch (e) {
      history.pop();
      cap.textContent = 'Chat failed: ' + e.message;
    } finally { send.disabled = false; }
  }
  send.addEventListener('click', go);
  msg.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go(); } });
});
