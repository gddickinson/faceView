// About: what faceView is, what this site does and does not include, how to run the desktop and web apps, credits.
import { route, app, api, esc } from './core.js';

route('about', async () => {
  const info = await api('/info');
  app().innerHTML = `
  <p class="kicker">About</p><h1 class="glow">faceView</h1>
  <p class="lead">${esc(info.summary)}</p>
  <h2>What runs where</h2>
  <div class="two">
    <div class="panel"><h3>In your browser (this site)</h3>
      <p>The avatar's face: the 12-AU FACS expression model and ${info.expressions} presets, idle blinks, breathing and
      saccades, text-to-viseme lip-sync with coarticulation (${info.dictionary_words}-word CMU dictionary plus letter
      rules) and the stylised renderer, ported from <span class="mono">src/faceview/vision/</span> to JavaScript and
      canvas. A seeded avatar matches the Python one frame by frame (<a href="#/verify">Verify</a>).</p>
      <p>Everything spoken here is a <span class="label-demo">scripted</span> line from the repository: test mode's
      canned prompts and the characters' catchphrases. No model is called and nothing you type leaves the page.</p></div>
    <div class="panel"><h3>Desktop app and local web app only</h3>
      <p>Chat with Claude (Anthropic API) or a local Ollama model, the Kokoro neural voice, speech recognition from
      the microphone, the webcam with presence, identity, emotion and mouth tracking, per-persona memory, the 3D
      ICT-FaceKit head and other renderers, effects, and the control API / MCP server.</p>
      <p>These need API keys, models or devices on your machine. Keys are read from the environment
      (<span class="mono">ANTHROPIC_API_KEY</span>) by the Python process and are never sent to a browser.</p></div>
  </div>
  <h2>Run the desktop app</h2>
  <pre>conda activate faceview
pip install -e ".[dev,speech,vision]"   # add identity,emotion,mcp as wanted
faceview                                  # launch the GUI</pre>
  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py                         # http://127.0.0.1:8199
# live chat: also  pip install anthropic  and start it with ANTHROPIC_API_KEY set
# (model: FACEVIEW_MODEL, default as in src/faceview/config.py)</pre>
  <p>The local web app adds the Python speech-timeline cross-check on the Lip-sync page and live chat on the Chat
  page. <span class="mono">python scripts/build_static.py</span> exports this static site.</p>
  <h2>Not ported</h2>
  <p>The anatomical, photo-warp and 3D renderers (USC ICT-FaceKit, MakeHuman, BodyParts3D), the visual effects
  (tears, blush, glitch...), mirror mode, the talking-tongue effect, and the cognition/memory layer. See
  <a href="#/gallery">Desktop renders</a> for what they look like.</p>
  <h2>Credits</h2>
  <p>faceView by George Dickinson (MIT licence). The FACS Action Units, expression presets, viseme alphabet and
  CMU dictionary subset are adapted from the FaceForge anatomy app's animation system. From the README:
  <a href="https://github.com/USC-ICT/ICT-FaceKit">USC ICT-FaceKit</a> (photo-real avatar mesh and blendshapes),
  <a href="https://github.com/thewh1teagle/kokoro-onnx">Kokoro-onnx</a> (local neural TTS),
  <a href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</a>,
  <a href="https://github.com/snakers4/silero-vad">silero-vad</a>,
  <a href="https://github.com/deepinsight/insightface">InsightFace</a>,
  <a href="https://google.github.io/mediapipe/">MediaPipe</a> and
  <a href="https://github.com/serengil/deepface">DeepFace</a>; the cognition architecture is adapted from the
  autonomous_world NPC memory system and the table_games Living-AI design.</p>`;
});
