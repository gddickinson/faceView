// Static-site data layer: answers api('/...') from the JSON files exported by scripts/build_static.py (every GET in
// STATIC_ENDPOINTS of faceview/web/app.py). POST /speech runs the same SpeechEngine in the browser (face_engine.js,
// node-tested against Python); live chat needs the local server and a key, so it reports itself unavailable here.
import { SpeechEngine } from './face_engine.js';

const files = new Map();
function getJson(rel) {
  if (!files.has(rel)) {
    files.set(rel, fetch('api/' + rel + '.json').then(r => {
      if (!r.ok) throw new Error(`${r.status} ${rel}`);
      return r.json();
    }));
  }
  return files.get(rel);
}

export async function staticApi(path, body) {
  const rel = path.split('?')[0].replace(/^\/+/, '');
  if (rel === 'speech') {
    const eng = new SpeechEngine(await getJson('dictionary'));
    const tl = eng.generate(body.text, body.speed ?? 1);
    return { text: body.text, speed: body.speed ?? 1, phonemes: eng.textToPhonemes(body.text),
      timeline: tl.map(t => ({ viseme: t.viseme, start: t.start, end: t.end })),
      duration: tl.length ? tl[tl.length - 1].end : 0, engine: 'browser' };
  }
  if (rel === 'chat/status') return { available: false, static: true };
  if (rel === 'chat') throw new Error('Live chat needs the local app with ANTHROPIC_API_KEY set.');
  return getJson(rel);
}
