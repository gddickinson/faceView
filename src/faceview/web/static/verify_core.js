// Python-vs-JavaScript comparison, shared by the node test (face_engine.test.mjs) and the Verify page.
// REF is tests/fixtures/face_reference.json (scripts/make_face_fixture.py); expr / dict / personas are the bundled
// JSON tables. Each section returns {name, checks, fails, maxErr, firstFail}.
import { PyRandom } from './mt19937.js';
import { applyExpression, applyPersona, faceParams, faceState, faceStateToParams, SpeechEngine, tonguePoseAt,
  visemeBlendAt } from './face_engine.js';
import { TalkingAvatar } from './avatar_engine.js';
import { QColor, skinColor } from './qcolor.js';

export const TOL = 1e-9;

class Section {
  constructor(name) { this.name = name; this.checks = 0; this.fails = 0; this.maxErr = 0; this.firstFail = null; }
  num(js, py, what) {
    this.checks++;
    if (js === null || py === null || js === undefined) { if (js !== py) this.fail(`${what}: js ${js} vs py ${py}`); return; }
    const err = Math.abs(js - py);
    this.maxErr = Math.max(this.maxErr, err);
    if (!(err <= TOL * Math.max(1, Math.abs(py)))) this.fail(`${what}: js ${js} vs py ${py}`);
  }
  eq(js, py, what) { this.checks++; if (JSON.stringify(js) !== JSON.stringify(py)) this.fail(`${what}: js ${JSON.stringify(js)} vs py ${JSON.stringify(py)}`); }
  obj(js, py, what) {
    for (const k of Object.keys(py)) typeof py[k] === 'number' ? this.num(js[k] ?? 0, py[k], `${what}.${k}`) : this.eq(js[k], py[k], `${what}.${k}`);
  }
  fail(msg) { this.fails++; if (!this.firstFail) this.firstFail = msg; }
}

function mt(REF) {
  const S = new Section('Mersenne Twister (random.Random)');
  const r = new PyRandom(7);
  REF.mt.seed7_random.forEach((v, i) => S.num(r.random(), v, `random()[${i}]`));
  const r2 = new PyRandom(123456789);
  REF.mt['seed123456789_uniform_2.5_5'].forEach((v, i) => S.num(r2.uniform(2.5, 5), v, `uniform[${i}]`));
  return S;
}

function speech(REF, dict) {
  const S = new Section('Text -> phonemes -> visemes, coarticulated blend, tongue');
  const eng = new SpeechEngine(dict);
  for (const T of REF.speech.texts) {
    S.eq(eng.textToPhonemes(T.text), T.phonemes, `phonemes "${T.text}"`);
    const tl = eng.generate(T.text, T.speed);
    S.eq(tl.length, T.timeline.length, `timeline length "${T.text}"`);
    tl.forEach((tv, i) => {
      if (!T.timeline[i]) return;
      S.eq(tv.viseme, T.timeline[i][0], 'viseme'); S.num(tv.start, T.timeline[i][1], 'start'); S.num(tv.end, T.timeline[i][2], 'end');
    });
    for (const P of T.samples) {
      const b = visemeBlendAt(tl, P.t);
      S.eq(Object.keys(b).sort(), Object.keys(P.blend).sort(), `blend keys @${P.t}`);
      S.obj(b, P.blend, `blend@${P.t}`);
      const tp = tonguePoseAt(tl, P.t);
      if (P.tongue === null) S.eq(tp, null, `tongue@${P.t}`); else P.tongue.forEach((v, k) => S.num(tp?.[k] ?? null, v, `tongue@${P.t}`));
    }
  }
  return S;
}

function expressions(REF, expr, personas) {
  const S = new Section('Expressions -> FaceParams, persona overlays');
  for (const [name, py] of Object.entries(REF.expressions)) {
    const s = applyExpression(faceState(), expr[name]);
    s.blink_amount = 0.25;
    S.obj(faceStateToParams(s), py, `expr ${name}`);
  }
  for (const [name, py] of Object.entries(REF.personas)) S.obj(applyPersona(faceParams(), personas[name] || personas.default), py, `persona ${name}`);
  return S;
}

function avatar(REF, expr, dict, personas) {
  const A = REF.avatar;
  const S = new Section(`Seeded TalkingAvatar, ${A.frames.length} frames (idle + speech + emotion change)`);
  const av = new TalkingAvatar({ expression: expr[A.emotion], persona: personas[A.persona], seed: A.seed, dict });
  A.frames.forEach((py, i) => {
    const t = i / A.fps;
    if (i === A.say_at_frame) av.say(A.text, { at: t });
    if (i === A.surprised_at_frame) av.setExpression(expr.surprised);
    const p = av.tick(t);
    const { tongue, blink, ...rest } = py;
    S.obj(p, rest, `frame ${i}`);
    S.num(av.state.blink_amount, blink, `blink ${i}`);
    if (tongue === null) S.eq(p.tongue, null, `tongue ${i}`); else tongue.forEach((v, k) => S.num(p.tongue?.[k] ?? null, v, `tongue ${i}`));
  });
  return S;
}

// Exact 8-bit match of the QColor port (needs tests/fixtures/face_colors.json).
export function colors(C) {
  const S = new Section('QColor skin tones, darker() / lighter()');
  for (const s of C.skins) S.eq(skinColor(s.hue, s.lightness).rgbList(), s.rgb, `skin ${s.hue}`);
  for (const o of C.ops) {
    const c = o.hex ? QColor.hex(o.hex) : QColor.rgb(...o.rgb);
    S.eq(c.darker(o.factor).rgbList(), o.darker, `darker ${o.hex || o.rgb} ${o.factor}`);
    S.eq(c.lighter(o.factor).rgbList(), o.lighter, `lighter ${o.hex || o.rgb} ${o.factor}`);
  }
  return S;
}

export function compareReference(REF, { expr, dict, personas }) {
  return [mt(REF), speech(REF, dict), expressions(REF, expr, personas), avatar(REF, expr, dict, personas)];
}
