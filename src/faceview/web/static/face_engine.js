// faceView's animation pipeline ported from Python (src/faceview/vision/): FACS Action Units -> FaceParams
// (face_state.py), expression presets (expressions.py), the 15-class viseme alphabet and tongue poses (visemes.py),
// text -> ARPAbet phonemes -> timed visemes with coarticulated blending (speech.py) and persona overlays
// (personas.py). Pure functions; the data tables that live in JSON (expressions, CMU dictionary, personas) are passed
// in from the API. Checked against Python output by face_engine.test.mjs.

export const AU_IDS = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU20', 'AU22', 'AU25', 'AU26'];
const POSE_FIELDS = ['head_yaw', 'head_pitch', 'head_roll', 'eye_look_x', 'eye_look_y', 'blink_amount'];

// FaceState (face_state.py): 12 AUs (0..1) + head pose + gaze + blink.
export function faceState() {
  const s = { skin_hue: 28, background: '#0c0f14' };
  for (const k of AU_IDS) s[k] = 0;
  for (const k of POSE_FIELDS) s[k] = 0;
  return s;
}
export const copyState = s => ({ ...s });
export const setAU = (s, au, v) => { s[au] = Math.max(0, Math.min(1, v)); };
export const resetAUs = s => { for (const k of AU_IDS) s[k] = 0; };

// FaceParams defaults (face_params.py) for the fields the stylised renderer reads.
export function faceParams(over = {}) {
  return {
    yaw: 0, pitch: 0, eye_open: 1, jaw_open: 0, smile: 0, brow_raise: 0, pupil_x: 0, pupil_y: 0,
    skin_hue: 28, background: '#0c0f14', hair_color: '#2c1810', lip_color: '#a44a4a', render_mode: 'stylised',
    mouth_pucker: 0, mouth_stretch: 0, cheek_raise: 0, nose_wrinkle: 0, upper_lid_raise: 0,
    inner_brow_raise: 0, outer_brow_raise: 0, brow_lower: 0, lip_corner_drop: 0, ...over,
  };
}

const clip = (x, lo = -1, hi = 1) => Math.max(lo, Math.min(hi, x));
const unit = x => Math.max(0, Math.min(1, x));

// face_state_to_params: FACS state -> renderer parameters.
export function faceStateToParams(s) {
  const smile = s.AU12 - s.AU15 + 0.3 * s.AU20 - 0.5 * s.AU22;
  const jaw = Math.max(0, 0.35 * s.AU25 + s.AU26);
  const brow = (s.AU1 + s.AU2) * 0.5 - s.AU4;
  let eye = Math.max(0.05, 1 - s.blink_amount);
  eye = Math.min(1.05, eye + 0.10 * s.AU5);
  return faceParams({
    yaw: clip(s.head_yaw), pitch: clip(s.head_pitch), eye_open: eye, jaw_open: unit(jaw), smile: clip(smile),
    brow_raise: clip(brow), pupil_x: clip(s.eye_look_x), pupil_y: clip(s.eye_look_y), skin_hue: s.skin_hue,
    background: s.background, mouth_pucker: unit(s.AU22), mouth_stretch: unit(s.AU20), cheek_raise: unit(s.AU6),
    nose_wrinkle: unit(s.AU9), upper_lid_raise: unit(s.AU5), inner_brow_raise: unit(s.AU1),
    outer_brow_raise: unit(s.AU2), brow_lower: unit(s.AU4), lip_corner_drop: unit(s.AU15),
  });
}

// expressions.apply_expression: replace AUs (and head pose, if the preset has it) with a weighted preset.
export function applyExpression(state, preset, weight = 1) {
  if (!preset || !Object.keys(preset).length) return state;
  resetAUs(state);
  for (const [k, v] of Object.entries(preset)) {
    if (AU_IDS.includes(k)) setAU(state, k, v * weight);
    else if (k === 'headYaw') state.head_yaw = v * weight;
    else if (k === 'headPitch') state.head_pitch = v * weight;
    else if (k === 'headRoll') state.head_roll = v * weight;
  }
  return state;
}

// personas.apply_persona: static appearance overlay.
export function applyPersona(params, persona) {
  const p = persona || {};
  params.skin_hue = p.skin_hue ?? 28;
  params.hair_color = p.hair_color ?? '#2c1810';
  params.lip_color = p.lip_color ?? '#a44a4a';
  params.background = p.background ?? '#0c0f14';
  params.render_mode = p.render_mode ?? 'stylised';
  return params;
}

// ── visemes.py ─────────────────────────────────────────────────────────
// [AU25 lips part, AU26 jaw drop, AU22 funneler, AU20 stretch, AU12 corner pull]
const V = (a25, a26, a22, a20, a12) => ({ AU25: a25, AU26: a26, AU22: a22, AU20: a20, AU12: a12 });
export const VISEMES = {
  REST: V(0, 0, 0, 0, 0), PP: V(0, 0.05, 0, 0, 0), FF: V(0.20, 0.05, 0, 0.10, 0), TH: V(0.30, 0.10, 0, 0, 0),
  DD: V(0.30, 0.10, 0, 0, 0), SS: V(0.20, 0.05, 0, 0.20, 0), SH: V(0.30, 0.10, 0.40, 0, 0), KK: V(0.20, 0.15, 0, 0, 0),
  RR: V(0.30, 0.10, 0.30, 0, 0), AA: V(0.60, 0.40, 0, 0, 0), EH: V(0.40, 0.20, 0, 0.20, 0), IH: V(0.30, 0.10, 0, 0.30, 0.10),
  OH: V(0.50, 0.30, 0.50, 0, 0), UH: V(0.30, 0.15, 0.60, 0, 0), WW: V(0.20, 0.10, 0.30, 0, 0),
};
// (extend, vertical, lateral, taper) per viseme.
export const TONGUE_POSE = {
  REST: [-0.85, 0.0, 0.0, 0.30], PP: [-0.85, -0.15, 0.0, 0.30], FF: [-0.55, -0.10, 0.0, 0.40],
  TH: [0.10, 0.20, 0.0, 0.65], DD: [-0.30, 0.55, 0.0, 0.55], SS: [-0.40, 0.45, 0.0, 0.50], SH: [-0.45, 0.30, 0.0, 0.55],
  KK: [-0.55, -0.15, 0.0, 0.30], RR: [-0.45, 0.20, 0.0, 0.55], AA: [-0.65, -0.40, 0.0, 0.30], EH: [-0.55, 0.10, 0.0, 0.30],
  IH: [-0.50, 0.50, 0.0, 0.30], OH: [-0.60, -0.20, 0.0, 0.30], UH: [-0.55, 0.20, 0.0, 0.30], WW: [-0.55, 0.10, 0.0, 0.35],
};
export const PHONEME_TO_VISEME = {
  P: 'PP', B: 'PP', M: 'PP', F: 'FF', V: 'FF', TH: 'TH', DH: 'TH', T: 'DD', D: 'DD', N: 'DD', L: 'DD', S: 'SS', Z: 'SS',
  SH: 'SH', ZH: 'SH', CH: 'SH', JH: 'SH', K: 'KK', G: 'KK', NG: 'KK', R: 'RR', W: 'WW', Y: 'WW', HH: 'WW', SIL: 'REST',
};
for (const [vow, vis] of [['ER', 'RR'], ['AA', 'AA'], ['AE', 'AA'], ['AH', 'AA'], ['AW', 'AA'], ['AY', 'AA'], ['EH', 'EH'],
  ['EY', 'EH'], ['IH', 'IH'], ['IY', 'IH'], ['AO', 'OH'], ['OW', 'OH'], ['OY', 'OH'], ['UH', 'UH'], ['UW', 'UH']]) {
  for (const suf of ['', '0', '1', '2']) PHONEME_TO_VISEME[vow + suf] = vis;
}
export function visemeForPhoneme(ph) {
  if (ph in PHONEME_TO_VISEME) return PHONEME_TO_VISEME[ph];
  return PHONEME_TO_VISEME[ph.replace(/[0-9]/g, '')] || 'REST';
}
export const visemeAUTargets = name => (VISEMES[name] ? { ...VISEMES[name] } : {});

// ── speech.py ──────────────────────────────────────────────────────────
const LETTER_RULES = {
  a: ['AE1'], b: ['B'], c: ['K'], d: ['D'], e: ['EH1'], f: ['F'], g: ['G'], h: ['HH'], i: ['IH1'], j: ['JH'], k: ['K'],
  l: ['L'], m: ['M'], n: ['N'], o: ['OW1'], p: ['P'], q: ['K', 'W'], r: ['R'], s: ['S'], t: ['T'], u: ['AH1'], v: ['V'],
  w: ['W'], x: ['K', 'S'], y: ['Y'], z: ['Z'],
};

export class SpeechEngine {
  constructor(dict = {}, { phonemeDuration = 0.085, wordGap = 0.04 } = {}) {
    this.dict = dict; this.phonemeDuration = phonemeDuration; this.wordGap = wordGap;
  }

  // Each phoneme comes back with its source: 'dict' (CMU entry) or 'rule' (letter fallback).
  words(text) {
    return (text.toLowerCase().match(/[a-z']+/g) || []).map(w => {
      const hit = Object.prototype.hasOwnProperty.call(this.dict, w) && this.dict[w].length ? this.dict[w] : null;
      return { word: w, phonemes: hit || this.ruleBased(w), source: hit ? 'dict' : 'rule' };
    });
  }

  textToPhonemes(text) {
    const out = [];
    for (const w of this.words(text)) { out.push(...w.phonemes, 'SIL'); }
    if (out.length && out[out.length - 1] === 'SIL') out.pop();
    return out;
  }

  ruleBased(word) {
    const out = [];
    for (const ch of word) if (LETTER_RULES[ch]) out.push(...LETTER_RULES[ch]);
    return out.length ? out : ['AH1'];
  }

  phonemesToVisemes(phonemes, speed = 1) {
    const dur = this.phonemeDuration / Math.max(speed, 0.1), gap = this.wordGap / Math.max(speed, 0.1);
    let t = 0;
    const out = [];
    for (const ph of phonemes) {
      if (ph === 'SIL') { out.push({ viseme: 'REST', start: t, end: t + gap, au: visemeAUTargets('REST'), phoneme: ph }); t += gap; continue; }
      const v = visemeForPhoneme(ph);
      out.push({ viseme: v, start: t, end: t + dur, au: visemeAUTargets(v), phoneme: ph });
      t += dur;
    }
    return out;
  }

  generate(text, speed = 1) { return this.phonemesToVisemes(this.textToPhonemes(text), speed); }
  duration(text, speed = 1) { const s = this.generate(text, speed); return s.length ? s[s.length - 1].end : 0; }
}

export function visemeAt(timeline, t) {
  if (!timeline.length || t < timeline[0].start || t > timeline[timeline.length - 1].end) return null;
  return timeline.find(tv => tv.start <= t && t <= tv.end) || null;
}

function visemeWeight(tv, t, attack, release) {
  if (t <= tv.start - attack || t >= tv.end + release) return 0;
  if (t < tv.start) return Math.max(0, (t - (tv.start - attack)) / Math.max(1e-6, attack));
  if (t > tv.end) return Math.max(0, 1 - (t - tv.end) / Math.max(1e-6, release));
  return 1;
}

// Weighted-max blend of overlapping visemes into one AU target dict (coarticulation).
export function visemeBlendAt(timeline, t, attack = 0.040, release = 0.060) {
  const blend = {};
  for (const tv of timeline) {
    if (t <= tv.start - attack) break;
    if (t >= tv.end + release) continue;
    const w = visemeWeight(tv, t, attack, release);
    if (w <= 0) continue;
    for (const [au, val] of Object.entries(tv.au)) {
      const c = w * val;
      if (c > (blend[au] ?? 0)) blend[au] = c;
    }
  }
  return blend;
}

// Weighted average of per-viseme tongue poses; null when no viseme is active.
export function tonguePoseAt(timeline, t, attack = 0.040, release = 0.060) {
  const ws = [], ps = [];
  for (const tv of timeline) {
    if (t <= tv.start - attack) break;
    if (t >= tv.end + release) continue;
    const w = visemeWeight(tv, t, attack, release);
    if (w <= 0) continue;
    ws.push(w); ps.push(TONGUE_POSE[tv.viseme] || TONGUE_POSE.REST);
  }
  if (!ws.length) return null;
  const total = ws.reduce((a, b) => a + b, 0);
  if (total <= 1e-6) return null;
  return [0, 1, 2, 3].map(k => ws.reduce((a, w, i) => a + w * ps[i][k], 0) / total);
}
