// TalkingAvatar ported from src/faceview/vision/avatar.py: a baseline FACS expression, idle behaviours (AutoBlink,
// AutoBreathing, AutoSaccade drawing from a Python-compatible Mersenne Twister) and an active utterance whose viseme
// blend drives the mouth AUs, all smoothed by exponential approach. tick(t) returns renderer FaceParams.
// Differences from the desktop class: time is always passed in explicitly (say(text, {at}) / tick(t)), so the page
// controls the clock; set the baseline from a preset object instead of a name.
import { PyRandom } from './mt19937.js';
import { AU_IDS, applyExpression, applyPersona, copyState, faceState, faceStateToParams, SpeechEngine,
  tonguePoseAt, visemeBlendAt } from './face_engine.js';

export class AutoBlink {
  static CLOSE = 0.08; static OPEN = 0.12; static MIN = 2.5; static MAX = 5.0;
  constructor(rng) { this.rng = rng; this.timer = 0; this.next = this.roll(); this.blinking = false; this.phase = 0; }
  roll() { return this.rng.uniform(AutoBlink.MIN, AutoBlink.MAX); }
  update(s, dt) {
    if (this.blinking) {
      this.phase += dt;
      const total = AutoBlink.CLOSE + AutoBlink.OPEN;
      if (this.phase < AutoBlink.CLOSE) s.blink_amount = this.phase / AutoBlink.CLOSE;
      else if (this.phase < total) s.blink_amount = Math.max(0, 1 - (this.phase - AutoBlink.CLOSE) / AutoBlink.OPEN);
      else { s.blink_amount = 0; this.blinking = false; this.timer = 0; this.next = this.roll(); }
    } else {
      this.timer += dt;
      if (this.timer >= this.next) { this.blinking = true; this.phase = 0; }
    }
  }
}

// Slow sinusoidal nostril flare (AU9) and lip part (AU25), ~0.4 Hz.
export class AutoBreathing {
  constructor() { this.phase = 0; }
  update(s, dt) {
    this.phase += dt * 2.5;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const b = (Math.sin(this.phase) + 1) * 0.5;
    s.AU9 = Math.max(s.AU9, b * 0.06);
    s.AU25 = Math.max(s.AU25, b * 0.03);
  }
}

export class AutoSaccade {
  constructor(rng) { this.rng = rng; this.nextAt = rng.uniform(1.0, 2.5); this.timer = 0; this.tx = 0; this.ty = 0; this.cx = 0; this.cy = 0; }
  update(s, dt) {
    this.timer += dt;
    if (this.timer >= this.nextAt) {
      this.tx = this.rng.uniform(-0.30, 0.30);
      this.ty = this.rng.uniform(-0.18, 0.18);
      this.timer = 0;
      this.nextAt = this.rng.uniform(1.2, 2.6);
    }
    const k = Math.min(1, dt * 6);
    this.cx += (this.tx - this.cx) * k;
    this.cy += (this.ty - this.cy) * k;
    s.eye_look_x += this.cx;
    s.eye_look_y += this.cy;
  }
}

const SMOOTHED = ['head_yaw', 'head_pitch', 'head_roll', 'eye_look_x', 'eye_look_y'];
const REST_TONGUE = [-0.85, 0.0, 0.0, 0.30];

export class TalkingAvatar {
  // expression: AU preset object; persona: {skin_hue, hair_color, ...}; dict: CMU pronouncing dictionary.
  constructor({ expression = {}, persona = null, seed = null, dict = {}, smoothingRate = 12 } = {}) {
    this.rng = new PyRandom(seed ?? Math.floor(Math.random() * 2 ** 31));
    this.state = faceState();
    this.baseline = applyExpression(faceState(), expression);
    this.target = copyState(this.baseline);
    this.blink = new AutoBlink(this.rng);
    this.breath = new AutoBreathing();
    this.saccade = new AutoSaccade(this.rng);
    this.engine = new SpeechEngine(dict);
    this.smoothingRate = smoothingRate;
    this.lastT = null; this.utterance = null; this.tongue = null;
    this.persona = persona;
    this.idle = { blink: true, breath: true, saccade: true };
    this.overrides = null;  // optional {AU: value, head_yaw: ...} added on top of the baseline (Studio sliders)
  }

  setExpression(preset) { this.baseline = applyExpression(faceState(), preset); }
  setPersona(p) { this.persona = p; }

  say(text, { speed = 1, at = 0 } = {}) {
    const timeline = this.engine.generate(text, speed);
    this.utterance = { text, start: at, timeline, duration: timeline.length ? timeline[timeline.length - 1].end : 0 };
    return this.utterance;
  }
  isSpeaking(t) { return !!this.utterance && t < this.utterance.start + this.utterance.duration; }

  tick(t) {
    const dt = this.lastT === null ? 1 / 60 : Math.max(0, Math.min(0.2, t - this.lastT));
    this.lastT = t;
    this.target = copyState(this.baseline);
    if (this.overrides) Object.assign(this.target, this.overrides);
    const u = this.utterance;
    if (u) {
      const rel = t - u.start;
      for (const [au, v] of Object.entries(visemeBlendAt(u.timeline, rel))) this.target[au] = Math.max(this.target[au], v);
      this.tongue = tonguePoseAt(u.timeline, rel) || REST_TONGUE;
      this.target.head_yaw += 0.13 * Math.sin(rel * 2.1);
      this.target.head_pitch += 0.07 * Math.sin(rel * 1.4 + 0.7);
      this.target.head_roll += 0.05 * Math.sin(rel * 1.7 + 2.3);
      if (rel >= u.duration + 0.05) { this.utterance = null; this.tongue = null; }
    } else this.tongue = null;
    // The RNG-driven systems always run so a seeded avatar stays in step with Python; the toggles only hide them.
    const keep = { blink: this.target.blink_amount, x: this.target.eye_look_x, y: this.target.eye_look_y };
    this.blink.update(this.target, dt);
    if (!this.idle.blink) this.target.blink_amount = keep.blink;
    if (this.idle.breath) this.breath.update(this.target, dt);
    else this.breath.phase = (this.breath.phase + dt * 2.5) % (2 * Math.PI);
    this.saccade.update(this.target, dt);
    if (!this.idle.saccade) { this.target.eye_look_x = keep.x; this.target.eye_look_y = keep.y; }
    const k = Math.min(1, dt * this.smoothingRate);
    for (const au of AU_IDS) this.state[au] += (this.target[au] - this.state[au]) * k;
    for (const f of SMOOTHED) this.state[f] += (this.target[f] - this.state[f]) * k;
    this.state.blink_amount = this.target.blink_amount;
    const params = faceStateToParams(this.state);
    params.tongue = this.tongue;
    return this.persona ? applyPersona(params, this.persona) : params;
  }
}
