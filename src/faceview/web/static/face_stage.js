// Shared face plumbing for the pages: loads the animation data once, and FaceStage wraps one canvas + one
// TalkingAvatar. Pages with several faces drive them from a single animate() loop (core.js allows one per page).
import { api, animate } from './core.js';
import { TalkingAvatar } from './avatar_engine.js';
import { drawToCanvas } from './face_render.js';
import { skinColor } from './qcolor.js';

let dataP = null;
// {expr: {name: AU preset}, aus: [{id, name}], dict, personas: [..], byName: {name: persona}, stylised: [names]}
export function faceData() {
  if (!dataP) {
    dataP = Promise.all([api('/expressions'), api('/dictionary'), api('/personas')]).then(([E, dict, personas]) => {
      const byName = Object.fromEntries(personas.map(p => [p.name, p]));
      return { expr: E.presets, aus: E.action_units, dict, personas, byName,
        stylised: personas.filter(p => p.render_mode === 'stylised').map(p => p.name) };
    });
  }
  return dataP;
}

export const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export class FaceStage {
  // canvas: element; D: faceData(); opts: {emotion, persona, seed, width, height, label}
  constructor(canvas, D, { emotion = 'neutral', persona = 'claude', seed = null, width = 640, height = 480, label } = {}) {
    this.canvas = canvas; this.D = D; this.w = width; this.h = height;
    this.emotion = emotion; this.personaName = persona;
    this.avatar = new TalkingAvatar({ expression: D.expr[emotion], persona: D.byName[persona], seed, dict: D.dict });
    this.t0 = performance.now();
    this.params = null;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', label || 'Animated face');
    this.onFrame = null;  // optional callback(params, t)
  }
  now() { return (performance.now() - this.t0) / 1000; }
  setEmotion(name) { this.emotion = name; this.avatar.setExpression(this.D.expr[name] || {}); }
  setPersona(name) { this.personaName = name; this.avatar.setPersona(this.D.byName[name]); }
  say(text, speed = 1) { return this.avatar.say(text, { speed, at: this.now() }); }
  speaking() { return this.avatar.isSpeaking(this.now()); }
  frame() {
    const t = this.now();
    this.params = this.avatar.tick(t);
    drawToCanvas(this.canvas, this.params, this.w, this.h);
    if (this.onFrame) this.onFrame(this.params, t);
    return this.params;
  }
}

// Run stages until the route changes. With reduced motion, only draw while someone is speaking (user-started)
// or when poke() is called, so nothing moves on its own.
export function runStages(stages, { always = false } = {}) {
  const rm = reducedMotion();
  let dirty = true;
  const stop = animate(() => {
    if (!rm || always || dirty || stages.some(s => s.speaking())) { stages.forEach(s => s.frame()); dirty = false; }
  });
  return { stop, poke: () => { dirty = true; } };
}

// Small face swatch of a persona, for pickers.
export function personaDot(p) {
  return `<span class="pdot" style="background:${skinColor(p.skin_hue).css()};border-color:${p.hair_color}"></span>`;
}
