// Canvas port of faceView's stylised face renderer (src/faceview/vision/sim_face.py: background, ears, head skin,
// hair; face_parts.js has brows, eyes, cheeks, nose and mouth from sim_face_parts.py). Same geometry, layer order and
// colours (QColor maths in qcolor.js) as the QPainter original; the Verify page shows the two side by side.
import { QColor, skinColor } from './qcolor.js';
import { drawBrows, drawCheeks, drawEyes, drawMouth, drawNose } from './face_parts.js';

// A thin QPainter-like wrapper over a 2D context: brush (fill) + pen (stroke) state, Qt's default square caps.
export class Painter {
  constructor(ctx) { this.c = ctx; this.brush = null; this.pen = null; }
  setBrush(b) { this.brush = b; }   // QColor, CanvasGradient or null
  setPen(color, width = 1, cap = 'square') { this.pen = color ? { color, width, cap } : null; }
  noPen() { this.pen = null; }
  fillStroke(path) {
    const c = this.c;
    if (this.brush) { c.fillStyle = this.brush instanceof QColor ? this.brush.css() : this.brush; c.fill(path); }
    if (this.pen) {
      c.strokeStyle = this.pen.color.css(); c.lineWidth = this.pen.width || 1;
      c.lineCap = this.pen.cap; c.lineJoin = 'bevel'; c.stroke(path);
    }
  }
  ellipse(x, y, rx, ry) { const p = new Path2D(); p.ellipse(x, y, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2); this.fillStroke(p); }
  path(p) { this.fillStroke(p); }
  line(x1, y1, x2, y2) {
    if (!this.pen) return;
    const p = new Path2D(); p.moveTo(x1, y1); p.lineTo(x2, y2);
    const b = this.brush; this.brush = null; this.fillStroke(p); this.brush = b;
  }
  rect(x, y, w, h) { const p = new Path2D(); p.rect(x, y, w, h); this.fillStroke(p); }
  roundRect(x, y, w, h, r) { const p = new Path2D(); p.roundRect(x, y, w, h, r); this.fillStroke(p); }
  radial(cx, cy, r, stops) {
    const g = this.c.createRadialGradient(cx, cy, 0, cx, cy, r);
    for (const [at, col] of stops) g.addColorStop(at, col.css());
    return g;
  }
  linear(x0, y0, x1, y1, stops) {
    const g = this.c.createLinearGradient(x0, y0, x1, y1);
    for (const [at, col] of stops) g.addColorStop(at, col.css());
    return g;
  }
  save() { this.c.save(); } restore() { this.c.restore(); } clip(p) { this.c.clip(p); }
}

function drawBackground(P, w, h, params) {
  const bg = QColor.hex(params.background);
  P.setBrush(P.radial(w / 2, h / 2, Math.max(w, h) * 0.7, [[0, bg.lighter(125)], [1, bg.darker(120)]]));
  P.noPen();
  P.rect(0, 0, w, h);
}

function drawEars(P, x) {
  for (const sign of [-1, 1]) {
    const ex = x.cx + sign * x.fw * 0.99, ey = x.cy + x.fh * 0.02;
    P.setBrush(x.skin); P.setPen(x.skinDark.darker(115), 1.2);
    P.ellipse(ex, ey, x.fw * 0.10, x.fh * 0.14);
    P.setBrush(x.skinDark.darker(135)); P.noPen();
    P.ellipse(ex, ey + x.fh * 0.01, x.fw * 0.04, x.fh * 0.07);
  }
}

function drawHeadSkin(P, x) {
  const { cx, cy, fw, fh } = x;
  P.setBrush(P.radial(cx - fw * 0.20, cy - fh * 0.30, Math.max(fw, fh) * 1.6,
    [[0, x.skin.lighter(115)], [0.55, x.skin], [1, x.skinDark.darker(110)]]));
  P.setPen(x.skinDark.darker(125), 1.5);
  P.ellipse(cx, cy, fw, fh);
  const side = P.c.createLinearGradient(cx, cy, cx + fw, cy);
  side.addColorStop(0, 'rgba(0,0,0,0)'); side.addColorStop(0.7, 'rgba(0,0,0,0)'); side.addColorStop(1, `rgba(0,0,0,${50 / 255})`);
  P.setBrush(side); P.noPen();
  P.ellipse(cx, cy, fw, fh);
}

function drawHair(P, x, params) {
  const { cx, cy, fw, fh } = x;
  const hair = QColor.hex(params.hair_color), light = hair.lighter(135);
  P.setBrush(hair); P.noPen();
  const cap = new Path2D();
  cap.moveTo(cx - fw * 1.05, cy + fh * 0.10);
  cap.quadraticCurveTo(cx - fw * 1.10, cy - fh * 0.95, cx, cy - fh * 1.05);
  cap.quadraticCurveTo(cx + fw * 1.10, cy - fh * 0.95, cx + fw * 1.05, cy + fh * 0.10);
  cap.quadraticCurveTo(cx + fw * 0.85, cy - fh * 0.40, cx + fw * 0.55, cy - fh * 0.55);
  cap.quadraticCurveTo(cx, cy - fh * 0.95, cx - fw * 0.55, cy - fh * 0.55);
  cap.quadraticCurveTo(cx - fw * 0.85, cy - fh * 0.40, cx - fw * 1.05, cy + fh * 0.10);
  cap.closePath();
  P.path(cap);
  const fr = new Path2D();
  fr.moveTo(cx - fw * 0.85, cy - fh * 0.55);
  fr.bezierCurveTo(cx - fw * 0.45, cy - fh * 0.78, cx + fw * 0.30, cy - fh * 0.72, cx + fw * 0.55, cy - fh * 0.50);
  fr.bezierCurveTo(cx + fw * 0.20, cy - fh * 0.60, cx - fw * 0.25, cy - fh * 0.55, cx - fw * 0.85, cy - fh * 0.55);
  fr.closePath();
  P.path(fr);
  P.setPen(light, 1.5, 'round'); P.setBrush(null);
  for (let i = 0; i < 7; i++) {
    const f = i / 6;
    const x0 = cx - fw * 0.85 + f * fw * 1.40, y0 = cy - fh * 0.55 - 4 + 6 * Math.sin(Math.PI * f * 1.2);
    P.line(x0, y0, x0 + 18, y0 - 14 + 4 * Math.sin(Math.PI * f));
  }
}

// render_face for render_mode "stylised": draw params into a canvas 2D context of size w x h.
export function renderFace(ctx, params, w, h) {
  const P = new Painter(ctx);
  ctx.save();
  ctx.fillStyle = QColor.hex(params.background).css();
  ctx.fillRect(0, 0, w, h);
  drawBackground(P, w, h, params);
  const fw = Math.min(w, h) * 0.38, fh = fw * 1.18;
  const cx = w / 2 + params.yaw * fw * 0.10, cy = h / 2 + params.pitch * h * 0.04;
  const x = { cx, cy, fw, fh, skin: skinColor(params.skin_hue), skinDark: skinColor(params.skin_hue, 0.62) };
  drawEars(P, x);
  drawHeadSkin(P, x);
  drawCheeks(P, x, params);
  drawHair(P, x, params);
  drawBrows(P, x, params);
  drawEyes(P, x, params);
  drawNose(P, x, params);
  drawMouth(P, x, params);
  ctx.restore();
}

// Draw the face into a canvas at a fixed pixel size (default 640x480, the desktop avatar size); CSS scales it.
export function drawToCanvas(canvas, params, lw = 640, lh = 480) {
  const ctx = canvas.getContext('2d');
  if (canvas.width !== lw || canvas.height !== lh) { canvas.width = lw; canvas.height = lh; }
  renderFace(ctx, params, lw, lh);
}
