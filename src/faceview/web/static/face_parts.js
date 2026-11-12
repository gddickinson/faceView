// Face parts for the canvas renderer, ported line by line from src/faceview/vision/sim_face_parts.py:
// stroke eyebrows (AU1/AU2/AU4), almond eyes with lids (AU5/AU6, blink, gaze), cheek apples, nose (AU9) and the
// viseme-aware mouth (jaw, smile, pucker AU22, stretch AU20, corner drop AU15) with teeth when open.
// x = geometry context {cx, cy, fw, fh, skin, skinDark}; P = Painter from face_render.js.
import { QColor } from './qcolor.js';

const rgba = (r, g, b, a = 255) => QColor.rgb(r, g, b, Math.max(0, Math.min(255, Math.trunc(a))));

export function drawBrows(P, x, p) {
  const { cx, cy, fw, fh } = x;
  const hair = QColor.hex(p.hair_color);
  const baseY = cy - fh * 0.36;
  const inner = p.inner_brow_raise * fh * 0.05, outer = p.outer_brow_raise * fh * 0.04;
  const overall = p.brow_raise * fh * 0.04, drop = p.brow_lower * fh * 0.05;
  const n = 12;
  for (const sign of [-1, 1]) {
    let ix = cx + sign * fw * 0.18;
    const ox = cx + sign * fw * 0.46;
    const iy = baseY - inner - overall + drop, oy = baseY - outer - overall + drop * 0.6;
    const ax = ix + (ox - ix) * 0.40, ay = (iy + oy) / 2 - fh * 0.025;
    ix += -sign * p.brow_lower * fw * 0.025;
    P.setPen(hair, fh * 0.018, 'round'); P.setBrush(null);
    const body = new Path2D();
    body.moveTo(ix, iy + 2); body.quadraticCurveTo(ax, ay, ox, oy);
    P.path(body);
    P.setPen(hair.lighter(110), 1.4, 'round');
    for (let i = 0; i < n; i++) {
      const f = i / Math.max(1, n - 1);
      const X = (1 - f) ** 2 * ix + 2 * (1 - f) * f * ax + f ** 2 * ox;
      const Y = (1 - f) ** 2 * iy + 2 * (1 - f) * f * ay + f ** 2 * oy;
      let tx = 2 * (1 - f) * (ax - ix) + 2 * f * (ox - ax), ty = 2 * (1 - f) * (ay - iy) + 2 * f * (oy - ay);
      const len = Math.hypot(tx, ty);
      if (len < 1e-3) continue;
      tx /= len; ty /= len;
      const sl = fh * 0.020;
      P.line(X - tx * sl * 0.4, Y - ty * sl * 0.4, X + tx * sl * 0.6, Y + ty * sl * 0.6);
    }
  }
  if (p.brow_lower > 0.4) {
    P.setPen(rgba(60, 40, 40, 120 * p.brow_lower), 1.5);
    const gy = baseY - overall + drop * 0.5;
    P.line(cx - 2, gy + 4, cx - 2, gy + fh * 0.05);
    P.line(cx + 4, gy + 4, cx + 4, gy + fh * 0.05);
  }
}

function almond(cx, cy, w, h, lift) {
  const hw = w / 2, hh = h / 2, path = new Path2D();
  path.moveTo(cx - hw, cy);
  path.quadraticCurveTo(cx, cy - hh * 1.2, cx + hw, cy);
  path.quadraticCurveTo(cx, cy + hh - lift, cx - hw, cy);
  path.closePath();
  return path;
}

export function drawEyes(P, x, p) {
  const { cx, cy, fw, fh, skinDark } = x;
  const eyeY = cy - fh * 0.18, eyeW = fw * 0.21, baseH = fh * 0.10;
  const open = Math.max(0.05, p.eye_open) + 0.20 * p.upper_lid_raise;
  const eyeH = baseH * Math.min(1.4, open);
  const lift = p.cheek_raise * fh * 0.020;
  for (const sign of [-1, 1]) {
    const ex = cx + sign * fw * 0.30;
    const al = almond(ex, eyeY, eyeW, eyeH, lift);
    P.setBrush(QColor.hex('#fafafa')); P.setPen(skinDark.darker(110), 1.2);
    P.path(al);
    if (p.eye_open > 0.18) {
      const r = Math.min(eyeW * 0.40, eyeH * 0.55);
      const ix = ex + p.pupil_x * (eyeW * 0.18), iy = eyeY + p.pupil_y * (eyeH * 0.10);
      P.save(); P.clip(al);
      P.setBrush(P.radial(ix - r * 0.2, iy - r * 0.2, r * 1.4,
        [[0, QColor.hex('#5d7aae')], [0.6, QColor.hex('#3a4a6c')], [1, QColor.hex('#1f2a45')]]));
      P.noPen();
      P.ellipse(ix, iy, r, r);
      P.setBrush(QColor.hex('#0a0a0a'));
      P.ellipse(ix, iy, r * 0.40, r * 0.40);
      P.setBrush(rgba(255, 255, 255, 220));
      P.ellipse(ix - r * 0.30, iy - r * 0.32, r * 0.18, r * 0.18);
      P.restore();
    }
    if (p.cheek_raise > 0.2) {
      P.setPen(skinDark.darker(115), 1.2); P.setBrush(null);
      const cr = new Path2D();
      cr.moveTo(ex - eyeW / 2, eyeY + eyeH / 2 - lift + 1);
      cr.quadraticCurveTo(ex, eyeY + eyeH / 2 - lift - fh * 0.005 * p.cheek_raise, ex + eyeW / 2, eyeY + eyeH / 2 - lift + 1);
      P.path(cr);
    }
    if (p.eye_open > 0.4) {
      P.setPen(QColor.hex('#1a0d0d'), 1.4, 'round');
      for (let i = 0; i < 6; i++) {
        const f = i / 5;
        const lx = ex - eyeW / 2 + f * eyeW, ly = eyeY - eyeH / 2 + Math.sin(Math.PI * f) * (-fh * 0.005);
        const ang = -0.55 + 1.1 * f;
        P.line(lx, ly, lx + Math.sin(ang) * sign * 4.5, ly - Math.cos(ang) * fh * 0.012);
      }
    }
  }
}

export function drawCheeks(P, x, p) {
  const { cx, cy, fw, fh } = x;
  const ay = cy + fh * 0.10 - p.cheek_raise * fh * 0.06;
  const alpha = Math.trunc(40 + 80 * Math.max(0, Math.max(p.cheek_raise, p.smile)));
  const r = fw * 0.16;
  for (const sign of [-1, 1]) {
    const X = cx + sign * fw * 0.42;
    P.setBrush(P.radial(X, ay, r, [[0, rgba(220, 130, 130, alpha)], [1, rgba(220, 130, 130, 0)]]));
    P.noPen();
    P.ellipse(X, ay, r, r * 0.65);
  }
}

export function drawNose(P, x, p) {
  const { cx, cy, fh, skinDark } = x;
  P.setPen(skinDark.darker(125), 1.6, 'round'); P.setBrush(null);
  const br = new Path2D();
  br.moveTo(cx - 5, cy - fh * 0.08);
  br.bezierCurveTo(cx - 12, cy, cx + 8, cy + fh * 0.04, cx + 4, cy + fh * 0.05);
  P.path(br);
  P.setBrush(skinDark.darker(115)); P.noPen();
  const tip = fh * 0.02;
  P.ellipse(cx + 3, cy + fh * 0.06, tip * 1.4, tip * 0.7);
  const nr = fh * 0.012 * (1 + 0.5 * p.nose_wrinkle);
  P.setBrush(QColor.rgb(40, 25, 25));
  for (const sign of [-1, 1]) P.ellipse(cx + sign * fh * 0.025, cy + fh * 0.075, nr, nr * 0.65);
  if (p.nose_wrinkle > 0.3) {
    P.setPen(rgba(60, 40, 40, 120 * p.nose_wrinkle), 1.2);
    P.line(cx - 6, cy - fh * 0.10, cx + 6, cy - fh * 0.10);
  }
}

// Upper-lip top edge (Cupid's bow) from the left corner to the right corner.
function upperTop(path, cx, hw, my, cdy, uh, topY, bow, cl, cr) {
  path.moveTo(cl[0], cl[1]);
  path.bezierCurveTo(cx - hw * 0.55, my + cdy - uh * 0.85, cx - hw * 0.18, topY, cx, topY + bow);
  path.bezierCurveTo(cx + hw * 0.18, topY, cx + hw * 0.55, my + cdy - uh * 0.85, cr[0], cr[1]);
}

export function drawMouth(P, x, p) {
  const { cx, cy, fw, fh } = x;
  const lip = QColor.hex(p.lip_color), lipDark = lip.darker(120), lipLight = lip.lighter(115);
  const my = cy + fh * 0.32;
  const wf = (1 + 0.45 * p.mouth_stretch + 0.25 * Math.max(0, p.smile)) * (1 - 0.55 * p.mouth_pucker);
  const mw = fw * 0.30 * Math.max(0.45, wf);
  const openH = fh * 0.18 * p.jaw_open;
  const sPos = Math.max(0, p.smile), sNeg = Math.max(0, -p.smile);
  let cdy = -p.smile * fh * 0.075 + p.lip_corner_drop * fh * 0.040;
  cdy = Math.max(-fh * 0.090, Math.min(fh * 0.060, cdy));
  const midDy = sPos * fh * 0.050 - sNeg * fh * 0.014 - p.lip_corner_drop * fh * 0.012;
  const uScale = Math.max(0.30, 1 - 0.75 * sNeg - 0.55 * p.lip_corner_drop);
  const hw = mw / 2;
  const cl = [cx - hw, my + cdy], cr = [cx + hw, my + cdy];
  const midY = my + midDy;
  const uh = fh * 0.030 * (1 + 0.6 * p.mouth_pucker) * uScale;
  const bow = fh * 0.012 * (1 - 0.4 * p.mouth_stretch);
  const lh = fh * 0.040 * (1 + 0.5 * p.mouth_pucker);

  if (openH > 4) {
    const upper = new Path2D();
    upperTop(upper, cx, hw, my, cdy, uh, midY - uh, bow, cl, cr);
    upper.lineTo(cr[0], cr[1]);
    upper.quadraticCurveTo(cx, midY - 1, cl[0], cl[1]);  // underside, reversed (connectPath + toReversed)
    upper.closePath();
    P.setBrush(lip); P.setPen(lipDark, 1);
    P.path(upper);
    const botMid = midY + openH;
    const inside = new Path2D();
    inside.moveTo(cl[0], cl[1]);
    inside.quadraticCurveTo(cx, midY - 1, cr[0], cr[1]);
    inside.quadraticCurveTo(cx, botMid - 1, cl[0], cl[1]);
    inside.closePath();
    P.setBrush(QColor.hex('#3a1a1a')); P.noPen();
    P.path(inside);
    if (openH > 14 && p.mouth_pucker < 0.6) {
      const tw = mw * 0.78, th = Math.max(4, Math.min(openH * 0.35, fh * 0.025));
      const tx = cx - tw / 2, ty = midY + 1;
      P.setBrush(P.linear(0, ty, 0, ty + th, [[0, QColor.hex('#fbf6e8')], [1, QColor.hex('#d8cfb3')]]));
      P.setPen(QColor.hex('#a89a76'), 1);
      P.roundRect(tx, ty, tw, th, 3);
      P.setPen(QColor.hex('#b6a988'), 1);
      for (let i = 1; i < 6; i++) { const xx = tx + tw * i / 6; P.line(xx, ty + 1, xx, ty + th - 1); }
    }
    const lower = new Path2D();
    lower.moveTo(cl[0], cl[1]);
    lower.quadraticCurveTo(cx, botMid - 1, cr[0], cr[1]);
    lower.bezierCurveTo(cx + hw * 0.55, botMid + lh, cx - hw * 0.55, botMid + lh, cl[0], cl[1]);
    lower.closePath();
    P.setBrush(P.linear(0, botMid, 0, botMid + lh, [[0, lipLight], [0.7, lip], [1, lipDark]]));
    P.setPen(lipDark, 1);
    P.path(lower);
  } else {
    const upper = new Path2D();
    upperTop(upper, cx, hw, my, cdy, uh, midY - uh, bow, cl, cr);
    upper.quadraticCurveTo(cx, midY, cl[0], cl[1]);
    upper.closePath();
    P.setBrush(lip); P.noPen();
    P.path(upper);
    const lower = new Path2D();
    lower.moveTo(cl[0], cl[1]);
    lower.quadraticCurveTo(cx, midY, cr[0], cr[1]);
    lower.bezierCurveTo(cx + hw * 0.55, midY + lh, cx - hw * 0.55, midY + lh, cl[0], cl[1]);
    lower.closePath();
    P.setBrush(P.linear(0, midY, 0, midY + lh, [[0, lipLight], [0.8, lip], [1, lipDark]]));
    P.noPen();
    P.path(lower);
    P.setPen(lipDark.darker(140), 1.5, 'round'); P.setBrush(null);
    const line = new Path2D();
    line.moveTo(cl[0], cl[1]); line.quadraticCurveTo(cx, midY, cr[0], cr[1]);
    P.path(line);
  }
  P.setBrush(P.radial(cx, my + fh * 0.10, fw * 0.16, [[0, rgba(0, 0, 0, 22)], [1, rgba(0, 0, 0, 0)]]));
  P.noPen();
  P.ellipse(cx, my + fh * 0.10, fw * 0.14, fh * 0.04);
}
