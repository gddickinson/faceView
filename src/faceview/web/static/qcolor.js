// Port of the bits of Qt 6's QColor that faceView's stylised renderer uses (sim_face_parts.skin_color, darker(),
// lighter()). Qt stores colours as 16-bit channels and converts RGB<->HSV in single precision; we mirror that with
// Math.fround so the canvas renderer gets the same 8-bit colours as the desktop app (checked against
// tests/fixtures/face_colors.json, made by scripts/make_face_fixture.py with PySide6).

const F = Math.fround;
const U = 65535;
const qRound = d => (d >= 0 ? Math.trunc(F(d + 0.5)) : Math.trunc(F(d - 0.5)));
const fuzzyNull = f => Math.abs(f) <= 0.00001;
const fuzzyEq = (a, b) => F(Math.abs(F(a - b)) * 100000) <= Math.min(Math.abs(a), Math.abs(b));
const div257 = x => (x + 128 - ((x + 128) >> 8)) >> 8;  // round(x / 257)

// A colour in Qt's internal form: 16-bit RGB plus 0..255 alpha.
export class QColor {
  constructor(r16, g16, b16, a = 255) { this.r16 = r16; this.g16 = g16; this.b16 = b16; this.a = a; }

  static rgb(r, g, b, a = 255) { return new QColor(r * 0x101, g * 0x101, b * 0x101, a); }

  static hex(s) {
    const h = s.replace('#', '');
    return QColor.rgb(parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16));
  }

  // QColor.setHsvF(h, s, v): h, s, v in 0..1.
  static hsvF(h, s, v, a = 255) {
    return QColor.fromHsv16(qRound(F(h) * 36000), qRound(F(s) * U), qRound(F(v) * U), a);
  }

  static fromHsv16(hue, sat, val, a = 255) {
    if (sat === 0 || hue === U) return new QColor(val, val, val, a);
    const h = hue === 36000 ? 0 : F(hue / 6000);
    const s = F(sat / U), v = F(val / U);
    const i = Math.trunc(h);
    const f = F(h - i);
    const p = F(v * F(1 - s));
    let r, g, b;
    if (i & 1) {
      const q = F(v * F(1 - F(s * f)));
      if (i === 1) [r, g, b] = [q, v, p]; else if (i === 3) [r, g, b] = [p, q, v]; else [r, g, b] = [v, p, q];
    } else {
      const t = F(v * F(1 - F(s * F(1 - f))));
      if (i === 0) [r, g, b] = [v, t, p]; else if (i === 2) [r, g, b] = [p, v, t]; else [r, g, b] = [t, p, v];
    }
    return new QColor(qRound(F(r * U)), qRound(F(g * U)), qRound(F(b * U)), a);
  }

  toHsv16() {
    const r = F(this.r16 / U), g = F(this.g16 / U), b = F(this.b16 / U);
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const delta = F(max - min);
    const val = qRound(F(max * U));
    if (fuzzyNull(delta)) return { hue: U, sat: 0, val };
    const sat = qRound(F(F(delta / max) * U));
    let hue = 0;
    if (fuzzyEq(r, max)) hue = F(F(g - b) / delta);
    else if (fuzzyEq(g, max)) hue = F(2 + F(F(b - r) / delta));
    else if (fuzzyEq(b, max)) hue = F(4 + F(F(r - g) / delta));
    hue = F(hue * 60);
    if (hue < 0) hue = F(hue + 360);
    return { hue: qRound(F(hue * 100)), sat, val };
  }

  darker(factor = 200) {
    if (factor <= 0) return this;
    if (factor < 100) return this.lighter(Math.trunc(10000 / factor));
    const { hue, sat, val } = this.toHsv16();
    return QColor.fromHsv16(hue, sat, Math.trunc((val * 100) / factor), this.a);
  }

  lighter(factor = 150) {
    if (factor <= 0) return this;
    if (factor < 100) return this.darker(Math.trunc(10000 / factor));
    let { hue, sat, val } = this.toHsv16();
    val = Math.trunc((factor * val) / 100);
    if (val > U) { sat -= val - U; if (sat < 0) sat = 0; val = U; }
    return QColor.fromHsv16(hue, sat, val, this.a);
  }

  withAlpha(a) { return new QColor(this.r16, this.g16, this.b16, a); }
  get r() { return div257(this.r16); }
  get g() { return div257(this.g16); }
  get b() { return div257(this.b16); }
  rgbList() { return [this.r, this.g, this.b]; }
  css() { return this.a === 255 ? `rgb(${this.r},${this.g},${this.b})` : `rgba(${this.r},${this.g},${this.b},${this.a / 255})`; }
}

// sim_face_parts.skin_color(hue, lightness=0.78, saturation=0.45).
export function skinColor(hue, lightness = 0.78, saturation = 0.45) {
  const h = ((hue / 360) % 1 + 1) % 1;
  return QColor.hsvF(h, saturation, lightness);
}
