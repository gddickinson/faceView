// Python-compatible random.Random: MT19937 seeded exactly like CPython's random.seed(int), with random() and
// uniform(a, b). TalkingAvatar's idle behaviours (blink timing, saccades) draw from it, so a seeded avatar in the
// browser blinks and glances on the same frames as random.Random(seed) does in Python.

const N = 624, M = 397;

export class PyRandom {
  constructor(seed = 0) { this.mt = new Uint32Array(N); this.mti = N + 1; this.seed(seed); }

  initGenrand(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.mti = N;
  }

  // CPython: init_by_array with the absolute value of the seed split into 32-bit words.
  seed(n) {
    let v = BigInt(Math.abs(Math.trunc(n)));
    const key = [];
    do { key.push(Number(v & 0xffffffffn)); v >>= 32n; } while (v > 0n);
    this.initGenrand(19650218);
    const mt = this.mt;
    let i = 1, j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(prev, 1664525)) + key[j] + j) >>> 0;
      i++; j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k--) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(prev, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
    this.mti = N;
  }

  genrand() {
    const mt = this.mt;
    if (this.mti >= N) {
      let kk = 0, y;
      const mag = y => (y & 1 ? 0x9908b0df : 0);
      for (; kk < N - M; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ mag(y);
      }
      for (; kk < N - 1; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ mag(y);
      }
      y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ mag(y);
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  random() {
    const a = this.genrand() >>> 5, b = this.genrand() >>> 6;
    return (a * 67108864 + b) * (1 / 9007199254740992);
  }

  uniform(a, b) { return a + (b - a) * this.random(); }
}
