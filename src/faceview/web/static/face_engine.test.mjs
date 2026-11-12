// Node test: the JS ports (face_engine, avatar_engine, mt19937, qcolor) agree with faceView's Python code on the
// fixtures written by scripts/make_face_fixture.py. The comparisons live in verify_core.js (the Verify page runs the
// same ones in the browser).  Run: node src/faceview/web/static/face_engine.test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { colors, compareReference, TOL } from './verify_core.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../../..');
const json = p => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));

const sections = compareReference(json('tests/fixtures/face_reference.json'), {
  expr: json('src/faceview/assets/config/expressions.json'),
  dict: json('src/faceview/assets/data/cmu_dict_compact.json'),
  personas: json('src/faceview/assets/config/personas.json'),
});
sections.push(colors(json('tests/fixtures/face_colors.json')));
let total = 0;
for (const s of sections) {
  assert.ok(s.checks > 0, `${s.name}: no checks ran`);
  assert.equal(s.fails, 0, `${s.name}: ${s.fails} failures, first: ${s.firstFail}`);
  console.log(`ok  ${s.name}: ${s.checks} checks, max |err| ${s.maxErr.toExponential(1)}`);
  total += s.checks;
}
console.log(`face_engine: ${total} checks passed against Python (tol ${TOL})`);
