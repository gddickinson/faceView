// Entry point: registers the lazily loaded pages, sets the footer mode text and starts the router.
import { start, route, lazy, app, isStatic, $ } from './core.js';

const PAGES = {
  'home.js': ['home'], 'studio.js': ['studio'], 'lipsync.js': ['lipsync'], 'visemes.js': ['visemes'],
  'duet.js': ['duet'], 'chat.js': ['chat'], 'characters.js': ['characters'], 'gallery.js': ['gallery'],
  'verify.js': ['verify'], 'about.js': ['about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the face runs in your browser; demo lines are scripted, no model is called'
  : 'local app (Python server)';
start();
