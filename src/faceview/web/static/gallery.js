// Desktop renders: screenshots and avatars from the desktop app committed in docs/images (captions from the README).
// They show the renderers and panels this site does not port (ICT-FaceKit 3D head, Qt docks, test mode with Ollama).
import { route, app, api, esc, list } from './core.js';

route('gallery', async () => {
  const items = await api('/gallery');
  app().innerHTML = `
  <p class="kicker">Library</p><h1 class="glow">Desktop renders</h1>
  <p class="lead">Images from the desktop app (docs/images in the repository). The browser port covers the stylised
  face; the USC ICT-FaceKit 3D head, the anatomical and photo-warp modes, the effects overlays and the Qt panels stay
  in the desktop app.</p>
  <div class="gal">${list(items, g => `<figure><img src="media/${esc(g.file)}" alt="${esc(g.alt)}" loading="lazy">
    <figcaption>${esc(g.caption)}</figcaption></figure>`)}</div>`;
});
