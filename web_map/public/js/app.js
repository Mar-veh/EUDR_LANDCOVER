/* App state and wiring. The URL hash keeps the layer: #mode=landcover */
(function () {
  'use strict';
  const WM = window.WM;
  const { toast } = WM.util;

  const MODES = ['eudr', 'landcover'];
  const LAYER_REFRESH_MS = 20 * 60 * 1000;

  const state = (WM.state = { mode: 'eudr', basemap: 'streets', fade: false, transparency: 10 });
  const $ = (id) => document.getElementById(id);

  function set(patch) {
    Object.assign(state, patch);
    syncSwitches();
    WM.map.setBasemap(state.basemap);
    WM.panel.render();
    refreshLayers();
    history.replaceState(null, '', `#mode=${state.mode}`);
  }

  function syncSwitches() {
    for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode));
    for (const b of document.querySelectorAll('[data-basemap]')) b.setAttribute('aria-pressed', String(b.dataset.basemap === state.basemap));
  }

  // The data layer for the chosen mode, plus the 2020 forest tint under the EUDR overlap.
  let layerSeq = 0;
  let lastError = null;
  async function refreshLayers() {
    const seq = ++layerSeq;
    const specs = {
      data: state.mode === 'eudr'
        ? ['eudr', { threshold: WM.source.threshold }]
        : ['landcover', { fade: state.fade }],
      forest: state.mode === 'eudr' ? ['forest2020', {}] : null,
    };
    const opacity = { data: dataOpacity(), forest: 0.5 };
    const resolved = await Promise.all(Object.entries(specs).map(async ([pane, spec]) => {
      if (!spec) return [pane, null];
      try {
        const url = await WM.source.layerUrl(spec[0], spec[1]);
        return [pane, url ? { key: url, url, opacity: opacity[pane] } : null];
      } catch (err) {
        if (err.message !== lastError) toast(`Couldn't load the map layer: ${err.message}`);
        lastError = err.message;
        return [pane, null];
      }
    }));
    if (seq !== layerSeq) return;
    for (const [pane, spec] of resolved) WM.map.setSlot(pane, spec);
  }

  // The transparency slider applies to land cover only.
  function dataOpacity() {
    return state.mode === 'landcover' ? 1 - state.transparency / 100 : 0.9;
  }

  function setTransparency(value) {
    state.transparency = value;
    WM.map.setOpacity('data', dataOpacity());
  }

  let pending = 0;
  function loading(delta) {
    pending = Math.max(0, pending + delta);
    $('loading').hidden = pending === 0;
  }

  function showConnection() {
    const notice = $('notice');
    notice.hidden = WM.source.connected;
    if (WM.source.connected) return;
    notice.textContent = location.protocol === 'file:'
      ? 'This page needs its map server. In the web_map folder run "python -m app", then open http://127.0.0.1:8000.'
      : `The map server is not responding. Check that "python -m app" is running. (${WM.source.error})`;
  }

  async function start() {
    const fromHash = new URLSearchParams(location.hash.slice(1)).get('mode');
    if (MODES.includes(fromHash)) state.mode = fromHash;

    // On phones the panel docks at the bottom; map controls sit just above it.
    const panel = document.querySelector('.panel');
    new ResizeObserver(() => document.documentElement.style.setProperty('--panel-h', `${panel.offsetHeight}px`)).observe(panel);
    WM.panel.render();

    WM.map.init($('map'));
    WM.inspector.init(WM.map.leaflet);
    for (const b of document.querySelectorAll('[data-mode]')) b.addEventListener('click', () => set({ mode: b.dataset.mode }));
    for (const b of document.querySelectorAll('[data-basemap]')) b.addEventListener('click', () => set({ basemap: b.dataset.basemap }));
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => WM.map.setBasemap(state.basemap));

    const about = $('about');
    $('about-open').addEventListener('click', () => about.showModal());
    $('about-close').addEventListener('click', () => about.close());
    about.addEventListener('click', (e) => { if (e.target === about) about.close(); });

    syncSwitches();
    WM.map.setBasemap(state.basemap);
    WM.panel.render();
    await WM.source.init();
    showConnection();
    WM.panel.render();
    refreshLayers();
    setInterval(refreshLayers, LAYER_REFRESH_MS);
    WM.tour.init();
  }

  window.addEventListener('error', (e) => toast(`Something went wrong: ${e.message}`));
  window.addEventListener('unhandledrejection', (e) => toast(`Something went wrong: ${(e.reason && e.reason.message) || e.reason}`));

  WM.app = { set, loading, setTransparency };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
