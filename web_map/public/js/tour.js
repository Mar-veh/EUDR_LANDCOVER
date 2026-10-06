/* "Show me around": a short guided tour of the portal, offered on the first visit. */
(function () {
  'use strict';
  const WM = window.WM;
  const { h } = WM.util;

  const SEEN_KEY = 'wm:tour-seen';
  const GHANA = [[4.7, -3.3], [11.2, 1.25]];

  // Ghana's outline on screen, for the "explore the map" step.
  function ghanaRect() {
    const map = WM.map.leaflet;
    const box = map.getContainer().getBoundingClientRect();
    const bounds = L.latLngBounds(GHANA);
    const nw = map.latLngToContainerPoint(bounds.getNorthWest());
    const se = map.latLngToContainerPoint(bounds.getSouthEast());
    const left = Math.max(box.left + nw.x, box.left), top = Math.max(box.top + nw.y, box.top);
    const right = Math.min(box.left + se.x, box.right), bottom = Math.min(box.top + se.y, box.bottom);
    return { left, top, right, bottom, width: right - left, height: bottom - top };
  }

  const STEPS = [
    {
      title: 'Welcome',
      text: 'This map shows where Ghana\'s tree crops in 2025 sit on land that was forest at the end of 2020, '
        + 'the cut-off date of the EU Deforestation Regulation (EUDR). Here is a quick look around.',
    },
    {
      target: '.panel .switch',
      title: 'Choose a map',
      text: 'EUDR overlap highlights tree crops on land that was forest in 2020. Land cover shows everything '
        + 'the model mapped in 2025.',
      before: () => WM.app.set({ mode: 'eudr' }),
    },
    {
      target: '#panel-body',
      title: 'Read the colours',
      text: 'Red: tree crop on 2020 forest where the model is confident. Amber: the same, but the model is less '
        + 'sure, so check on the ground. Green: forest in 2020, from the EU\'s forest map.',
      before: () => WM.app.set({ mode: 'eudr' }),
    },
    {
      target: ghanaRect,
      title: 'Explore the map',
      text: 'Scroll, pinch or use the + and − buttons to zoom. Zoomed out, each map pixel shows the most common '
        + 'class in its area, so small patches appear as you zoom in. Click any spot to see what is there.',
    },
    {
      target: '#fade-row',
      title: 'See where the map is reliable',
      text: 'In Land cover, turn this on to fade areas where the model is unsure. Solid colours can be trusted '
        + 'most; faded areas need a closer look.',
      before: () => WM.app.set({ mode: 'landcover' }),
    },
    {
      target: '#transparency-row',
      title: 'Adjust the transparency',
      text: 'Slide right to make the land-cover colours more transparent, so the map or satellite image shows '
        + 'underneath. Pair it with Satellite to check the map against what is on the ground.',
      before: () => WM.app.set({ mode: 'landcover' }),
    },
    {
      target: '.basemap-switch',
      title: 'Change the background',
      text: 'Switch between a plain map and satellite imagery to compare the map with what is on the ground.',
    },
    {
      target: '#about-open',
      title: 'How the map was made',
      text: 'Read about the data, the model and the limits of this map.',
    },
    {
      target: '#tour-open',
      title: 'That\'s the tour',
      text: 'You can replay it any time from here.',
    },
  ];

  let index = 0;
  let root = null, spot = null, card = null;
  let modeBefore = null, focusBefore = null;

  function seen() {
    try { return localStorage.getItem(SEEN_KEY) === '1'; } catch { return false; }
  }

  function markSeen() {
    try { localStorage.setItem(SEEN_KEY, '1'); } catch { /* storage unavailable: the invite shows again next time */ }
  }

  function button(label, className, onClick) {
    const b = h('button', { class: className, type: 'button' }, label);
    b.addEventListener('click', onClick);
    return b;
  }

  function targetRect(step) {
    if (!step.target) return null;
    if (typeof step.target === 'function') return step.target();
    const el = document.querySelector(step.target);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return r.width || r.height ? r : null;
  }

  function start() {
    if (root) return;
    closeInvite();
    markSeen();
    modeBefore = WM.state.mode;
    focusBefore = document.activeElement;
    WM.map.leaflet.closePopup();
    spot = h('div', { class: 'tour-spot', 'aria-hidden': 'true' });
    card = h('div', { class: 'tour-card', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'tour-title', 'aria-describedby': 'tour-text' });
    root = h('div', { class: 'tour' }, h('div', { class: 'tour-blocker' }), spot, card);
    document.body.append(root);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', position);
    show(0);
  }

  function end() {
    if (!root) return;
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', position);
    root.remove();
    root = spot = card = null;
    if (WM.state.mode !== modeBefore) WM.app.set({ mode: modeBefore });
    if (focusBefore && focusBefore.focus) focusBefore.focus();
  }

  function show(i) {
    index = i;
    const step = STEPS[i];
    if (step.before) step.before();
    const last = i === STEPS.length - 1;
    card.replaceChildren(
      h('p', { class: 'tour-count' }, `${i + 1} of ${STEPS.length}`),
      h('h2', { id: 'tour-title' }, step.title),
      h('p', { id: 'tour-text' }, step.text),
      h('div', { class: 'tour-actions' },
        last ? h('span') : button('Skip tour', 'btn-quiet', end),
        h('span', { class: 'tour-nav' },
          i > 0 ? button('Back', 'btn-quiet', () => show(i - 1)) : null,
          button(last ? 'Done' : 'Next', 'btn-primary', () => (last ? end() : show(i + 1))))));
    card.querySelector('.btn-primary').focus();
    // Let a layer switch re-render the panel before measuring the target.
    requestAnimationFrame(() => requestAnimationFrame(position));
  }

  function position() {
    if (!root) return;
    const rect = targetRect(STEPS[index]);
    const vw = window.innerWidth, vh = window.innerHeight;
    const cw = card.offsetWidth, ch = card.offsetHeight;
    const gap = 14, edge = 8, ring = 6;

    if (!rect) {
      Object.assign(spot.style, { left: `${vw / 2}px`, top: `${vh / 2}px`, width: '0px', height: '0px' });
      Object.assign(card.style, { left: `${(vw - cw) / 2}px`, top: `${(vh - ch) / 2}px` });
      return;
    }
    Object.assign(spot.style, {
      left: `${rect.left - ring}px`, top: `${rect.top - ring}px`,
      width: `${rect.width + 2 * ring}px`, height: `${rect.height + 2 * ring}px`,
    });

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const sides = {
      right: { x: rect.right + ring + gap, y: clamp(rect.top, edge, vh - ch - edge) },
      bottom: { x: clamp(rect.left, edge, vw - cw - edge), y: rect.bottom + ring + gap },
      left: { x: rect.left - ring - gap - cw, y: clamp(rect.top, edge, vh - ch - edge) },
      top: { x: clamp(rect.left, edge, vw - cw - edge), y: rect.top - ring - gap - ch },
    };
    const fits = (p) => p.x >= edge && p.y >= edge && p.x + cw <= vw - edge && p.y + ch <= vh - edge;
    const pos = Object.values(sides).find(fits) || { x: (vw - cw) / 2, y: vh - ch - edge };
    Object.assign(card.style, { left: `${pos.x}px`, top: `${pos.y}px` });
  }

  function onKey(e) {
    if (e.key === 'Escape') end();
    else if (e.key === 'ArrowRight' && index < STEPS.length - 1) show(index + 1);
    else if (e.key === 'ArrowLeft' && index > 0) show(index - 1);
    else return;
    e.preventDefault();
  }

  // First visit: a small invitation instead of starting the tour uninvited.
  let invite = null;
  function closeInvite() {
    if (invite) invite.remove();
    invite = null;
  }

  function offer() {
    invite = h('div', { class: 'tour-invite', role: 'dialog', 'aria-label': 'Tour' },
      h('p', null, h('b', null, 'New here? '), 'Take a one-minute tour of the map.'),
      h('div', { class: 'tour-actions' },
        button('Not now', 'btn-quiet', () => { markSeen(); closeInvite(); }),
        button('Show me around', 'btn-primary', start)));
    document.querySelector('.map-wrap').append(invite);
  }

  function init() {
    document.getElementById('tour-open').addEventListener('click', start);
    if (!seen()) offer();
  }

  WM.tour = { init, start };
})();
