/* Click the map to see what the 10 m pixel at that spot holds. */
(function () {
  'use strict';
  const WM = window.WM;
  const { h, fmt, statusIcon } = WM.util;

  function title(v) {
    const lc = v.legend_class;
    const swatch = lc ? h('span', { class: 'swatch', style: { background: lc.color } }) : null;
    if (v.eudr === 'high') return [statusIcon('flag'), 'Tree crop'];
    if (v.eudr === 'low') return [statusIcon('review'), 'Tree crop'];
    return [swatch, lc ? lc.name : 'Unknown class'];
  }

  function content(latlng, v, error) {
    const where = h('p', { class: 'pixel-where' }, fmt.latlng(latlng.lat, latlng.lng));
    if (error) return h('div', null, h('p', null, `Couldn't read this spot: ${error}`), where);
    if (!v) return h('div', null, h('p', null, 'The map server is needed to read pixels.'), where);
    if (!v.covered) return h('div', null, h('p', { class: 'pixel-title' }, 'Outside the mapped area'), where);

    const lc = v.legend_class;
    const detail = v.field_class && lc && v.field_class.name.toLowerCase() !== lc.name.toLowerCase()
      ? ` (${v.field_class.name})` : '';
    const line = (label, value) => h('li', null, h('span', null, label), h('span', null, value));
    return h('div', null,
      h('p', { class: 'pixel-title' }, title(v)),
      h('ul', { class: 'pixel-lines' },
        line('2025', lc ? `${lc.name}${detail}` : '—'),
        line('Confidence', v.confidence == null ? '—'
          : `${fmt.conf(v.confidence)} (${v.confidence >= WM.source.threshold ? 'high' : 'low'})`),
        line('2020', v.forest2020 ? 'In forest baseline' : 'Not in forest baseline')),
      where);
  }

  function init(map) {
    let popup = null;
    map.on('click', async (e) => {
      popup = L.popup({ maxWidth: 340, autoPanPaddingTopLeft: [20, 70] })
        .setLatLng(e.latlng)
        .setContent(h('p', { class: 'small' }, 'Reading this spot…'))
        .openOn(map);
      const current = popup;
      let values = null, error = null;
      try {
        values = await WM.source.inspect(e.latlng.lat, e.latlng.lng);
      } catch (err) {
        error = err.message;
      }
      if (current === popup) current.setContent(content(e.latlng, values, error));
    });
  }

  WM.inspector = { init };
})();
