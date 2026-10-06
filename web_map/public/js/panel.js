/* The floating panel: one sentence and a legend for the layer on the map. */
(function () {
  'use strict';
  const WM = window.WM;
  const { h, fmt } = WM.util;

  function legendItem(color, label, note, tint) {
    return h('li', null,
      h('span', { class: `swatch${tint ? ' is-tint' : ''}`, style: { background: color } }),
      h('span', null, label, note ? h('small', null, note) : null));
  }

  // Tree crop on 2020 forest, from the national statistics, if they have been computed.
  function headline() {
    const stats = WM.source.stats();
    if (!stats) return null;
    const sum = (a) => a.reduce((x, y) => x + y, 0);
    const n = stats.national;
    const onForest = sum(n.tc_forest_bins);
    const allTree = onForest + sum(n.tc_outside_bins);
    return h('p', { class: 'headline' }, h('b', null, fmt.ha(onForest)),
      ` of tree crop in Ghana is on 2020 forest (${fmt.pct(onForest / allTree)} of all tree crop).`);
  }

  function eudr() {
    const t = fmt.conf(WM.source.threshold);
    return [
      h('p', { class: 'panel-text' },
        'Tree crops mapped in 2025 on land that was forest at the end of 2020, the EUDR cut-off date.'),
      headline(),
      h('ul', { class: 'legend' },
        legendItem('var(--map-high)', 'Tree crop on 2020 forest', `Model is confident (${t} or more)`),
        legendItem('var(--map-low)', 'Tree crop on 2020 forest', 'Model is less sure: check on the ground'),
        legendItem('var(--map-forest)', 'Forest in 2020', 'EU forest map (JRC GFC2020)', true)),
    ];
  }

  function toggle(name, label, note) {
    const input = h('input', { type: 'checkbox', id: `${name}-switch`, checked: WM.state[name] || null });
    input.addEventListener('change', () => WM.app.set({ [name]: input.checked }));
    return h('label', { class: 'toggle', id: `${name}-row` },
      input,
      h('span', { class: 'toggle-track', 'aria-hidden': 'true' }),
      h('span', null, label, h('small', null, note)));
  }

  // Updates the map while dragging; the panel is not re-rendered, so the slider keeps focus.
  function transparencySlider() {
    const value = WM.state.transparency;
    const out = h('span', { class: 'slider-value' }, `${value}%`);
    const input = h('input', { type: 'range', id: 'transparency-slider', min: 0, max: 100, step: 5, value });
    input.addEventListener('input', () => {
      out.textContent = `${input.value}%`;
      WM.app.setTransparency(Number(input.value));
    });
    return h('div', { class: 'slider', id: 'transparency-row' },
      h('div', { class: 'slider-head' }, h('label', { for: 'transparency-slider' }, 'Transparency'), out),
      input,
      h('small', null, 'Slide right to see the map or satellite image underneath.'));
  }

  function landcover() {
    const { classes, calibration: cal } = WM.source.tables;
    return [
      h('p', { class: 'panel-text' }, 'Land cover in 2025, mapped by TabPFN-3.5 at 10 m.'),
      toggle('fade', 'Fade areas where the model is unsure', WM.state.fade
        ? `Solid colour: the model is sure. In testing it was right ${fmt.pct(cal.accuracy_at_or_above)} of the time `
          + `at confidence ${fmt.conf(cal.threshold)} or more, and ${fmt.pct(cal.accuracy_below)} below that.`
        : 'Shows where the map can be trusted most.'),
      transparencySlider(),
      h('ul', { class: 'legend-grid' }, classes.legend.map((c) =>
        h('li', { class: c.code === classes.tree_crop_code ? 'is-key' : null, title: c.name },
          h('span', { class: 'swatch', style: { background: c.color } }), c.name))),
    ];
  }

  const builders = { eudr, landcover };

  function render() {
    const body = document.getElementById('panel-body');
    const focusId = body.contains(document.activeElement) ? document.activeElement.id : null;
    body.replaceChildren(...builders[WM.state.mode]().filter(Boolean));
    if (focusId && document.getElementById(focusId)) document.getElementById(focusId).focus();
  }

  WM.panel = { render };
})();
