/* Leaflet map: basemap, place names, and the Earth Engine layers. */
(function () {
  'use strict';
  const WM = window.WM;

  const GHANA_BOUNDS = [[4.7, -3.3], [11.2, 1.25]];
  const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services';
  const CANVAS_ATTR = 'Basemap &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS user community';
  const IMAGERY_ATTR = 'Imagery &copy; Esri &mdash; Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community';
  const BASEMAPS = {
    light: { url: `${ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`, attribution: CANVAS_ATTR, maxNativeZoom: 16 },
    dark: { url: `${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, attribution: CANVAS_ATTR, maxNativeZoom: 16 },
    satellite: { url: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`, attribution: IMAGERY_ATTR, maxNativeZoom: 18 },
  };
  const LABELS = {
    light: { url: `${ESRI}/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, maxNativeZoom: 16 },
    dark: { url: `${ESRI}/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, maxNativeZoom: 16 },
    satellite: { url: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`, maxNativeZoom: 18 },
  };
  // Earth Engine layers: the 2020 forest tint under the data layer, place names on top.
  const PANES = { forest: 350, data: 380, labels: 450 };

  let map, baseLayer, labelsLayer;
  const slots = {};

  // Scale bar drawn as a ruler: a line with ticks, labelled at 0, half and the full distance.
  const RulerScale = L.Control.extend({
    options: { position: 'bottomright', maxWidth: 120 },

    onAdd(m) {
      this._el = L.DomUtil.create('div', 'ruler-scale');
      m.on('move zoom', this._update, this);
      this._update();
      return this._el;
    },

    onRemove(m) {
      m.off('move zoom', this._update, this);
    },

    _update() {
      const m = this._map;
      const y = m.getSize().y / 2;
      const maxMeters = m.distance(m.containerPointToLatLng([0, y]), m.containerPointToLatLng([this.options.maxWidth, y]));
      const pow = 10 ** Math.floor(Math.log10(maxMeters));
      const lead = maxMeters / pow;
      const meters = pow * (lead >= 5 ? 5 : lead >= 2 ? 2 : 1);
      const width = Math.round((this.options.maxWidth * meters) / maxMeters);
      const km = meters >= 1000;
      const label = (v) => `${km ? v / 1000 : v}`;
      const ticks = [0, 25, 50, 75, 100].map((p) => `<i class="${p % 50 ? 'minor' : ''}" style="left:${p}%"></i>`).join('');
      this._el.innerHTML = `
        <div class="ruler-labels" style="width:${width}px">
          <span style="left:0">0</span><span style="left:50%">${label(meters / 2)}</span><span style="left:100%">${label(meters)} ${km ? 'km' : 'm'}</span>
        </div>
        <div class="ruler-line" style="width:${width}px">${ticks}</div>`;
    },
  });

  function init(el) {
    map = L.map(el, {
      zoomControl: false,
      minZoom: 6,
      maxZoom: 18,
      // Generous enough that a tall phone screen at the minimum zoom still fits inside it.
      maxBounds: [[-12, -20], [24, 16]],
      maxBoundsViscosity: 0.8,
      zoomSnap: 0.5,
    });
    // Fit Ghana into the part of the map the floating panel leaves free.
    const narrow = window.matchMedia('(max-width: 640px)').matches;
    const panel = document.querySelector('.panel');
    map.fitBounds(GHANA_BOUNDS, narrow
      ? { paddingTopLeft: [8, 56], paddingBottomRight: [8, panel.offsetHeight + 16] }
      : { paddingTopLeft: [panel.offsetWidth + 32, 16], paddingBottomRight: [16, 16] });
    map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
    // Bottom-corner controls stack upwards in the order they are added: scale first puts it under the zoom.
    new RulerScale().addTo(map);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    for (const [name, z] of Object.entries(PANES)) {
      const pane = map.createPane(name);
      pane.style.zIndex = z;
      if (name === 'labels') pane.style.pointerEvents = 'none';
    }
    return map;
  }

  function themeKey(kind) {
    if (kind === 'satellite') return 'satellite';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function setBasemap(kind) {
    const key = themeKey(kind);
    if (baseLayer && baseLayer.wmKey === key) return;
    const cfg = BASEMAPS[key];
    const layer = L.tileLayer(cfg.url, { attribution: cfg.attribution, maxNativeZoom: cfg.maxNativeZoom, maxZoom: 18 });
    layer.wmKey = key;
    layer.addTo(map).bringToBack();
    if (baseLayer) {
      const old = baseLayer;
      layer.once('load', () => map.hasLayer(old) && map.removeLayer(old));
      setTimeout(() => map.hasLayer(old) && map.removeLayer(old), 4000);
    }
    baseLayer = layer;

    if (labelsLayer) map.removeLayer(labelsLayer);
    const labels = LABELS[key];
    labelsLayer = L.tileLayer(labels.url, { pane: 'labels', maxNativeZoom: labels.maxNativeZoom, maxZoom: 18 }).addTo(map);
  }

  // Put an Earth Engine tile layer in a pane, or empty it (spec null).
  // A new layer replaces the old one once its tiles have loaded, so the map never flashes empty.
  function setSlot(pane, spec) {
    const current = slots[pane];
    if (!spec) {
      if (current) map.removeLayer(current.layer);
      delete slots[pane];
      return;
    }
    if (current && current.key === spec.key) {
      current.layer.setOpacity(spec.opacity);
      return;
    }
    const layer = L.tileLayer(spec.url, { pane, opacity: spec.opacity, maxZoom: 18 });
    let busy = false;
    layer.on('loading', () => { if (!busy) { busy = true; WM.app.loading(+1); } });
    layer.on('load remove', () => { if (busy) { busy = false; WM.app.loading(-1); } });
    layer.addTo(map);
    if (current) {
      const old = current.layer;
      layer.once('load', () => map.hasLayer(old) && map.removeLayer(old));
      setTimeout(() => map.hasLayer(old) && map.removeLayer(old), 8000);
    }
    slots[pane] = { key: spec.key, layer };
  }

  function setOpacity(pane, value) {
    if (slots[pane]) slots[pane].layer.setOpacity(value);
  }

  WM.map = {
    init, setBasemap, setSlot, setOpacity,
    get leaflet() { return map; },
  };
})();
