/* The one place the page gets data from: the web map's backend (app/main.py),
   plus the tables built from the repository (classes, calibration).

   The page is served by the backend, so API paths are same-origin. To use a
   backend on another host, set window.WM_API_BASE before this script loads. */
(function () {
  'use strict';
  const WM = window.WM;
  const BASE = (window.WM_API_BASE || '').replace(/\/$/, '');

  async function request(path) {
    let res;
    try {
      res = await fetch(`${BASE}${path}`);
    } catch {
      throw new Error('The map server is not reachable.');
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail;
      throw new Error(detail || `${res.status} ${res.statusText}`);
    }
    return body;
  }

  WM.source = {
    connected: false,
    error: null,
    tables: window.WM_TABLES,
    threshold: window.WM_TABLES.calibration.threshold,

    async init() {
      try {
        const config = await request('/api/config');
        this.threshold = config.threshold;
        this.connected = true;
      } catch (err) {
        this.error = err.message;
      }
    },

    // XYZ tile URL for an Earth Engine layer.
    async layerUrl(name, params) {
      if (!this.connected) return null;
      return (await request(`/api/layers/${name}?${new URLSearchParams(params)}`)).url;
    },

    // What the 10 m pixel under a point holds.
    async inspect(lat, lng) {
      if (!this.connected) return null;
      return request(`/api/point?${new URLSearchParams({ lon: lng, lat, threshold: this.threshold })}`);
    },
  };
})();
