/* Shared helpers: DOM building, formatting, toast, status icons. */
(function () {
  'use strict';
  const WM = (window.WM = window.WM || {});

  // Strings become text nodes, so nothing from the server is parsed as HTML.
  // `html` is only for the static icon markup in this file.
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value == null || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'style') Object.assign(el.style, value);
      else if (key === 'html') el.innerHTML = value;
      else el.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat(Infinity)) {
      if (child == null || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  const fmt = {
    pct(v) { return `${Math.round(v * 100)}%`; },
    conf(v) { return v == null ? '—' : v.toFixed(2); },
    latlng(lat, lng) {
      return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(4)}° ${lng >= 0 ? 'E' : 'W'}`;
    },
  };

  let toastTimer = null;
  function toast(message, ms = 8000) {
    const el = document.getElementById('toast');
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }

  const icons = {
    flag: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.3 15.2 14H.8z" fill="#d03b3b"/><path d="M8 6v3.6" stroke="#fff" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="11.7" r="0.95" fill="#fff"/></svg>',
    review: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7.2" fill="#fab219"/><path d="M8 4.4v4.2" stroke="#0b0b0b" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="11.3" r="0.95" fill="#0b0b0b"/></svg>',
  };

  function statusIcon(name) {
    return h('span', { class: 'status-icon', html: icons[name] });
  }

  WM.util = { h, fmt, toast, statusIcon };
})();
