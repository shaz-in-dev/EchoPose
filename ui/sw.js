/* ================================================================
   EchoPose — Service Worker
   Caches static UI assets for offline/flaky-network resilience.
   ================================================================ */

const CACHE   = 'echopose-v3';
const OFFLINE = [
  './mobile.html',
  './tablet.html',
  './index.html',
  './style.css',
  './app.js',
  './skeleton.js',
  './heatmap.js',
  './recorder.js',
  './wasm_bridge.js',
  './vendor/three.min.js',
  './vendor/OrbitControls.js',
  './manifest.json',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap'
];

// ── Install: pre-cache shell assets ──────────────────────────────
self.addEventListener('install', evt => {
  evt.waitUntil(
    caches.open(CACHE).then(cache =>
      // cache.addAll fails atomically; ignore individual errors via allSettled
      Promise.allSettled(OFFLINE.map(url => cache.add(url)))
    ).then(() => self.skipWaiting())
  );
});

// ── Activate: purge old caches ────────────────────────────────────
self.addEventListener('activate', evt => {
  evt.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// ── Fetch: network-first with cache fallback ──────────────────────
self.addEventListener('fetch', evt => {
  const url = new URL(evt.request.url);

  // Never intercept WebSocket upgrades or cross-origin API calls
  if (evt.request.url.startsWith('ws://') || evt.request.url.startsWith('wss://')) return;
  if (url.pathname.startsWith('/ws/') || url.port === '8765' || url.port === '3000') return;

  // Network-first for navigation and core assets; cache fallback
  evt.respondWith(
    fetch(evt.request)
      .then(res => {
        if (res && res.status === 200 && res.type !== 'opaque') {
          const clone = res.clone();
          caches.open(CACHE).then(cache => cache.put(evt.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(evt.request))
  );
});
