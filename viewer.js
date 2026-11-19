'use strict';
(() => {
  const VER = '?v=4';   // bump when the data layout changes, so that no cached file of an earlier version is used
  const N = 15360, GW = 160, GH = 96, PW = 640, PH = 384, PN = PW * PH, UMAX = 5;
  const NODE_HEX = ['#2563eb', '#ea7c0a', '#059669', '#db2777', '#7c3aed'];
  const BG = [236, 241, 247];
  const BW_LABEL = { 1: '1', 0.25: '1/4', 0.0625: '1/16' };
  const MODES = [
    { id: 'rgb', t: 'Image', g: 0, d: 'Camera images' },
    { id: 'gt', t: 'Ground truth', g: 0, d: 'Semantic labels' },
    { id: 'dec', t: 'Node decision', g: 0, d: 'Class decisions of the node encoders, sent on the digital layer' },
    { id: 'fc', t: 'Prediction', g: 0, d: 'World-model prediction before any data of the frame arrives' },
    { id: 'fu', t: 'Fused', g: 0, d: 'Decisions of the centre after two-layer decoding and fusion' },
    { id: 'err', t: 'Errors', g: 0, d: 'Fused decisions that differ from the ground truth' },
    { id: 'lv', t: 'Uncertainty', g: 1, d: 'Uncertainty level broadcast by the centre before the frame (8 levels)' },
    { id: 'k', t: 'Analog rows', g: 1, d: 'Measurement rows sent per token' },
    { id: 'rx', t: 'Arrival', g: 1, d: 'Whether the token decision reached the centre' },
    { id: 'nl', t: 'JEPA error', g: 1, d: 'Predictive NLL of the transmitted latent (nat/dim)' },
  ];
  const PAL = {
    uav: { names: ['Road', 'Ground', 'Building', 'Vegetation', 'Car', 'Bus', 'Truck', 'Pole', 'Furniture'],
      colors: [[124, 108, 214], [222, 196, 140], [140, 158, 196], [38, 166, 91], [37, 99, 235], [245, 158, 11], [234, 88, 12], [71, 85, 105], [250, 204, 21], [14, 165, 233]] },
    v2x: { names: ['Road', 'Sidewalk', 'Building', 'Vegetation', 'Vehicle', 'Pedestrian', 'Fence', 'Pole/sign', 'Other'],
      colors: [[124, 108, 214], [222, 196, 140], [140, 158, 196], [38, 166, 91], [37, 99, 235], [244, 63, 94], [180, 140, 100], [250, 204, 21], [45, 212, 191], [14, 165, 233]] },
  };
  PAL.uav5 = PAL.uav;
  const ERR = [225, 29, 72], RX = [245, 158, 11], NORX = [203, 213, 225], UNK = [203, 213, 225];

  const $ = (s, r = document) => r.querySelector(s);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const range = n => Array.from({ length: n }, (_, i) => i);
  const p3 = i => String(i).padStart(3, '0');
  const rgb = (c, a) => a == null ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const air = () => S.ds.id !== 'v2x';
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v;
      else if (k === 'style') Object.assign(e.style, v); else e.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  }
  function lut(stops) {
    const out = new Uint8Array(768);
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * (stops.length - 1), j = Math.min(stops.length - 2, Math.floor(t)), f = t - j;
      for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.round(stops[j][c] + (stops[j + 1][c] - stops[j][c]) * f);
    }
    return out;
  }
  const CMAP = {
    viridis: lut([[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37]]),
    purples: lut([[245, 240, 255], [221, 206, 250], [192, 164, 243], [158, 115, 230], [124, 72, 210], [94, 38, 180], [62, 16, 130]]),
    ylorrd: lut([[255, 250, 210], [254, 230, 150], [254, 196, 90], [252, 141, 60], [240, 80, 40], [205, 25, 35], [140, 0, 38]]),
  };
  const cm = (name, t) => { const L = CMAP[name], i = clamp(Math.round(t * 255), 0, 255) * 3; return [L[i], L[i + 1], L[i + 2]]; };
  const cmCss = name => `linear-gradient(90deg,${range(13).map(i => rgb(cm(name, i / 12)) + ` ${(100 * i / 12).toFixed(1)}%`).join(',')})`;

  // ---------------------------------------------------------------- data
  // A scene comes in chunks of `chunk` frames: the shared data and geometry at 640x384 (tex) or 320x192 (tex2), the fused
  // decisions of one setting (c), and the other data of that setting for the views that need them (x). The compressed
  // chunks of the open scene stay in memory (NET.raw) and are fetched in play order from the playhead, spread over GitHub
  // Pages and mirrors of the same commit. Playback streams the full resolution when the measured speed sustains it and
  // the half resolution otherwise; a paused frame is brought to the full resolution.
  const LOW = (navigator.deviceMemory || 8) <= 4 || matchMedia('(pointer:coarse)').matches;
  const NET = { raw: new Map(), live: new Map(), fail: new Map(), bytes: 0, budget: (LOW ? 200 : 480) * 2 ** 20, slots: 4, hist: [], t0: 0, idle: 0, hold: false,
    size: {} };
  const HOSTS = [{ base: '', n: 0, off: 0, rate: 0 }];
  function setMirrors(sha) {
    if (!sha) return;
    const repo = 'jepa-wm-semcom/jepa-wm-semcom.github.io';
    for (const b of [`https://raw.githubusercontent.com/${repo}/${sha}/`, `https://testingcf.jsdelivr.net/gh/${repo}@${sha}/`,
      `https://gcore.jsdelivr.net/gh/${repo}@${sha}/`, `https://fastly.jsdelivr.net/gh/${repo}@${sha}/`, `https://cdn.jsdelivr.net/gh/${repo}@${sha}/`])
      HOSTS.push({ base: b, n: 0, off: 0, rate: 0 });
    NET.slots = 8;
  }
  function host(skip) {   // the usable host with the fewest requests in flight, then the highest measured speed
    let best = null; const t = performance.now();
    for (const h of HOSTS) if (h.off < t && h !== skip && h.n < 2 && (!best || h.n < best.n || (h.n === best.n && h.rate > best.rate))) best = h;
    return best || HOSTS[0];
  }
  const DT = { u1: Uint8Array, i1: Int8Array, u2: Uint16Array, i2: Int16Array, u4: Uint32Array, i4: Int32Array, f4: Float32Array };
  const tag = (s = S.snr, b = S.bw) => `snr${s}_bw${b}`;
  const base = () => `data/${S.ds.id}/${S.clip.id}/`;
  const grp = f => Math.floor(f / S.clip.chunk);
  const urlC = (g, tg = tag()) => base() + `${tg}/c${p3(g)}.bin`, urlX = g => base() + `${tag()}/x${p3(g)}.bin`;
  const urlT = (g, hi) => base() + `${hi ? 'tex' : 'tex2'}/t${p3(g)}.bin`;
  const needX = () => ['k', 'rx', 'fc', 'nl', 'lv'].includes(S.mode) || S.layers.cells;
  const urls = (f, hi) => { const g = grp(f), u = [urlT(g, hi), urlC(g)]; if (needX()) u.push(urlX(g)); return u; };
  const has = (f, hi) => urls(f, hi).every(u => NET.raw.has(u));
  const hiPlay = () => S.res === 'full' || (S.res === 'auto' && S.streamHi);
  const isBuf = f => has(f, hiPlay());
  const kind = u => u.includes('/tex/') ? 't' : u.includes('/tex2/') ? 't2' : /\/x\d+\.bin$/.test(u) ? 'x' : 'c';
  function wantList() {
    const n = S.clip.n, s = clamp(S.want, 0, n - 1), out = new Set(), hp = hiPlay(), lead = Math.min(n, 3 * S.clip.chunk);
    for (let k = 0; k < lead; k++) for (const u of urls((s + k) % n, hp)) out.add(u);   // the next frames in the streamed resolution,
    if (!S.playing && S.res !== 'fast') for (const u of urls(s, true)) out.add(u);     // then a paused frame at full resolution
    for (let k = lead; k < n; k++) for (const u of urls((s + k) % n, hp)) out.add(u);
    if (!hp && S.res !== 'fast') for (let k = 0; k < n; k++) out.add(urlT(grp((s + k) % n), true));
    if (!LOW) for (const sn of S.D.snrs) for (const bw of S.D.bws) out.add(urlC(grp(s), tag(sn, bw)));
    return [...out];
  }
  function speed(win) {   // bytes per second over the last `win` ms of download activity
    const t = performance.now(), H = NET.hist;
    while (H.length && H[0][0] < t - 20000) H.shift();
    let b = 0; for (const x of H) if (x[0] >= t - win) b += x[1];
    return b ? b / Math.max(1000, t - Math.max(NET.t0, t - win)) * 1000 : 0;
  }
  function bytesPerFrame(hi) {
    const z = NET.size, m = k => (z[k] ? z[k][0] / z[k][1] : 0) / S.clip.chunk;
    const t = hi ? (m('t') || 2.8 * m('t2')) : (m('t2') || m('t') / 2.8);
    return m('c') + (needX() ? m('x') : 0) + t;
  }
  // One download of url from host h; aborted when no byte arrives for 8 s.
  async function get(url, h, job) {
    const ctl = new AbortController(), stop = () => ctl.abort(), t0 = performance.now();
    let seen = t0, n = 0;
    const dog = setInterval(() => { if (performance.now() - seen > 8000) ctl.abort(); }, 1000);
    job.ctl.signal.addEventListener('abort', stop); h.n++;
    try {
      const r = await fetch(h.base ? h.base + url : url + VER, { signal: ctl.signal });
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      const parts = [], rd = r.body.getReader();
      for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); n += value.length; seen = performance.now(); NET.hist.push([seen, value.length]); }
      const b = new Uint8Array(n); n = 0;
      for (const v of parts) { b.set(v, n); n += v.length; }
      const r1 = b.length / Math.max(0.05, (performance.now() - t0) / 1000); h.rate = h.rate ? 0.7 * h.rate + 0.3 * r1 : r1;
      return b;
    } finally { h.n--; clearInterval(dog); job.ctl.signal.removeEventListener('abort', stop); }
  }
  function fetchRaw(url, urgent) {
    if (NET.raw.has(url)) return Promise.resolve(NET.raw.get(url));
    let job = NET.live.get(url);
    if (job) { job.urgent = job.urgent || urgent; return job.p; }
    if (!NET.live.size && performance.now() - NET.idle > 1000) NET.t0 = performance.now();
    job = { ctl: new AbortController(), clip: S.clip, urgent };
    job.p = (async () => {   // a mirror that fails or stalls is left out for 30 s and the chunk is fetched from another host
      let last = null;
      for (let k = 0; ; k++) {
        const h = host(last);
        try { return await get(url, h, job); }
        catch (e) { if (job.ctl.signal.aborted || k >= 2) throw e; if (h.base) h.off = performance.now() + 30000; last = h; }
      }
    })();
    NET.live.set(url, job);
    const done = () => { if (NET.live.get(url) === job) NET.live.delete(url); if (!NET.live.size) NET.idle = performance.now(); pump(); };
    job.p.then(b => { if (job.clip === S.clip) store(url, b); done(); netChanged(); },
      () => { if (!job.ctl.signal.aborted) { NET.fail.set(url, performance.now() + 4000); setTimeout(pump, 4100); } done(); });
    return job.p;
  }
  function store(url, b) {
    NET.raw.set(url, b); NET.bytes += b.length;
    const z = NET.size[kind(url)] || (NET.size[kind(url)] = [0, 0]); z[0] += b.length; z[1]++;
    if (NET.bytes <= NET.budget) return;
    const keep = new Set(wantList());
    for (const [k, v] of NET.raw) if (NET.bytes > NET.budget && !keep.has(k)) { NET.raw.delete(k); NET.bytes -= v.length; }
  }
  function pump() {
    if (!S.clip || NET.hold) return;
    const t = performance.now();
    for (const u of wantList()) {
      if (NET.live.size >= NET.slots) break;
      if (!NET.raw.has(u) && !NET.live.has(u) && !(NET.fail.get(u) > t)) fetchRaw(u, false).catch(() => {});
    }
  }
  function resetNet() {
    for (const j of NET.live.values()) j.ctl.abort();
    NET.live.clear(); NET.raw.clear(); NET.fail.clear(); NET.bytes = 0; NET.size = {};
    dec.clear(); prep.clear(); geoCache.clear(); bmpCache.clear();
  }
  const worker = new Worker('worker.js' + VER), wjobs = new Map();
  let wid = 0;
  worker.onmessage = e => { const j = wjobs.get(e.data.id); wjobs.delete(e.data.id); if (e.data.err) j.bad(new Error(e.data.err)); else j.ok(e.data); };
  const unpack = gz => new Promise((ok, bad) => { const id = ++wid, b = gz.slice().buffer; wjobs.set(id, { ok, bad }); worker.postMessage({ id, buf: b }, [b]); });
  const dec = new Map();
  function chunk(url) {
    if (dec.has(url)) { const v = dec.get(url); dec.delete(url); dec.set(url, v); return v; }
    const job = fetchRaw(url, true).then(unpack).then(m => {
      if (m.tex) return { meta: m.meta, arr: m.tex };
      const arr = {};
      for (const [k, s] of Object.entries(m.spec)) arr[k] = new DT[s.dtype](m.raw, s.offset, s.nbytes / DT[s.dtype].BYTES_PER_ELEMENT);
      return { meta: m.meta, arr };
    });
    dec.set(url, job); job.catch(() => dec.delete(url));
    let nt = 0, n2 = 0, no = 0;
    for (const k of [...dec.keys()].reverse()) { const t = kind(k); if (t === 't' ? ++nt > (LOW ? 2 : 3) : t === 't2' ? ++n2 > 4 : ++no > 16) dec.delete(k); }
    return job;
  }

  async function loadFrame(i, hi) {
    const U = S.U, g = grp(i), f = i - g * S.clip.chunk, x = needX();
    const [tx, st, ex] = await Promise.all([chunk(urlT(g, hi)), chunk(urlC(g)), x ? chunk(urlX(g)) : null]);
    const o = f * U * N, sl = a => a.subarray(o, o + U * N), jo = tx.arr.jpg_off, P = tx.arr.w * tx.arr.h, s0 = f * U * P;
    const fr = {
      i, hi, t: tx.meta.frames[f].t, depth: sl(tx.arr.tok), gt: sl(tx.arr.gt), dec: sl(tx.arr.dec), cams: tx.arr.cams.subarray(f * 16 * U, (f + 1) * 16 * U),
      fu: sl(st.arr.fu), c0: st.meta.c0, meta: st.meta.frames[f],
      jpg: range(U).map(u => tx.arr.jpg.subarray(jo[(f * U + u) * 2], jo[(f * U + u) * 2] + jo[(f * U + u) * 2 + 1])), mime: tx.meta.img === 'webp' ? 'image/webp' : 'image/jpeg',
      lab: tx.arr.lab ? tx.arr.lab.subarray(s0, s0 + U * P) : null, deps: range(U).map(u => tx.arr.dep.subarray(s0 + u * P, s0 + (u + 1) * P)),
    };
    if (ex) {
      const off = ex.arr.cell_off;
      Object.assign(fr, { k: sl(ex.arr.k), fl: sl(ex.arr.fl), fc: sl(ex.arr.fc), nl: sl(ex.arr.nl), cells: ex.arr.cells.subarray(off[f] * 3, off[f + 1] * 3), cellv: ex.arr.cellv.subarray(off[f], off[f + 1]) });
    }
    return fr;
  }
  const prep = new Map();
  function prepare(i, hi) {
    const clip = S.clip, key = `${clip.id}/${tag()}/${hi ? 'H' : 'L'}${needX() ? 'x' : ''}/${i}`;
    if (prep.has(key)) return prep.get(key);
    const job = (async () => {
      const fr = await loadFrame(i, hi);
      if (S.clip !== clip) throw new Error('stale');
      fr.bmps = await bitmaps(fr, clip);
      if (S.clip !== clip) throw new Error('stale');
      fr.geo = geometry(fr);
      return fr;
    })();
    prep.set(key, job); job.catch(() => prep.delete(key));
    while (prep.size > (LOW ? 3 : 4)) prep.delete(prep.keys().next().value);
    return job;
  }
  const geoCache = new Map();
  function geometry(fr) {
    const key = S.clip.id + '/' + fr.i;
    if (geoCache.has(key)) return geoCache.get(key);
    const U = S.U, pos = new Float32Array(U * N * 3), valid = new Uint8Array(U * N);
    const vx = new Int32Array(U * N), vy = new Int32Array(U * N), vz = new Int32Array(U * N);
    const [O0, O1, O2] = S.O, [V0, V1, V2] = S.Ov, dmed = [], ys = [];
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (let u = 0; u < U; u++) {
      const c = fr.cams.subarray(u * 16, u * 16 + 16), ds = [];
      for (let j = 0; j < N; j++) {
        const t = u * N + j, d = fr.depth[t] / 100;
        if (!d) continue;
        const b = ((j % GW) * 4 + 2 - c[2]) / c[0] * d, e = (((j / GW) | 0) * 4 + 2 - c[3]) / c[1] * d;
        const wx = c[13] + c[4] * d + c[5] * b + c[6] * e, wy = c[14] + c[7] * d + c[8] * b + c[9] * e, wz = c[15] + c[10] * d + c[11] * b + c[12] * e;
        valid[t] = 1; pos[t * 3] = wx - O0; pos[t * 3 + 1] = -(wz - O2); pos[t * 3 + 2] = wy - O1;
        vx[t] = Math.floor(wx / 0.5) - V0; vy[t] = Math.floor(wy / 0.5) - V1; vz[t] = Math.floor(wz / 0.5) - V2;
        if ((j & 7) === 0) { ds.push(d); sx += pos[t * 3]; sy += pos[t * 3 + 1]; sz += pos[t * 3 + 2]; n++; ys.push(pos[t * 3 + 1]); }
      }
      ds.sort((p, q) => p - q); dmed.push(ds.length ? ds[Math.floor(ds.length * 0.55)] : 20);
    }
    ys.sort((p, q) => p - q);
    const geo = { pos, valid, vx, vy, vz, dmed, centre: n ? [sx / n, sy / n, sz / n] : [0, 0, 0], ground: ys.length ? ys[Math.floor(ys.length * 0.04)] : 0 };
    geoCache.set(key, geo);
    while (geoCache.size > 8) geoCache.delete(geoCache.keys().next().value);
    return geo;
  }
  function tokenLevels(fr, geo) {
    if (fr._lv) return fr._lv;
    const map = new Map(), c = fr.cells, v = fr.cellv, [V0, V1, V2] = S.Ov, c0 = fr.c0, out = new Uint8Array(S.U * N).fill(255);
    for (let q = 0; q < v.length; q++) map.set(((c[q * 3] + 32768) * 65536 + (c[q * 3 + 1] + 32768)) * 65536 + (c[q * 3 + 2] + 32768), v[q]);
    for (let t = 0; t < S.U * N; t++) {
      if (!geo.valid[t]) continue;
      const cx = Math.floor((geo.vx[t] + V0 + 0.5) / 4) - c0[0], cy = Math.floor((geo.vy[t] + V1 + 0.5) / 4) - c0[1], cz = Math.floor((geo.vz[t] + V2 + 0.5) / 4) - c0[2];
      const l = map.get(((cx + 32768) * 65536 + (cy + 32768)) * 65536 + (cz + 32768));
      out[t] = l === undefined ? 255 : l;
    }
    return (fr._lv = out);
  }
  const bmpCache = new Map();
  function bitmaps(fr, clip) {
    const key = `${clip.id}/${fr.hi ? 'H' : 'L'}/${fr.i}`;
    if (!bmpCache.has(key)) bmpCache.set(key, Promise.all(fr.jpg.map(b => createImageBitmap(new Blob([b], { type: fr.mime })))));
    while (bmpCache.size > 8) bmpCache.delete(bmpCache.keys().next().value);
    return bmpCache.get(key);
  }
  function miou(cm) {
    if (!cm) return NaN;
    let s = 0, n = 0;
    for (let c = 0; c < 9; c++) {
      let tr = 0, pr = 0; for (let j = 0; j < 11; j++) tr += cm[c * 11 + j];
      for (let r = 0; r < 10; r++) pr += cm[r * 11 + c];
      if (tr > 0) { s += cm[c * 11 + c] / Math.max(tr + pr - cm[c * 11 + c], 1); n++; }
    }
    return n ? 100 * s / n : NaN;
  }

  // ---------------------------------------------------------------- colours
  function tokenColors(mode, fr, geo, out) {
    const pal = PAL[S.ds.id].colors, lv = mode === 'lv' && fr.cells ? tokenLevels(fr, geo) : null;
    for (let t = 0; t < S.U * N; t++) {
      const o = t * 4;
      if (!geo.valid[t]) { out[o + 3] = 0; continue; }
      let c;
      switch (mode) {
        case 'gt': c = fr.gt[t] < 10 ? pal[fr.gt[t]] : UNK; break;
        case 'dec': c = fr.dec[t] < 10 ? pal[fr.dec[t]] : UNK; break;
        case 'fc': c = fr.fc && fr.fc[t] < 10 ? pal[fr.fc[t]] : UNK; break;
        case 'fu': c = fr.fu[t] < 10 ? pal[fr.fu[t]] : UNK; break;
        case 'err': {
          const g = fr.gt[t], p = fr.fu[t];
          if (g >= 10 || p >= 10) c = UNK;
          else if (g === p) { const b = pal[g]; out[o] = b[0] * 0.4 + 140; out[o + 1] = b[1] * 0.4 + 146; out[o + 2] = b[2] * 0.4 + 154; out[o + 3] = 205; continue; }
          else c = ERR;
          break;
        }
        case 'lv': c = !lv || lv[t] === 255 ? UNK : cm('purples', 0.08 + 0.92 * (lv[t] & 7) / 7); break;
        case 'k': c = !fr.k ? UNK : fr.k[t] ? cm('viridis', 0.05 + 0.95 * Math.min(fr.k[t], 16) / 16) : [208, 216, 228]; break;
        case 'rx': c = !fr.fl ? UNK : (fr.fl[t] & 1) ? RX : NORX; break;
        case 'nl': c = !fr.nl || fr.nl[t] === 255 ? UNK : cm('ylorrd', clamp(fr.nl[t] / 32, 0, 1)); break;
        default: c = UNK;
      }
      out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = 215;
    }
    return out;
  }
  // Per-pixel colour pass on the GPU: each pixel takes one of its 2x2 neighbouring tokens by depth, position and colour,
  // then the colour of the view mode, mixed with the photo.
  const CP_VS = `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;
  const CP_FS = `uniform sampler2D uDep, uLab, uImg, uTokD, uTokC; uniform vec3 uPal[10]; uniform int uMode, uS, uLS; uniform float uRel, uAb, uKs, uKb, uTint;
    vec3 img(ivec2 p) { return texelFetch(uImg, p, 0).rgb * 255.0; }
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy); float d = texelFetch(uDep, p, 0).r;
      if (d <= 0.0) { gl_FragColor = vec4(0.0); return; }
      vec3 px = img(p); vec2 pf = vec2(p * uS);
      int i0 = clamp(int(floor((pf.y - 2.0) * 0.25)), 0, ${GH - 1}), i1 = min(i0 + 1, ${GH - 1});
      int j0 = clamp(int(floor((pf.x - 2.0) * 0.25)), 0, ${GW - 1}), j1 = min(j0 + 1, ${GW - 1});
      float tol = d * uRel + uAb, bc = 1e9; ivec2 best = ivec2(-1);
      for (int k = 0; k < 4; k++) {
        ivec2 t = ivec2((k & 1) == 1 ? j1 : j0, (k & 2) == 2 ? i1 : i0);
        float dt = texelFetch(uTokD, t, 0).r;
        if (dt <= 0.0) continue;
        vec2 s = (pf - 4.0 * vec2(t) - 2.0) * 0.25; vec3 q = px - img((4 * t + 2) / uS);
        float dd = (dt - d) / tol, c = dd * dd + 0.35 * dot(s, s) + dot(q, q) / 7000.0;
        if (c < bc) { bc = c; best = t; }
      }
      if (best.x < 0) { gl_FragColor = vec4(0.0); return; }
      vec3 col = px;
      if (uMode == 2) { int L = int(texelFetch(uLab, ivec2(pf) / uLS, 0).r * 255.0 + 0.5); if (L >= 10) { gl_FragColor = vec4(0.0); return; } col = uPal[L]; }
      else if (uMode == 0) { vec4 c = texelFetch(uTokC, best, 0); if (c.a <= 0.0) { gl_FragColor = vec4(0.0); return; } col = c.rgb * 255.0; }
      if (uTint > 0.5) { col *= 1.0 - uKs + uKs * (0.62 + 0.76 * dot(px, vec3(0.299, 0.587, 0.114)) / 255.0); col += (px - col) * uKb; }
      gl_FragColor = vec4(min(col, vec3(255.0)) / 255.0, 1.0);
    }`;
  const HP_FS = `uniform sampler2D uSrc; uniform int uOff, uAvg;
    void main() { ivec2 q = ivec2(gl_FragCoord.xy) - ivec2(0, uOff), p = q * 2; vec4 s = vec4(0.0);
      if (uAvg == 0) { gl_FragColor = texelFetch(uSrc, q, 0); return; }
      for (int k = 0; k < 4; k++) { vec4 c = texelFetch(uSrc, p + ivec2(k & 1, k >> 1), 0); s += vec4(c.rgb * c.a, c.a); }
      gl_FragColor = s.a > 0.0 ? vec4(s.rgb / s.a, s.a * 0.25) : vec4(0.0); }`;

  // ---------------------------------------------------------------- 3D view
  const SU_VS = `attribute vec2 pix;
    uniform sampler2D uDep, uCol, uLab; uniform vec4 uK; uniform mat3 uR; uniform vec3 uT; uniform vec2 uWH, uTexWH, uLabWH; uniform float uStep, uMaxD, uS, uLS;
    varying vec3 vCol; varying float vZ, vY; varying vec2 vXZ;
    vec2 tc(vec2 p) { return (floor(p / uS) + 0.5) / uTexWH; }
    float dep(vec2 p) { return texture2D(uDep, tc(p)).r * 0.01; }
    vec3 P(vec2 p, float d) { vec2 q = (p - uK.zw) / uK.xy; return uT + uR * vec3(d, q.x * d, q.y * d); }
    vec4 nb(vec2 p, vec3 c, float s, float lim) {
      if (p.x < 0.0 || p.y < 0.0 || p.x >= uWH.x || p.y >= uWH.y) return vec4(0.0);
      float d = dep(p); if (d <= 0.0) return vec4(0.0);
      vec3 v = (P(p, d) - c) * s; float l = length(v);
      if (l < 1e-6 || l > lim) return vec4(0.0);
      return vec4(v, l); }
    vec4 near(vec4 a, vec4 b) { if (a.w <= 0.0) return b; if (b.w <= 0.0) return a; return a.w < b.w ? a : b; }
    void main() {
      vec4 col = texture2D(uCol, tc(pix)); float d = dep(pix), lb = texture2D(uLab, (floor(pix / uLS) + 0.5) / uLabWH).r * 255.0;
      vCol = col.rgb; vZ = 0.0; vY = 0.0; vXZ = vec2(0.0);
      if (d <= 0.0 || col.a < 0.02 || (lb < 1.5 && d > uMaxD)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
      vec3 c = P(pix, d), ray = c - uT, rt = uR[1], dn = uR[2];
      float sx = d * uStep / uK.x, sy = d * uStep / uK.y, lim = max(0.3, 12.0 * uStep * d / min(uK.x, uK.y));
      vec3 N = uR[0]; bool fit = false;
      vec4 dx = near(nb(pix - vec2(uStep, 0.0), c, -1.0, lim), nb(pix + vec2(uStep, 0.0), c, 1.0, lim));
      vec4 dy = near(nb(pix - vec2(0.0, uStep), c, -1.0, lim), nb(pix + vec2(0.0, uStep), c, 1.0, lim));
      if (dx.w > 0.0 && dy.w > 0.0) {
        vec3 n = cross(dx.xyz, dy.xyz); float l = length(n);
        if (l >= 0.1 * dx.w * dy.w) {
          n /= l; float num = dot(n, ray); bool gr = abs(n.y) > 0.92 || abs(n.y) < 0.25;
          if (abs(num) / length(ray) >= (gr ? 0.015 : 0.05)) {
            vec4 den = vec4(dot(n, ray - 0.5 * rt * sx - 0.5 * dn * sy), dot(n, ray - 0.5 * rt * sx + 0.5 * dn * sy),
                            dot(n, ray + 0.5 * rt * sx - 0.5 * dn * sy), dot(n, ray + 0.5 * rt * sx + 0.5 * dn * sy));
            vec4 sc = vec4(num) / den; float lo = gr ? 0.12 : 0.55, hi = gr ? 8.0 : 1.8;
            if (all(greaterThanEqual(sc, vec4(lo))) && all(lessThanEqual(sc, vec4(hi)))) { N = n; fit = true; }
          }
        }
      }
      float inf = 1.0;
      if (!fit && lb < 1.5) {
        vec3 up = vec3(0.0, 1.0, 0.0); float num = dot(up, ray);
        vec4 den = vec4(dot(up, ray - 0.5 * rt * sx - 0.5 * dn * sy), dot(up, ray - 0.5 * rt * sx + 0.5 * dn * sy),
                        dot(up, ray + 0.5 * rt * sx - 0.5 * dn * sy), dot(up, ray + 0.5 * rt * sx + 0.5 * dn * sy));
        vec4 sc = vec4(num) / den;
        if (abs(num) / length(ray) >= 0.01 && all(greaterThanEqual(sc, vec4(0.2))) && all(lessThanEqual(sc, vec4(5.0)))) { N = up; inf = 1.35; }
      }
      vec3 cr = ray + inf * (position.x * rt * sx + position.y * dn * sy);
      vec3 w = uT + cr * (dot(N, ray) / dot(N, cr));
      vY = w.y; vXZ = w.xz; vec4 mv = modelViewMatrix * vec4(w, 1.0); vZ = -mv.z; gl_Position = projectionMatrix * mv; }`;
  const SU_FS = `uniform vec3 uFog; uniform float uNear, uFar, uClip, uRad; uniform vec2 uCtr; varying vec3 vCol; varying float vZ, vY; varying vec2 vXZ;
    void main() { if (vY > uClip || length(vXZ - uCtr) > uRad) discard; gl_FragColor = vec4(mix(vCol, uFog, smoothstep(uNear, uFar, vZ) * 0.3), 1.0); }`;
  const PT_VS = `attribute vec3 col; uniform float uSize, uScale; varying vec3 vCol;
    void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
      gl_PointSize = clamp(uSize * uScale / max(-mv.z, 0.1), 1.6, 30.0); vCol = col; }`;
  const PT_FS = `varying vec3 vCol; void main(){ vec2 c = gl_PointCoord * 2.0 - 1.0; if (dot(c, c) > 1.0) discard; gl_FragColor = vec4(vCol, 1.0); }`;
  const LN_VS = `attribute float lineDistance; varying float vD; void main(){ vD = lineDistance; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
  const LN_FS = `uniform vec3 uCol; uniform float uT, uA; varying float vD;
    void main(){ float p = mod(vD - uT, 6.0); if (p > 3.4) discard; gl_FragColor = vec4(uCol, uA * (0.55 + 0.45 * smoothstep(0.0, 3.4, p))); }`;
  const LIM = { clip: { value: 1e6 }, rad: { value: 1e6 }, ctr: { value: new THREE.Vector2() }, maxd: { value: 1e6 } };

  function textSprite(txt, hex) {
    const cv = document.createElement('canvas'), s = 2, ctx = cv.getContext('2d'), font = `600 ${13 * s}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(txt).width) + 26 * s, hh = 24 * s;
    cv.width = w; cv.height = hh; ctx.font = font;
    ctx.fillStyle = 'rgba(255,255,255,.95)'; ctx.strokeStyle = hex; ctx.lineWidth = 1.5 * s;
    ctx.beginPath(); ctx.roundRect(1, 1, w - 2, hh - 2, 8 * s); ctx.fill(); ctx.stroke();
    ctx.fillStyle = hex; ctx.beginPath(); ctx.arc(12 * s, hh / 2, 3.5 * s, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#0f172a'; ctx.textBaseline = 'middle'; ctx.fillText(txt, 20 * s, hh / 2 + 1);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, sizeAttenuation: false, transparent: true }));
    sp.scale.set(w / hh * 0.032, 0.032, 1); sp.renderOrder = 10; sp.center.set(0.5, 0);
    return sp;
  }
  function radialTex(stops) {
    const cv = document.createElement('canvas'); cv.width = cv.height = 128;
    const ctx = cv.getContext('2d'), g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    for (const [o, c] of stops) g.addColorStop(o, c);
    ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 128);
    const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
  }

  class Viewer {
    constructor(el) {
      this.el = el;
      const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      r.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5)); r.setClearColor(0, 0); el.append(r.domElement);
      this.scene = new THREE.Scene(); this.bg = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(48, 1, 0.3, 5000);
      this.orbit = { theta: -2.25, phi: 0.8, r: 140, target: new THREE.Vector3(), goal: null };
      this.vel = { t: 0, p: 0 }; this.dirty = true;
      this.disk = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({
        map: radialTex([[0, 'rgba(203,212,224,1)'], [0.55, 'rgba(214,222,232,0.85)'], [1, 'rgba(236,241,247,0)']]), transparent: true, depthWrite: false, depthTest: false }));
      this.disk.rotation.x = -Math.PI / 2; this.bg.add(this.disk);
      const fog = () => ({ uFog: { value: new THREE.Color(BG[0] / 255, BG[1] / 255, BG[2] / 255) }, uNear: { value: 150 }, uFar: { value: 700 } });
      this.fogU = fog();
      const quad = new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, 0.5, 0.5, 0], 3);
      const grid = step => {
        const w = PW / step, n = w * (PH / step), pix = new Float32Array(n * 2);
        for (let p = 0; p < n; p++) { pix[2 * p] = (p % w) * step; pix[2 * p + 1] = ((p / w) | 0) * step; }
        const g = new THREE.InstancedBufferGeometry(); g.setAttribute('position', quad); g.setIndex([0, 1, 2, 2, 1, 3]);
        g.setAttribute('pix', new THREE.InstancedBufferAttribute(pix, 2)); g.instanceCount = n; g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
        return g;
      };
      this.grids = { full: grid(1), fast: grid(2) };
      const nearest = x => { x.magFilter = x.minFilter = THREE.NearestFilter; x.generateMipmaps = false; x.flipY = false; return x; };
      const dtex = (data, w, hh, fmt, type) => { const x = nearest(new THREE.DataTexture(data, w, hh, fmt, type)); x.needsUpdate = true; return x; };
      const target = (w, hh) => new THREE.WebGLRenderTarget(w, hh, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, generateMipmaps: false });
      const level = s => {   // per-node textures at full (s = 1) or half (s = 2) resolution
        const w = PW / s, hh = PH / s;
        return { s, w, hh, dep: dtex(new Float32Array(w * hh), w, hh, THREE.RedFormat, THREE.FloatType), lab: dtex(new Uint8Array(w * hh), w, hh, THREE.RedFormat, THREE.UnsignedByteType),
          img: nearest(new THREE.Texture()), rt: target(w, hh) };
      };
      this.su = range(UMAX).map(u => {
        const H = level(1), L = level(2);
        const tokD = dtex(new Float32Array(N), GW, GH, THREE.RedFormat, THREE.FloatType), tokC = dtex(new Uint8Array(N * 4), GW, GH, THREE.RGBAFormat, THREE.UnsignedByteType);
        const tokL = dtex(new Uint8Array(N), GW, GH, THREE.RedFormat, THREE.UnsignedByteType);
        const uni = { uDep: { value: H.dep }, uLab: { value: H.lab }, uCol: { value: H.rt.texture }, uK: { value: new THREE.Vector4() }, uR: { value: new THREE.Matrix3() },
          uT: { value: new THREE.Vector3() }, uWH: { value: new THREE.Vector2(PW, PH) }, uTexWH: { value: new THREE.Vector2(PW, PH) }, uS: { value: 1 }, uLS: { value: 1 }, uLabWH: { value: new THREE.Vector2(PW, PH) }, uStep: { value: 1 },
          uMaxD: LIM.maxd, uClip: LIM.clip, uRad: LIM.rad, uCtr: LIM.ctr, ...this.fogU };
        const mesh = new THREE.Mesh(this.grids.full, new THREE.ShaderMaterial({ vertexShader: SU_VS, fragmentShader: SU_FS, uniforms: uni, side: THREE.DoubleSide,
          polygonOffset: true, polygonOffsetFactor: 0.5 * u, polygonOffsetUnits: 4 * u }));
        mesh.frustumCulled = false; mesh.visible = false; this.scene.add(mesh);
        return { u, H, L, cur: H, tokD, tokC, tokL, lab: H.lab, ls: 1, uni, mesh, live: false };
      });
      const tri = new THREE.BufferGeometry(); tri.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
      const pass = (fs, u) => {
        const m = new THREE.Mesh(tri, new THREE.ShaderMaterial({ vertexShader: CP_VS, fragmentShader: fs, uniforms: u, depthTest: false, depthWrite: false }));
        m.frustumCulled = false; const s = new THREE.Scene(); s.add(m); return { s, m: m.material, u };
      };
      this.cp = pass(CP_FS, { uDep: { value: null }, uLab: { value: null }, uImg: { value: null }, uTokD: { value: null }, uTokC: { value: null },
        uPal: { value: range(10).map(() => new THREE.Vector3()) }, uMode: { value: 0 }, uS: { value: 1 }, uLS: { value: 1 }, uRel: { value: 0 }, uAb: { value: 0 }, uKs: { value: 1 }, uKb: { value: 0 }, uTint: { value: 0 } });
      this.hp = pass(HP_FS, { uSrc: { value: null }, uOff: { value: 0 }, uAvg: { value: 1 } });
      this.passCam = new THREE.Camera(); this.half = target(PW / 2, PH / 2 * UMAX); this.halfBuf = new Uint8Array(PW / 2 * PH / 2 * UMAX * 4);
      this.onscreen = true;
      this.glyphs = new THREE.Group(); this.scene.add(this.glyphs);
      this.glow = radialTex([[0, 'rgba(255,255,255,1)'], [0.25, 'rgba(255,255,255,.55)'], [1, 'rgba(255,255,255,0)']]);
      this.centre = new THREE.Group();
      const core = new THREE.Mesh(new THREE.OctahedronGeometry(2.2, 0), new THREE.MeshBasicMaterial({ color: new THREE.Color('#0d9488'), wireframe: true }));
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, color: new THREE.Color('#14b8a6'), transparent: true, opacity: 0.16, depthWrite: false }));
      halo.scale.set(10, 10, 1);
      const lbl = textSprite('Fusion centre', '#0d9488'); lbl.position.y = 3.5;
      this.centre.add(core, halo, lbl); this.scene.add(this.centre);
      this.mast = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.3, 1, 10), new THREE.MeshBasicMaterial({ color: new THREE.Color('#94a3b8') }));
      this.scene.add(this.mast);
      this.links = range(UMAX).map(u => {
        const ln = new THREE.Line(new THREE.BufferGeometry(), new THREE.ShaderMaterial({ vertexShader: LN_VS, fragmentShader: LN_FS, transparent: true, depthWrite: false,
          uniforms: { uCol: { value: new THREE.Color(NODE_HEX[u]) }, uT: { value: 0 }, uA: { value: 0.6 } } }));
        ln.frustumCulled = false; this.scene.add(ln); return ln;
      });
      this.NP = 14; this.curves = [];
      this.pkPos = new THREE.BufferAttribute(new Float32Array(UMAX * this.NP * 3), 3); this.pkCol = new THREE.BufferAttribute(new Uint8Array(UMAX * this.NP * 3), 3, true);
      const pg = new THREE.BufferGeometry(); pg.setAttribute('position', this.pkPos); pg.setAttribute('col', this.pkCol);
      this.pkU = { uSize: { value: 0.7 }, uScale: { value: 500 } };
      this.packets = new THREE.Points(pg, new THREE.ShaderMaterial({ vertexShader: PT_VS, fragmentShader: PT_FS, uniforms: this.pkU, depthTest: false }));
      this.packets.frustumCulled = false; this.packets.renderOrder = 5; this.scene.add(this.packets);
      this.cells = new THREE.InstancedMesh(new THREE.BoxGeometry(1.84, 1.84, 1.84), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.2, depthWrite: false }), 60000);
      this.cells.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(180000), 3); this.cells.count = 0; this.cells.frustumCulled = false; this.scene.add(this.cells);
      this.controls();
      new ResizeObserver(() => this.resize()).observe(el);
      this.resize();
    }
    resize() {
      const w = this.el.clientWidth || 1, hh = this.el.clientHeight || 1;
      this.renderer.setSize(w, hh, false);
      this.pkU.uScale.value = hh * this.renderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
      this.dirty = true; kick();
    }
    controls() {
      const c = this.renderer.domElement, ptr = new Map(); let drag = null, pinch = 0;
      c.addEventListener('contextmenu', e => e.preventDefault());
      c.addEventListener('pointerdown', e => { ptr.set(e.pointerId, [e.clientX, e.clientY]); drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; c.setPointerCapture(e.pointerId); this.vel.t = this.vel.p = 0; this.held = true; E.hint.style.opacity = 0; });
      c.addEventListener('pointermove', e => {
        if (!ptr.has(e.pointerId)) return; ptr.set(e.pointerId, [e.clientX, e.clientY]);
        const o = this.orbit;
        if (ptr.size === 2) { const p = [...ptr.values()], d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]); if (pinch) o.r = clamp(o.r * pinch / d, 6, 900); pinch = d; }
        else if (drag) {
          const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          if (drag.pan) { const s = o.r / 700, st = Math.sin(o.theta), ct = Math.cos(o.theta); o.target.x += (-dx * st - dy * ct) * s; o.target.z += (dx * ct - dy * st) * s; o.goal = null; S.follow = false; syncFollow(); }
          else { this.vel.t = -dx * 0.0055; this.vel.p = -dy * 0.0055; o.theta += this.vel.t; o.phi = clamp(o.phi + this.vel.p, 0.05, 1.52); }
          drag.x = e.clientX; drag.y = e.clientY;
        }
        this.dirty = true; kick();
      });
      const up = e => { ptr.delete(e.pointerId); drag = null; pinch = 0; this.held = ptr.size > 0; kick(); };
      c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
      c.addEventListener('wheel', e => { e.preventDefault(); this.orbit.r = clamp(this.orbit.r * Math.exp(e.deltaY * 0.0011), 6, 900); this.dirty = true; kick(); }, { passive: false });
    }
    lift(geo) {
      if (!air()) return 1;
      let top = -1e9; for (let u = 0; u < S.U; u++) top = Math.max(top, -(S.fr.cams[u * 16 + 15] - S.O[2]));
      return Math.max(4, (top - geo.ground) * 0.25);
    }
    home(geo) {
      const P = geo.pos, xs = [], zs = [], c = S.fr.cams;
      for (let t = 0; t < S.U * N; t += 7) if (geo.valid[t]) { xs.push(P[t * 3]); zs.push(P[t * 3 + 2]); }
      for (let u = 0; u < S.U; u++) { xs.push(c[u * 16 + 13] - S.O[0]); zs.push(c[u * 16 + 14] - S.O[1]); }
      xs.sort((a, b) => a - b); zs.sort((a, b) => a - b);
      const q = (a, f) => a[Math.min(a.length - 1, Math.floor(a.length * f))];
      const x0 = q(xs, 0.03), x1 = q(xs, 0.97), z0 = q(zs, 0.03), z1 = q(zs, 0.97), ext = Math.max(x1 - x0, z1 - z0, 20), lift = this.lift(geo), o = this.orbit;
      o.target.set((x0 + x1) / 2, geo.ground + lift, (z0 + z1) / 2);
      o.theta = air() ? -2.25 : -2.2; o.phi = air() ? 0.8 : 0.74;
      o.r = air() ? clamp(Math.max(ext * 0.66, lift * 4.6), 50, 900) : clamp(ext * 0.62, 26, 600);
      o.goal = null; this.vel.t = this.vel.p = 0; this.dirty = true;
    }
    follow(geo) { if (S.follow) this.orbit.goal = new THREE.Vector3(geo.centre[0], geo.ground + this.lift(geo), geo.centre[2]); }
    step(dt) {
      let moving = false;
      if (!this.held && Math.abs(this.vel.t) + Math.abs(this.vel.p) > 1e-4) {
        this.vel.t *= 0.9; this.vel.p *= 0.9; this.orbit.theta += this.vel.t * 0.6; this.orbit.phi = clamp(this.orbit.phi + this.vel.p * 0.6, 0.05, 1.52); moving = true;
      }
      if (this.orbit.goal) {
        const g = this.orbit.goal, t = this.orbit.target;
        if (t.distanceTo(g) > 0.02) { t.lerp(g, 1 - Math.pow(0.0025, dt)); moving = true; } else this.orbit.goal = null;
      }
      return moving;
    }
    render(t) {
      const w = this.el.clientWidth, hh = this.el.clientHeight, o = this.orbit, cam = this.camera, sp = Math.sin(o.phi);
      cam.position.set(o.target.x + o.r * sp * Math.cos(o.theta), o.target.y + o.r * Math.cos(o.phi), o.target.z + o.r * sp * Math.sin(o.theta));
      cam.lookAt(o.target); cam.near = clamp(o.r * 0.012, 0.3, 4); cam.far = Math.max(2000, o.r * 12); cam.aspect = w / hh; cam.updateProjectionMatrix();
      this.fogU.uNear.value = o.r * 1.6; this.fogU.uFar.value = o.r * 1.6 + Math.max(300, o.r * 3.2) * 1.5;
      for (const ln of this.links) ln.material.uniforms.uT.value = t * 0.012;
      if (this.curves.length && this.packets.visible) {
        const P = this.pkPos.array, v = new THREE.Vector3();
        this.curves.forEach((cv, u) => { for (let q = 0; q < this.NP; q++) { cv.getPoint(((t * 0.00028 + q / this.NP + u * 0.13) % 1 + 1) % 1, v); const k = (u * this.NP + q) * 3; P[k] = v.x; P[k + 1] = v.y; P[k + 2] = v.z; } });
        this.pkPos.needsUpdate = true;
      }
      const r = this.renderer;
      r.setViewport(0, 0, w, hh); r.autoClear = false; r.clear(); r.render(this.bg, cam); r.render(this.scene, cam); r.autoClear = true;
      this.dirty = false;
    }
    setSurfels(fr) {
      const [O0, O1, O2] = S.O;
      for (const o of this.su) o.live = false;
      for (let u = 0; u < S.U; u++) {
        const o = this.su[u], c = fr.cams.subarray(u * 16, u * 16 + 16), td = o.tokD.image.data, L = fr.hi ? o.H : o.L, P = L.w * L.hh;
        L.dep.image.data = fr.deps[u]; L.dep.needsUpdate = true;
        if (fr.lab) { L.lab.image.data = fr.lab.subarray(u * P, (u + 1) * P); L.lab.needsUpdate = true; o.lab = L.lab; o.ls = L.s; }
        else { o.tokL.image.data.set(fr.gt.subarray(u * N, (u + 1) * N)); o.tokL.needsUpdate = true; o.lab = o.tokL; o.ls = 4; }   // token labels
        L.img.image = fr.bmps[u]; L.img.needsUpdate = true;
        for (let j = 0; j < N; j++) td[j] = fr.depth[u * N + j];
        o.tokD.needsUpdate = true;
        o.cur = L; o.mesh.geometry = this.grids[L.s === 1 ? 'full' : 'fast'];
        o.uni.uDep.value = L.dep; o.uni.uLab.value = o.lab; o.uni.uLS.value = o.ls; o.uni.uLabWH.value.set(PW / o.ls, PH / o.ls); o.uni.uCol.value = L.rt.texture; o.uni.uS.value = L.s; o.uni.uStep.value = L.s; o.uni.uTexWH.value.set(L.w, L.hh);
        o.uni.uK.value.set(c[0], c[1], c[2], c[3]);
        o.uni.uR.value.set(c[4], c[5], c[6], -c[10], -c[11], -c[12], c[7], c[8], c[9]);
        o.uni.uT.value.set(c[13] - O0, -(c[15] - O2), c[14] - O1);
        o.live = true;
      }
    }
    paint(tc) {
      const U = this.cp.u, s = S.mix, photo = S.mode === 'rgb', r = this.renderer;
      U.uMode.value = photo ? 1 : S.mode === 'gt' ? 2 : 0;
      U.uRel.value = air() ? 0.012 : 0.03; U.uAb.value = air() ? 20 : 8;
      U.uTint.value = !photo && s < 0.999 ? 1 : 0; U.uKs.value = s >= 0.5 ? 2 * (1 - s) : 1; U.uKb.value = s >= 0.5 ? 0 : 1 - 2 * s;
      PAL[S.ds.id].colors.forEach((c, i) => U.uPal.value[i].set(c[0], c[1], c[2]));
      for (const o of this.su) {
        if (!o.live) continue;
        if (tc) { o.tokC.image.data.set(tc.subarray(o.u * N * 4, (o.u + 1) * N * 4)); o.tokC.needsUpdate = true; }
        const L = o.cur;
        U.uDep.value = L.dep; U.uLab.value = o.lab; U.uLS.value = o.ls; U.uImg.value = L.img; U.uTokD.value = o.tokD; U.uTokC.value = o.tokC; U.uS.value = L.s;
        this.cp.m.uniformsNeedUpdate = true;
        r.setRenderTarget(L.rt); r.render(this.cp.s, this.passCam);
      }
      r.setRenderTarget(null); this.dirty = true;
    }
    cards() {   // node colours at 320x192 for the side cards, all nodes in one target and one read-back
      const r = this.renderer, ac = r.autoClear, h2 = PH / 2;
      r.autoClear = false;
      for (let u = 0; u < S.U; u++) {
        const L = this.su[u].cur;
        this.hp.u.uSrc.value = L.rt.texture; this.hp.u.uOff.value = u * h2; this.hp.u.uAvg.value = L.s === 1 ? 1 : 0; this.hp.m.uniformsNeedUpdate = true;
        this.half.viewport.set(0, u * h2, PW / 2, h2); r.setRenderTarget(this.half); r.render(this.hp.s, this.passCam);
      }
      r.readRenderTargetPixels(this.half, 0, 0, PW / 2, h2 * S.U, this.halfBuf);
      r.setRenderTarget(null); r.autoClear = ac;
      return this.halfBuf;
    }
    pixelRatio() { this.renderer.setPixelRatio(S.res === 'fast' ? 1 : Math.min(devicePixelRatio || 1, 1.5)); this.resize(); }
    visibility() {
      for (const o of this.su) o.mesh.visible = o.live && o.u < S.U && (S.node < 0 || S.node === o.u);
      this.cells.visible = S.layers.cells;
      for (const g of this.glyphs.children) g.userData.fr.visible = S.layers.frusta && (S.node < 0 || S.node === g.userData.u);
      this.centre.visible = this.packets.visible = S.layers.links;
      this.links.forEach((l, i) => { l.visible = S.layers.links && i < S.U; });
      this.mast.visible = S.layers.links && air();
      this.dirty = true;
    }
    nodes(fr, geo) {
      if (this.glyphs.userData.ds !== S.ds.id) this.buildGlyphs();
      const [O0, O1, O2] = S.O, ps = [];
      for (const g of this.glyphs.children) {
        const u = g.userData.u, c = fr.cams.subarray(u * 16, u * 16 + 16), p = new THREE.Vector3(c[13] - O0, -(c[15] - O2), c[14] - O1);
        const fwd = new THREE.Vector3(c[4], -c[10], c[7]).normalize();
        g.userData.body.position.copy(p); g.userData.body.rotation.y = Math.atan2(fwd.x, fwd.z);
        if (!air()) {
          if (u === 0) { const hgt = Math.max(1, p.y - geo.ground); g.userData.pole.position.set(p.x, geo.ground + hgt / 2, p.z); g.userData.pole.scale.set(1, hgt, 1); }
          else g.userData.body.position.y = p.y - 1.25;
        }
        g.userData.label.position.set(p.x, p.y + (air() ? 6.5 : 2.6), p.z); g.userData.halo.position.copy(p);
        const D = air() ? clamp(geo.dmed[u] * 0.97, 10, 300) : clamp(geo.dmed[u] * 0.55, 6, 30);
        const corner = (x, y) => { const b = (x - c[2]) / c[0] * D, e = (y - c[3]) / c[1] * D; return new THREE.Vector3(c[4] * D + c[5] * b + c[6] * e + c[13] - O0, -(c[10] * D + c[11] * b + c[12] * e + c[15] - O2), c[7] * D + c[8] * b + c[9] * e + c[14] - O1); };
        const k = [corner(0, 0), corner(639, 0), corner(639, 383), corner(0, 383)], A = g.userData.fr.geometry.attributes.position.array;
        let q = 0; const put = v => { A[q++] = v.x; A[q++] = v.y; A[q++] = v.z; };
        for (let i = 0; i < 4; i++) { put(p); put(k[i]); put(k[i]); put(k[(i + 1) % 4]); }
        g.userData.fr.geometry.attributes.position.needsUpdate = true;
        ps.push(p);
      }
      const cx = ps.reduce((a, p) => a + p.x, 0) / ps.length, cz = ps.reduce((a, p) => a + p.z, 0) / ps.length, hmax = Math.max(...ps.map(p => p.y)), alt = hmax - geo.ground;
      const goal = air() ? new THREE.Vector3(cx + alt * 0.55, geo.ground + alt * 0.45, cz - alt * 0.35) : new THREE.Vector3(cx, hmax + 16, cz);
      if (air()) { const mh = Math.max(1, goal.y - geo.ground); this.mast.position.set(goal.x, geo.ground + mh / 2, goal.z); this.mast.scale.set(1, mh, 1); }
      this.centre.position.lerp(goal, this.centre.userData.init ? 0.35 : 1); this.centre.userData.init = true;
      this.packets.geometry.setDrawRange(0, ps.length * this.NP); this.pkU.uSize.value = air() ? 0.7 : 0.35; this.curves = [];
      ps.forEach((a, u) => {
        const b = this.centre.position, m = a.clone().add(b).multiplyScalar(0.5); m.y += air() ? 8 : 6;
        const curve = new THREE.QuadraticBezierCurve3(a.clone(), m, b.clone()); this.curves.push(curve);
        const nd = fr.meta.nodes[u], tot = Math.max(1, nd.nd + nd.ga.length), nDig = Math.round(this.NP * nd.nd / tot), C = this.pkCol.array;
        for (let q = 0; q < this.NP; q++) { const col = q < nDig ? [245, 158, 11] : [2, 132, 199], o = (u * this.NP + q) * 3; C[o] = col[0]; C[o + 1] = col[1]; C[o + 2] = col[2]; }
        this.pkCol.needsUpdate = true;
        const ln = this.links[u]; ln.geometry.dispose(); ln.geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)); ln.computeLineDistances();
        ln.material.uniforms.uA.value = 0.25 + 0.45 * (nd.sent ? nd.recv / nd.sent : 0);
      });
      this.dirty = true;
    }
    buildGlyphs() {
      for (const g of [...this.glyphs.children]) this.glyphs.remove(g);
      this.glyphs.userData.ds = S.ds.id;
      for (let u = 0; u < S.U; u++) {
        const col = new THREE.Color(NODE_HEX[u]), mat = new THREE.MeshBasicMaterial({ color: col }), g = new THREE.Group(), body = new THREE.Group();
        g.userData.u = u;
        if (air()) {
          body.add(new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.45, 1.3), mat));
          for (const a of [Math.PI / 4, -Math.PI / 4]) { const arm = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.16, 0.22), mat); arm.rotation.y = a; body.add(arm); }
          for (const [x, z] of [[1.48, 1.48], [-1.48, 1.48], [1.48, -1.48], [-1.48, -1.48]]) { const ring = new THREE.Mesh(new THREE.TorusGeometry(0.72, 0.07, 6, 24), mat); ring.rotation.x = Math.PI / 2; ring.position.set(x, 0.18, z); body.add(ring); }
          body.scale.setScalar(1.7);
        } else if (u === 0) {
          body.add(new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.55, 0.6), mat));
          const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.16, 1, 10), new THREE.MeshBasicMaterial({ color: new THREE.Color('#64748b') }));
          g.add(pole); g.userData.pole = pole;
        } else {
          const car = new THREE.Mesh(new THREE.BoxGeometry(1.9, 1.15, 4.5), mat); car.position.set(0, 0, -1.2);
          const cab = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.7, 2.2), new THREE.MeshBasicMaterial({ color: new THREE.Color('#334155') })); cab.position.set(0, 0.9, -1.4);
          body.add(car, cab);
        }
        body.rotation.order = 'YXZ';
        const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, color: col, transparent: true, opacity: 0.28, depthWrite: false }));
        halo.scale.set(air() ? 15 : 6, air() ? 15 : 6, 1);
        const label = textSprite(S.ds.nodes[u], NODE_HEX[u]);
        const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(48), 3));
        const fr = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: air() ? 0.32 : 0.55, depthWrite: false }));
        fr.frustumCulled = false;
        g.add(body, halo, label, fr); Object.assign(g.userData, { body, halo, label, fr });
        this.glyphs.add(g);
      }
    }
    setCells(fr) {
      if (!fr.cells) { this.cells.count = 0; return; }
      const m = this.cells, c = fr.cells, v = fr.cellv, mat = new THREE.Matrix4(), col = new THREE.Color(), [O0, O1, O2] = S.O, c0 = fr.c0;
      let n = 0;
      for (let q = 0; q < v.length && n < 60000; q++) {
        const l = v[q] & 7; if (l < 2) continue;
        mat.makeTranslation((c[q * 3] + c0[0] + 0.5) * 2 - O0, -((c[q * 3 + 2] + c0[2] + 0.5) * 2 - O2), (c[q * 3 + 1] + c0[1] + 0.5) * 2 - O1); m.setMatrixAt(n, mat);
        const k = cm('purples', 0.25 + 0.75 * l / 7); col.setRGB(k[0] / 255, k[1] / 255, k[2] / 255, THREE.SRGBColorSpace); m.setColorAt(n, col);
        n++;
      }
      m.count = n; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true;
      this.dirty = true;
    }
    ground(geo) { const s = air() ? 520 : 170; this.disk.position.set(geo.centre[0], geo.ground - 0.25, geo.centre[2]); this.disk.scale.set(s, s, 1); }
  }

  // ---------------------------------------------------------------- page
  const S = { D: null, ds: null, clip: null, U: 3, snr: 0, bw: 1, mode: 'fu', node: -1, i: 0, want: 0, fr: null, geo: null, O: null, Ov: null,
    playing: false, wait: false, speed: 1, pace: 1, token: 0, mix: 0.85, follow: true, res: 'auto', streamHi: false, userQ: false, busyMsg: '', waitPct: 0,
    layers: { cells: false, frusta: false, links: true }, rgba: new Uint8Array(UMAX * N * 4) };
  const E = {};
  let V = null;
  const ICON = {
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>',
    pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4.5" width="4.2" height="15" rx="1"/><rect x="13.8" y="4.5" width="4.2" height="15" rx="1"/></svg>',
    prev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M15 6l-6 6 6 6"/></svg>',
    next: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 6l6 6-6 6"/></svg>',
    top: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="12" cy="12" r="2.2"/></svg>',
    home: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M4 11l8-6 8 6"/><path d="M6 10v9h12v-9"/></svg>',
    follow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>',
    full: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
    layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
  };
  function seg(items, val, on, cls = '') {
    const s = h('div', { class: 'seg ' + cls });
    for (const it of items) {
      const b = h('button', { 'data-v': String(it.v), title: it.d || null }, it.t);
      if (String(it.v) === String(val)) b.classList.add('on');
      b.addEventListener('click', () => { for (const x of s.children) x.classList.toggle('on', x === b); on(it.v); });
      s.append(b);
    }
    s.set = v => { for (const x of s.children) x.classList.toggle('on', x.dataset.v === String(v)); };
    return s;
  }
  const field = (label, el) => h('div', { class: 'field' }, h('label', null, label), el);
  const vbtn = (icon, title, fn) => { const b = h('button', { class: 'vbtn', title, html: icon }); b.addEventListener('click', fn); return b; };
  function toggle(label, get, set) { const i = h('input', { type: 'checkbox' }); i.checked = get(); i.addEventListener('change', () => set(i.checked)); return h('label', { class: 'chk' }, i, label); }
  function syncFollow() { E.followBtn.classList.toggle('on', S.follow); }

  function build(root) {
    E.dsSeg = seg(S.D.datasets.map(d => ({ v: d.id, t: d.title })), S.D.datasets[0].id, v => selectDataset(v), 'ds');
    E.clipSel = h('select', { class: 'sel', 'aria-label': 'Scene' }); E.clipSel.addEventListener('change', () => selectClip(E.clipSel.value));
    E.snrSeg = seg(S.D.snrs.map(v => ({ v, t: v + ' dB' })), S.snr, v => { S.snr = v; goTo(S.i); });
    E.bwSeg = seg(S.D.bws.map(v => ({ v, t: BW_LABEL[v], d: v === 1 ? 'One complex channel use per token' : `${BW_LABEL[v]} channel use per token` })), S.bw, v => { S.bw = v; goTo(S.i); });
    const bar = h('div', { class: 'bar' }, E.dsSeg, field('Scene', E.clipSel), field('SNR', E.snrSeg), field('Bandwidth', E.bwSeg));
    const kpi = (id, label, grad) => { const v = h('div', { class: 'kv' }, '—'), d = h('div', { class: 'kd' }), b = h('i', { style: { background: grad } }); E['k_' + id] = { v, d, b }; return h('div', { class: 'kpi' }, h('div', { class: 'kl' }, label), v, h('div', { class: 'kb' }, b), d); };
    const kpis = h('div', { class: 'kpis' }, kpi('fu', 'Fused mIoU · this frame', 'linear-gradient(90deg,#14b8a6,#5eead4)'),
      kpi('fc', 'World-model prediction · before reception', 'linear-gradient(90deg,#8b5cf6,#c4b5fd)'),
      kpi('nl', 'JEPA prediction error (nat/dim)', 'linear-gradient(90deg,#7c3aed,#a78bfa)'), kpi('rx', 'Decisions received', 'linear-gradient(90deg,#f59e0b,#fcd34d)'));
    E.vp = h('div', { class: 'vp' });
    E.modeSegs = [0, 1].map(g => seg(MODES.filter(m => m.g === g).map(m => ({ v: m.id, t: m.t, d: m.d })), S.mode, v => selectMode(v)));
    const modebar = h('div', { class: 'glass modebar' }, h('div', { class: 'grp' }, h('span', { class: 'gl' }, 'SEMANTICS'), E.modeSegs[0]), h('div', { class: 'grp' }, h('span', { class: 'gl' }, 'TRANSMISSION'), E.modeSegs[1]));
    E.nodeSeg = seg([{ v: -1, t: 'All' }, ...range(UMAX).map(u => ({ v: u, t: String(u + 1) }))], -1, v => { S.node = +v; V.visibility(); paintCams(); kick(); }, 'small');
    E.mix = h('input', { type: 'range', class: 'rng', min: 0, max: 100, value: Math.round(S.mix * 100), title: '0: camera image, 100: class colour' });
    E.mix.addEventListener('input', () => { S.mix = E.mix.value / 100; recolor(); });
    E.qSeg = seg([{ v: 'auto', t: 'Auto', d: 'Full resolution when paused; during playback as the network allows' }, { v: 'full', t: 'Full' }, { v: 'fast', t: 'Fast' }],
      S.res, v => setRes(v, true), 'small');
    E.frChk = toggle('Camera frusta', () => S.layers.frusta, v => { S.layers.frusta = v; V.visibility(); kick(); });
    E.layers = h('div', { class: 'glass layers', hidden: true },
      h('div', { class: 'row' }, h('span', null, 'Nodes'), E.nodeSeg), h('div', { class: 'row' }, h('span', null, 'Class colour'), E.mix),
      h('div', { class: 'row' }, h('span', null, 'Resolution'), E.qSeg),
      toggle('Uncertainty cells', () => S.layers.cells, v => { S.layers.cells = v; if (v && S.fr && !S.fr.cells) goTo(S.i); else { if (v && S.fr) V.setCells(S.fr); V.visibility(); kick(); } }), E.frChk,
      toggle('Uplinks', () => S.layers.links, v => { S.layers.links = v; V.visibility(); kick(); }));
    const layBtn = h('button', { class: 'glass laybtn', html: ICON.layers + '<span>Layers</span>' });
    layBtn.addEventListener('click', () => { E.layers.hidden = !E.layers.hidden; });
    E.legend = h('div', { class: 'glass legend' });
    E.followBtn = vbtn(ICON.follow, 'Follow the nodes', () => { S.follow = !S.follow; syncFollow(); if (S.follow && S.geo) V.follow(S.geo); kick(); });
    const vbtns = h('div', { class: 'ov br' }, E.followBtn, vbtn(ICON.top, 'Top view', () => { V.orbit.phi = 0.06; V.dirty = true; kick(); }),
      vbtn(ICON.home, 'Reset view', () => { if (S.geo) { V.home(S.geo); kick(); } }), vbtn(ICON.full, 'Full screen', () => document.fullscreenElement ? document.exitFullscreen() : E.vpCard.requestFullscreen()));
    E.loading = h('div', { class: 'loading', hidden: true }, h('div', { class: 'glass box' }, h('div', { class: 'spinner' }), E.loadMsg = h('span', null, 'Loading')));
    E.hint = h('div', { class: 'hint' }, 'Drag to rotate · right-drag to pan · scroll to zoom · space to play');
    E.vpCard = h('div', { class: 'vpcard' }, E.vp, h('div', { class: 'ov tl' }, modebar), h('div', { class: 'ov tr' }, layBtn, E.layers),
      h('div', { class: 'ov bl' }, E.legend), vbtns, E.hint, E.loading);
    E.cams = range(UMAX).map(u => {
      const cv = h('canvas', { width: PW / 2, height: PH / 2 }), lcv = h('canvas', { class: 'lbar' }), nm = h('span'), st = h('span', { class: 'nst' }), lst = h('div', { class: 'lst' });
      const box = h('div', { class: 'ncard' }, h('div', { class: 'nhead' }, h('div', { class: 'nname' }, h('i', { style: { background: NODE_HEX[u] } }), nm), st), cv, lcv, lst);
      cv.addEventListener('mousemove', e => camTip(e, u, cv)); cv.addEventListener('mouseleave', () => { E.tip.hidden = true; });
      return { cv, lcv, nm, st, lst, box };
    });
    E.side = h('div', { class: 'side' }, h('div', { class: 'stitle' }, 'Node cameras and uplink blocks'), E.cams.map(c => c.box),
      h('div', { class: 'lkey' }, h('span', null, h('i', { style: { background: '#f59e0b' } }), 'digital, decoded'), h('span', null, h('i', { style: { background: '#e11d48' } }), 'digital, lost'),
        h('span', null, h('i', { style: { background: 'linear-gradient(90deg,rgba(2,132,199,.25),rgba(2,132,199,.95))' } }), 'analog, gain |α|²')));
    E.play = h('button', { class: 'play', html: ICON.play, title: 'Play / pause (space)' }); E.play.addEventListener('click', () => S.playing ? pause() : play());
    const stepBtn = (icon, d) => { const b = h('button', { class: 'vbtn', html: icon }); b.addEventListener('click', () => { pause(); goTo(clamp(S.i + d, 0, S.clip.n - 1)); }); return b; };
    E.tm = h('b', null, ''); E.tms = h('span', null, '');
    E.spark = h('canvas'); E.sparkL = h('div', { class: 'sparkl' });
    const sp = h('div', { class: 'spark' }, E.spark, E.sparkL);
    sp.addEventListener('pointerdown', e => { sp._drag = true; sp.setPointerCapture(e.pointerId); seek(e, sp); });
    sp.addEventListener('pointermove', e => { if (sp._drag) seek(e, sp); });
    sp.addEventListener('pointerup', e => { if (sp._drag) seek(e, sp, true); sp._drag = false; });
    E.speed = seg([0.5, 1, 2, 4].map(v => ({ v, t: v + '×' })), 1, v => { S.speed = v; if (S.playing) { setPace(); S.t0 = S.clip.t[S.i]; S.w0 = performance.now(); } }, 'small');
    E.net = h('div', { class: 'net', title: 'Frames of this scene held in memory and current download speed' });
    const tl = h('div', { class: 'timeline' }, E.play, h('div', { class: 'tmw' }, stepBtn(ICON.prev, -1), h('div', { class: 'tm' }, E.tm, E.tms), stepBtn(ICON.next, 1)), sp,
      h('div', { class: 'spd' }, E.speed, E.net));
    E.tip = h('div', { class: 'tip', hidden: true }); document.body.append(E.tip);
    root.append(h('div', { class: 'demo' }, bar, kpis, h('div', { class: 'stage' }, E.vpCard, E.side), tl));
    V = new Viewer(E.vp);
    if (matchMedia('(pointer:coarse)').matches || innerWidth < 900) setRes('fast');
    new IntersectionObserver(es => { V.onscreen = es[es.length - 1].isIntersecting; if (V.onscreen) { V.dirty = true; kick(); } }).observe(E.vpCard);
    document.addEventListener('keydown', e => {
      if (e.target.closest('input,select,textarea')) return;
      const r = E.vpCard.getBoundingClientRect(); if (r.bottom < 0 || r.top > innerHeight) return;
      if (e.code === 'Space') { e.preventDefault(); S.playing ? pause() : play(); }
      else if (e.code === 'ArrowRight') { pause(); goTo(Math.min(S.clip.n - 1, S.i + 1)); }
      else if (e.code === 'ArrowLeft') { pause(); goTo(Math.max(0, S.i - 1)); }
    });
    new ResizeObserver(() => { drawSpark(); paintLinks(); }).observe(root);
    syncFollow();
  }

  async function selectDataset(id) {
    pause();
    S.ds = S.D.datasets.find(d => d.id === id); S.U = S.ds.nodes.length; S.node = -1; E.nodeSeg.set(-1); E.dsSeg.set(id);
    for (const b of E.nodeSeg.children) b.hidden = +b.dataset.v >= S.U;
    E.side.classList.toggle('n5', S.U > 3);
    S.layers.frusta = !air(); E.frChk.querySelector('input').checked = S.layers.frusta;
    E.clipSel.replaceChildren(...S.ds.clips.map(c => h('option', { value: c.id }, `${c.title} · ${c.n} frames`)));
    V.glyphs.userData.ds = null;
    await selectClip(S.ds.clips[0].id);
  }
  async function selectClip(id) {
    pause(); resetNet();
    S.clip = S.ds.clips.find(c => c.id === id); E.clipSel.value = id;
    S.i = 0; S.want = 0; S.fr = null; S.geo = null; S.streamHi = S.res === 'full'; V.centre.userData.init = false;
    E.cams.forEach((c, u) => { c.box.hidden = u >= S.U; if (u < S.U) c.nm.textContent = S.ds.nodes[u]; });
    busy(true, 'Loading scene');
    NET.hold = true;   // the first frame gets the whole bandwidth
    try {
      for (const u of urls(0, S.res === 'full')) fetchRaw(u, true).catch(() => {});
      const c = (await chunk(urlT(0, S.res === 'full'))).arr.cams;
      S.O = [0, 1, 2].map(k => range(S.U).reduce((a, u) => a + c[u * 16 + 13 + k], 0) / S.U); S.Ov = S.O.map(x => Math.floor(x / 0.5));
      await goTo(0, { home: true, quiet: true });
    } catch (e) { console.error(e); }
    NET.hold = false; pump();
    busy(false); legend(); drawSpark();
  }
  function selectMode(m) {
    S.mode = m; for (const s of E.modeSegs) s.set(m);
    if (S.fr) { if (needX() && !S.fr.cells) goTo(S.i); else recolor(); }
    legend();
  }
  function setRes(v, user) {
    S.res = v; if (user) S.userQ = true;
    if (v !== 'auto') S.streamHi = v === 'full';
    E.qSeg.set(v); V.pixelRatio(); pump();
    if (S.fr && !S.playing) goTo(S.i, { quiet: true });
  }
  function busy(on, msg) { E.loading.hidden = !on; E.loading.classList.toggle('soft', !!S.fr); if (msg) S.busyMsg = msg; netUI(); }

  // resolution of frame i: the streamed one during playback; when paused the full one as soon as it is loaded
  const hiFor = i => S.playing ? hiPlay() : S.res === 'full' || (S.res === 'auto' && has(i, true));
  async function goTo(i, opt = {}) {
    const tok = ++S.token, hi = hiFor(i);
    S.want = i;
    if (!has(i, hi)) { const near = new Set(wantList().slice(0, 9)); for (const [u, j] of NET.live) if (!j.urgent && !near.has(u)) j.ctl.abort(); }
    pump();
    const tm = opt.quiet ? 0 : setTimeout(() => { if (tok === S.token) busy(true, 'Loading'); }, 300);
    let fr;
    try { fr = await prepare(i, hi); }
    catch (e) { clearTimeout(tm); if (tok === S.token) { busy(false); if (e.message !== 'stale') console.error(e); } return false; }
    clearTimeout(tm);
    if (tok !== S.token) return false;
    if (!opt.quiet) busy(false);
    show(i, fr, opt);
    return true;
  }
  function show(i, fr, opt) {
    S.i = i; S.fr = fr; S.geo = fr.geo;
    if (opt.home) V.home(S.geo);
    LIM.clip.value = air() ? 1e6 : S.geo.ground + 12; LIM.rad.value = air() ? 1e6 : 48; LIM.maxd.value = air() ? 1e6 : 32; LIM.ctr.value.set(S.geo.centre[0], S.geo.centre[2]);
    V.setSurfels(fr); paintAll();
    V.nodes(fr, S.geo); if (S.layers.cells) V.setCells(fr); V.ground(S.geo); V.visibility();
    if (S.playing || S.follow) V.follow(S.geo);
    kpis(); paintLinks(); drawSpark();
    E.tm.textContent = `${i + 1} / ${S.clip.n}`; E.tms.textContent = `${fr.t.toFixed(1)} s · ${S.ds.title} · ${S.clip.title}`;
    S.shown = performance.now(); ahead(i); kick();
  }
  function paintAll() {
    V.paint(S.mode === 'rgb' || S.mode === 'gt' ? null : tokenColors(S.mode, S.fr, S.geo, S.rgba)); paintCams();
  }
  function recolor() { if (S.fr) { paintAll(); kick(); } }
  let aheadTok = 0;
  function ahead(i) {   // during playback, decode the next chunks and frames while the current one is shown
    const tok = ++aheadTok, n = S.clip.n, hi = hiPlay();
    if (!S.playing) return;
    for (const u of urls(Math.min((grp(i) + 1) * S.clip.chunk, n - 1), hi)) if (NET.raw.has(u)) chunk(u).catch(() => {});
    setTimeout(async () => {
      for (const f of [(i + 1) % n, (i + 2) % n]) {
        if (tok !== aheadTok || !has(f, hi)) return;
        try { await prepare(f, hi); } catch (e) { return; }
      }
    }, 30);
  }
  function sharpen() {   // a paused frame shown at half resolution is replaced once its full resolution has arrived
    if (!S.playing && !S.wait && S.fr && !S.fr.hi && S.res !== 'fast' && has(S.i, true)) goTo(S.i, { quiet: true });
  }

  const camCv = h('canvas', { width: PW / 2, height: PH / 2 }), camCtx = camCv.getContext('2d'), camImg = camCtx.createImageData(PW / 2, PH / 2);
  function paintCams() {
    if (!S.fr) return;
    const buf = S.mode !== 'rgb' ? V.cards() : null, n = PW / 2 * PH / 2 * 4;
    for (let u = 0; u < S.U; u++) {
      const c = E.cams[u], ctx = c.cv.getContext('2d'), W = c.cv.width, H = c.cv.height;
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(S.fr.bmps[u], 0, 0, W, H);
      if (buf) { camImg.data.set(buf.subarray(u * n, (u + 1) * n)); camCtx.putImageData(camImg, 0, 0); ctx.drawImage(camCv, 0, 0, W, H); }
      c.box.style.opacity = S.node >= 0 && S.node !== u ? 0.5 : 1;
      const nd = S.fr.meta.nodes[u]; c.st.textContent = `${(nd.bits / 1000).toFixed(1)} kbit`;
    }
  }
  function camTip(e, u, cv) {
    const fr = S.fr; if (!fr) return;
    const r = cv.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * GW), y = Math.floor((e.clientY - r.top) / r.height * GH), t = u * N + y * GW + x;
    if (x < 0 || y < 0 || x >= GW || y >= GH || !S.geo.valid[t]) { E.tip.hidden = true; return; }
    const nm = c => c < 10 ? PAL[S.ds.id].names[c] : '—', row = (a, b) => `<div class="tr"><span>${a}</span><span>${b}</span></div>`;
    E.tip.innerHTML = `<b>${S.ds.nodes[u]} · token (${x}, ${y})</b>` + row('Ground truth', nm(fr.gt[t])) + row('Node decision', nm(fr.dec[t])) +
      (fr.fl ? row('Decision received', (fr.fl[t] & 1) ? 'yes' : 'no') + row('Analog rows', fr.k[t]) + row('Prediction', nm(fr.fc[t])) : '') + row('Fused', nm(fr.fu[t]));
    E.tip.hidden = false;
    E.tip.style.left = Math.min(e.clientX + 14, innerWidth - E.tip.offsetWidth - 10) + 'px'; E.tip.style.top = Math.min(e.clientY + 14, innerHeight - E.tip.offsetHeight - 10) + 'px';
  }
  function size(cv) { const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2), w = Math.max(10, Math.round(r.width * dpr)), hh = Math.max(10, Math.round(r.height * dpr)); if (cv.width !== w || cv.height !== hh) { cv.width = w; cv.height = hh; } return [w, hh, dpr]; }
  function paintLinks() {
    const fr = S.fr; if (!fr) return;
    for (let u = 0; u < S.U; u++) {
      const nd = fr.meta.nodes[u], c = E.cams[u], [W, H, dpr] = size(c.lcv), ctx = c.lcv.getContext('2d');
      ctx.clearRect(0, 0, W, H);
      const nD = nd.nd, nA = nd.ga.length, tot = Math.max(fr.meta.nb, nD + nA, 1), gap = Math.min(3 * dpr, W / tot / 4), bw = (W - gap * (tot - 1)) / tot;
      for (let b = 0; b < tot; b++) {
        ctx.fillStyle = b < nD ? (nd.okb[b] ? 'rgba(245,158,11,.95)' : 'rgba(225,29,72,.85)') : b - nD < nA ? `rgba(2,132,199,${0.22 + 0.7 * Math.min(nd.ga[b - nD], 3.2) / 3.2})` : 'rgba(15,23,42,.06)';
        ctx.beginPath(); ctx.roundRect(b * (bw + gap), 0, Math.max(bw, 0.5), H, 3 * dpr); ctx.fill();
      }
      c.lst.innerHTML = `<b style="color:#b45309">${nD}</b> digital · <b style="color:#0369a1">${nA}</b> analog · ${(100 * nd.recv / Math.max(nd.sent, 1)).toFixed(0)}% decisions received`;
    }
  }
  function kpis() {
    const m = S.fr.meta, st = S.clip.settings[tag()];
    const set = (id, v, p, d) => { const k = E['k_' + id]; k.v.innerHTML = v; k.b.style.width = clamp(p, 0, 100) + '%'; k.d.textContent = d; };
    const fu = miou(m.cm_fu), fc = miou(m.cm_fc);
    set('fu', `${fu.toFixed(1)}<small>%</small>`, fu, `scene ${st.miou.toFixed(1)}% · ${S.snr} dB · bandwidth ${BW_LABEL[S.bw]}`);
    set('fc', `${fc.toFixed(1)}<small>%</small>`, fc, `${(100 * m.seen).toFixed(0)}% of tokens at observed places · scene ${st.miou_fc.toFixed(1)}%`);
    set('nl', m.nll.toFixed(2), 100 * clamp((1.5 - m.nll) / 3, 0, 1), `N(0, I) reference ${m.nll0.toFixed(2)}`);
    const sent = m.nodes.reduce((a, x) => a + x.sent, 0), recv = m.nodes.reduce((a, x) => a + x.recv, 0), bits = m.nodes.reduce((a, x) => a + x.bits, 0), bc = m.nodes.reduce((a, x) => a + x.bits_ctx, 0);
    set('rx', `${(100 * recv / Math.max(sent, 1)).toFixed(0)}<small>%</small>`, 100 * recv / Math.max(sent, 1), `${(bits / 1000).toFixed(1)} kbit, ${(100 * (1 - bits / Math.max(bc, 1))).toFixed(0)}% fewer with world-model side information`);
  }
  function legend() {
    if (!S.ds) return;
    const pal = PAL[S.ds.id], m = S.mode;
    let html = '';
    if (['rgb', 'gt', 'dec', 'fc', 'fu'].includes(m)) html = `<div class="lt">CLASSES</div><div class="items">${pal.names.map((n, i) => `<span><i style="background:${rgb(pal.colors[i])}"></i>${n}</span>`).join('')}</div>`;
    else if (m === 'err') html = `<div class="lt">FUSED VS GROUND TRUTH</div><div class="items"><span><i style="background:${rgb(ERR)}"></i>error</span><span><i style="background:#c9d4dc"></i>correct</span></div>`;
    else if (m === 'k') html = `<div class="lt">ANALOG ROWS PER TOKEN</div><div class="cbar" style="background:${cmCss('viridis')}"></div><div class="cbl"><span>0</span><span>8</span><span>16</span></div>`;
    else if (m === 'lv') html = `<div class="lt">UNCERTAINTY LEVEL</div><div class="cbar" style="background:${cmCss('purples')}"></div><div class="cbl"><span>0</span><span>7</span></div>`;
    else if (m === 'nl') html = `<div class="lt">JEPA NLL (NAT/DIM)</div><div class="cbar" style="background:${cmCss('ylorrd')}"></div><div class="cbl"><span>−3</span><span>1</span><span>5</span></div>`;
    else if (m === 'rx') html = `<div class="lt">DIGITAL LAYER</div><div class="items"><span><i style="background:${rgb(RX)}"></i>decision received</span><span><i style="background:${rgb(NORX)}"></i>lost</span></div>`;
    E.legend.innerHTML = html;
  }
  function drawSpark() {
    if (!S.clip || !E.spark) return;
    const [W, H, dpr] = size(E.spark), ctx = E.spark.getContext('2d'), n = S.clip.n, st = S.clip.settings[tag()];
    ctx.clearRect(0, 0, W, H);
    const ser = [{ c: '#0d9488', n: 'Fused', d: st.fu, w: 2 }, { c: '#7c3aed', n: 'Prediction before reception', d: st.fc, w: 1.4 }];
    const all = ser.flatMap(s => s.d.filter(v => v != null)).sort((a, b) => a - b), lo = clamp(Math.floor(((all[Math.floor(all.length * 0.04)] ?? 50) - 6) / 10) * 10, 0, 80);
    const pad = 6 * dpr, ty = H - 5 * dpr, X = i => pad + (W - 2 * pad) * (n > 1 ? i / (n - 1) : 0), Y = v => ty - 7 * dpr - (ty - 17 * dpr) * (Math.max(v, lo) - lo) / (100 - lo);
    ctx.strokeStyle = 'rgba(15,23,42,.06)'; ctx.lineWidth = 1;
    for (let g = lo; g <= 100; g += 10) { ctx.beginPath(); ctx.moveTo(pad, Y(g)); ctx.lineTo(W - pad, Y(g)); ctx.stroke(); }
    const smooth = d => d.map((_, i) => { let s = 0, k = 0; for (let j = Math.max(0, i - 2); j <= Math.min(d.length - 1, i + 2); j++) if (d[j] != null) { s += d[j]; k++; } return k ? s / k : null; });
    for (const s of ser.slice().reverse()) {
      const d = n > 60 ? smooth(s.d) : s.d;
      ctx.beginPath(); let started = false;
      d.forEach((v, i) => { if (v == null) return; if (!started) { ctx.moveTo(X(i), Y(v)); started = true; } else ctx.lineTo(X(i), Y(v)); });
      ctx.strokeStyle = s.c; ctx.lineWidth = s.w * dpr; ctx.lineJoin = 'round'; ctx.stroke();
    }
    const fw = (W - 2 * pad) / Math.max(n - 1, 1) / 2, bar = (a, b) => { const x0 = Math.max(pad, X(a) - fw); ctx.fillRect(x0, ty, Math.min(W - pad, X(b) + fw) - x0, 3 * dpr); };
    ctx.fillStyle = 'rgba(15,23,42,.08)'; bar(0, n - 1);
    ctx.fillStyle = 'rgba(100,116,139,.4)';
    for (let a = 0; a < n; a++) { if (!isBuf(a)) continue; let b = a; while (b + 1 < n && isBuf(b + 1)) b++; bar(a, b); a = b; }
    ctx.fillStyle = '#0d9488'; bar(0, S.i);
    const x = X(S.i); ctx.strokeStyle = 'rgba(15,23,42,.5)'; ctx.lineWidth = 1.2 * dpr; ctx.beginPath(); ctx.moveTo(x, 2 * dpr); ctx.lineTo(x, ty + 3 * dpr); ctx.stroke();
    ctx.fillStyle = 'rgba(100,116,139,.85)'; ctx.font = `${10 * dpr}px ${getComputedStyle(document.body).fontFamily}`; ctx.fillText(`mIoU ${lo}–100%`, pad, 11 * dpr);
    E.sparkL.innerHTML = ser.map(s => `<span><i style="background:${s.c}"></i>${s.n}</span>`).join('');
    E.spark._g = { pad, W, n, dpr };
  }
  function seek(e, el, final) {
    const g = E.spark._g; if (!g) return;
    const r = el.getBoundingClientRect(), i = clamp(Math.round(((e.clientX - r.left) * g.dpr - g.pad) / (g.W - 2 * g.pad) * (g.n - 1)), 0, g.n - 1);
    pause();
    if (i === S.i) return;
    if (final || isBuf(i)) goTo(i); else { S.want = i; pump(); }
  }

  // ---------------------------------------------------------------- loading state and playback
  const mbs = b => b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB/s` : `${Math.max(1, Math.round(b / 1e3))} kB/s`;
  function netUI() {
    if (!S.clip || !E.net) return;
    let b = 0; for (let f = 0; f < S.clip.n; f++) if (isBuf(f)) b++;
    const pct = Math.floor(100 * b / S.clip.n), sp = NET.live.size ? speed(5000) : 0, slow = S.playing && S.pace < 0.99 ? ` · ${(S.speed * S.pace).toFixed(1)}×` : '';
    E.net.textContent = `Loaded ${pct}%` + (sp && pct < 100 ? ` · ${mbs(sp)}` : '') + slow;
    if (!E.loading.hidden) E.loadMsg.textContent = S.busyMsg + (sp ? ` · ${mbs(sp)}` : '');
  }
  let netT = 0;
  function netChanged() { if (!netT) netT = setTimeout(() => { netT = 0; drawSpark(); netUI(); sharpen(); }, 200); }
  setInterval(() => { if (!E.loading) return; if (S.wait) waitUI(); else if (NET.live.size || !E.loading.hidden || S.playing) netUI(); }, 400);
  const fps = (p = S.pace) => S.speed * p * (S.clip.n - 1) / Math.max(S.clip.t[S.clip.n - 1] - S.clip.t[0], 1e-3);
  // Frames buffered from frame s on, and frames needed before playback at pace p can run to the end without stalling at
  // the speed measured over the last 15 s.
  function need(s, hi = hiPlay(), p = S.pace, m = 1) {
    const n = S.clip.n, rem = n - s;
    let a = 0; while (a < rem && has(s + a, hi)) a++;
    if (a >= rem) return [a, a];
    const rate = m * speed(15000) / Math.max(bytesPerFrame(hi), 1);
    return [a, Math.min(rem, Math.ceil(rem * Math.max(0, 1 - rate / fps(p))) + 5)];
  }
  const nextFrame = () => S.i >= S.clip.n - 1 ? 0 : S.i + 1;
  const canPlay = (hi, p, m) => { const [a, k] = need(nextFrame(), hi, p, m); return a >= k; };
  function decideRes() {   // Auto: full resolution when it is loaded or the measured speed sustains it with a margin
    if (S.res === 'auto') S.streamHi = canPlay(true, 1) || speed(15000) > 1.3 * bytesPerFrame(true) * fps(1);
  }
  // Playback pace: 1 when the loaded frames and the measured speed sustain real time, otherwise the pace the network
  // sustains with a margin, not below 0.35. Playback thus keeps moving on a slow connection instead of stopping to buffer.
  function setPace() {
    const rate = speed(15000) / Math.max(bytesPerFrame(hiPlay()), 1);
    S.pace = canPlay(undefined, 1, 0.8) ? 1 : clamp(0.8 * rate / fps(1), 0.35, 1);
  }
  function waitUI() {   // the shown percentage never decreases within one wait
    const [a, k] = need(nextFrame());
    S.waitPct = Math.max(S.waitPct, Math.min(99, Math.floor(100 * a / Math.max(k, 1))));
    busy(true, `Buffering ${S.waitPct}%`);
  }
  function startWait() { S.wait = true; S.waitPct = 0; waitUI(); }
  function stopWait() { if (S.wait) { S.wait = false; busy(false); } }

  function play() {
    if (!S.clip) return;
    S.playing = true; E.play.innerHTML = ICON.pause; E.hint.style.opacity = 0;
    decideRes();
    if (!hiPlay()) for (const [u, j] of NET.live) if (kind(u) === 't' && !j.urgent) j.ctl.abort();   // full resolution is not streamed
    pump();
    const go = () => { if (!S.playing) return; setPace(); S.t0 = S.clip.t[S.i]; S.w0 = performance.now(); if (!canPlay()) startWait(); kick(); };
    if (S.i >= S.clip.n - 1) { S.busy = true; goTo(0, { quiet: true }).then(() => { S.busy = false; go(); }); } else go();
  }
  function pause() { S.playing = false; stopWait(); if (E.play) E.play.innerHTML = ICON.play; setTimeout(() => { pump(); sharpen(); }, 0); }
  // Playback shows every frame: it waits while the next frame is not loaded, and slows down instead of skipping frames.
  // In Auto, a stall at full resolution switches the stream to half resolution.
  function advance(now) {
    const c = S.clip, n = c.n;
    if (S.wait) {
      if (S.res === 'auto' && S.streamHi && !canPlay(true, 1) && canPlay(false, 1)) { S.streamHi = false; pump(); }
      setPace();
      if (canPlay()) { stopWait(); S.t0 = c.t[S.i]; S.w0 = now; }
      return;
    }
    const tt = S.t0 + (now - S.w0) / 1000 * S.speed * S.pace, j = nextFrame();
    if (j === 0 ? tt < c.t[n - 1] + 1.2 : tt < c.t[j]) return;
    if (S.res === 'auto' && !S.streamHi && canPlay(true, 1)) S.streamHi = true;
    if (!isBuf(j) && S.res === 'auto' && S.streamHi) { S.streamHi = false; pump(); }
    if (!isBuf(j)) { startWait(); return; }
    if (S.pace < 1 && canPlay(undefined, 1, 0.8)) { S.pace = 1; S.t0 = c.t[S.i]; S.w0 = now; }
    S.busy = true;
    goTo(j, { quiet: true }).then(() => {
      S.busy = false;
      const t = performance.now();
      if (j === 0 || S.t0 + (t - S.w0) / 1000 * S.speed * S.pace > c.t[Math.min(j + 1, n - 1)]) { S.t0 = c.t[j]; S.w0 = t; }
    });
  }
  let gaps = [];
  function autoQuality(g) {   // switch to Fast once if the full-resolution view renders below about 20 frames per second
    if (S.res !== 'auto' || S.userQ || !S.fr || !S.fr.hi) return;
    gaps.push(g); if (gaps.length < 40) return;
    gaps.sort((a, b) => a - b); const med = gaps[20]; gaps = [];
    if (med > 50) setRes('fast');
  }
  let rafOn = false, lastT = 0, idle = 0, lastR = 0, prevR = 0;
  function kick() { if (!rafOn && !document.hidden) { rafOn = true; lastT = performance.now(); requestAnimationFrame(loop); } }
  function loop(now) {
    const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
    if (S.playing && S.clip && !S.busy) advance(now);
    const moving = V.step(dt), anim = S.playing && S.layers.links && now - lastR > 45;
    if ((moving || V.dirty || anim) && V.onscreen) {
      V.render(now);
      if (moving && prevR && now - (S.shown || 0) > 300) autoQuality(now - prevR);
      prevR = moving ? now : 0; lastR = now;
    } else prevR = 0;
    idle = moving || S.playing || V.dirty ? 0 : idle + dt;
    if (document.hidden || idle > 1 || (!V.onscreen && !S.playing)) { rafOn = false; return; }
    requestAnimationFrame(loop);
  }

  async function init() {
    const root = $('#demo');
    try { S.D = await (await fetch('data/manifest.json' + VER)).json(); }
    catch (e) { root.append(h('p', { class: 'err' }, 'The demo data could not be loaded.')); return; }
    setMirrors(S.D.mirror);
    if (!window.WebGL2RenderingContext) { root.append(h('p', { class: 'err' }, 'This demo needs a browser with WebGL 2.')); return; }
    build(root);
    window.__demo = { S, NET, get V() { return V; }, goTo, play, pause, selectDataset, selectClip, selectMode, isBuf };
    const q = new URLSearchParams(location.hash.slice(1));
    if (q.get('snr') && S.D.snrs.includes(+q.get('snr'))) { S.snr = +q.get('snr'); E.snrSeg.set(S.snr); }
    if (q.get('bw') && S.D.bws.includes(+q.get('bw'))) { S.bw = +q.get('bw'); E.bwSeg.set(S.bw); }
    if (q.get('mode') && MODES.some(m => m.id === q.get('mode'))) selectMode(q.get('mode'));
    const ds = S.D.datasets.find(d => d.id === q.get('ds')) || S.D.datasets[0];
    await selectDataset(ds.id);
    if (q.get('clip') && ds.clips.some(c => c.id === q.get('clip'))) await selectClip(q.get('clip'));
  }
  init();
})();
