'use strict';
// Decodes gzip WDB1 chunks off the main thread. For geometry chunks it also rebuilds the depth images (cm) from the
// planar-predicted zigzag residual planes, takes the token depth at the token centres, and returns only the arrays
// the page uses.
const DT = { u1: Uint8Array, i1: Int8Array, u2: Uint16Array, i2: Int16Array, u4: Uint32Array, i4: Int32Array, f4: Float32Array };
const GW = 160, GH = 96;
onmessage = async e => {
  const { id, buf } = e.data;
  try {
    const raw = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    const hl = new DataView(raw).getUint32(4, true), head = JSON.parse(new TextDecoder().decode(new Uint8Array(raw, 8, hl))), arr = {};
    for (const [k, s] of Object.entries(head.arrays)) arr[k] = new DT[s.dtype](raw, s.offset, s.nbytes / DT[s.dtype].BYTES_PER_ELEMENT);
    if (!arr.dlo) { postMessage({ id, meta: head.meta, raw, spec: head.arrays }, [raw]); return; }
    const [F, U, H, W] = head.arrays.dlo.shape, P = H * W, sc = 640 / W, dep = new Float32Array(F * U * P), tok = new Float32Array(F * U * GW * GH);
    for (let m = 0; m < F * U; m++) {
      const lo = arr.dlo.subarray(m * P, (m + 1) * P), hi = arr.dhi.subarray(m * P, (m + 1) * P), q = dep.subarray(m * P, (m + 1) * P);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const p = y * W + x, z = lo[p] | (hi[p] << 8), r = (z >>> 1) ^ -(z & 1);
        q[p] = ((y ? (x ? q[p - 1] + q[p - W] - q[p - W - 1] : q[p - W]) : (x ? q[p - 1] : 0)) + r) & 0xFFFF;
      }
      for (let i = 0; i < GH; i++) for (let j = 0; j < GW; j++) tok[m * GW * GH + i * GW + j] = q[((4 * i + 2) / sc) * W + (4 * j + 2) / sc];
    }
    const t = { dep, tok, w: W, h: H };
    for (const k of ['gt', 'dec', 'cams', 'lab', 'jpg', 'jpg_off']) if (arr[k]) t[k] = arr[k].slice();
    postMessage({ id, meta: head.meta, tex: t }, [dep, tok, t.gt, t.dec, t.cams, t.lab, t.jpg, t.jpg_off].filter(a => a).map(a => a.buffer));
  } catch (err) { postMessage({ id, err: String(err) }); }
};
