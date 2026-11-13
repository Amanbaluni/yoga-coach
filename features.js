// Must match training/features.py exactly.
export const T = 16;
export const KEY = [0, ...Array.from({ length: 22 }, (_, i) => i + 11)];
const TRI = [[23,25,27],[24,26,28],[11,13,15],[12,14,16],[13,11,23],[14,12,24],[11,23,25],[12,24,26]];
export const ANGLE_NAMES = ["Left knee","Right knee","Left elbow","Right elbow","Left shoulder","Right shoulder","Left hip","Right hip"];
const ang = (a, b, c) => {
  const x1 = a.x - b.x, y1 = a.y - b.y, x2 = c.x - b.x, y2 = c.y - b.y;
  const m = Math.hypot(x1, y1) * Math.hypot(x2, y2) || 1;
  return Math.acos(Math.max(-1, Math.min(1, (x1 * x2 + y1 * y2) / m))) * 180 / Math.PI;
};
export const angles = p => TRI.map(([a, b, c]) => ang(p[a], p[b], p[c]));
export const canon = a => a[0] > a[1] ? [a[1],a[0],a[3],a[2],a[5],a[4],a[7],a[6]] : a;
export function frameFeatures(p) {         // p: 33 landmarks as {x,y} in PIXELS
  const hx = (p[23].x + p[24].x) / 2, hy = (p[23].y + p[24].y) / 2;
  const s = Math.hypot((p[11].x + p[12].x) / 2 - hx, (p[11].y + p[12].y) / 2 - hy) || 1;
  const f = [];
  for (const i of KEY) f.push((p[i].x - hx) / s, (p[i].y - hy) / s);
  return f.concat(angles(p).map(a => a / 180));
}
