#!/usr/bin/env node
// NDJSON REPL bridging @hornlab/geometry to Python (and other) consumers.
//
// Protocol:
//   stdin: one JSON object per line: {"id": "...", "op": "...", "params": {...}}
//   stdout: one JSON object per line: {"id": "...", "result": {...}} or {"id": "...", "error": "..."}
//   stderr: free-form logs, never consumed as protocol.
//
// Operations:
//   health                  -> {"status": "ok", "version": "..."}
//   compute_osse_profile    -> {x: [...], y: [...], total_length}
//   compute_rosse_profile   -> {x: [...], y: [...]}
//   compute_lookup_profile  -> {x: [...], y: [...], total_length}
//   build_inner_points      -> {inner_points: [...], grid_n_phi, grid_n_length, full_circle, angle_list, slice_map}
//   build_point_grid        -> {inner_points: [...], outer_points: [...] | null, grid_n_phi, grid_n_length, full_circle, angle_list, slice_map}
//
// Run: node bin/geometry-cli.js
// Smoke: echo '{"id":"1","op":"health"}' | node bin/geometry-cli.js

import readline from 'node:readline';
import { calculateOSSE, calculateROSSE } from '@hornlab/geometry/engine/index.js';
import { prepareGeometryParams } from '@hornlab/geometry/params.js';
import { pchipEval } from '@hornlab/geometry/engine/interp.js';
import { buildGeometryShape, buildGeometryMeshFromShape } from '@hornlab/geometry/pipeline.js';
import { extractPointGrid } from '@hornlab/geometry/pointGridExtractor.js';

const VERSION = '0.1.0';

function computeOsseProfile({ phi = 0, t_values, params }) {
  if (!Array.isArray(t_values)) throw new Error('compute_osse_profile: t_values must be an array');
  if (!params || typeof params !== 'object') throw new Error('compute_osse_profile: params required');

  const prepared = prepareGeometryParams({ ...params, type: 'OSSE' });
  const evalOrConst = (v, p) => (typeof v === 'function' ? v(p) : Number(v ?? 0));
  const L = evalOrConst(prepared.L, phi);
  const extLen = Math.max(0, evalOrConst(prepared.throatExtLength ?? 0, phi));
  const slotLen = Math.max(0, evalOrConst(prepared.slotLength ?? 0, phi));
  const totalLength = L + extLen + slotLen;

  const x = new Array(t_values.length);
  const y = new Array(t_values.length);
  for (let i = 0; i < t_values.length; i += 1) {
    const t = t_values[i];
    const z = t * totalLength;
    const { x: xi, y: yi } = calculateOSSE(z, phi, prepared);
    x[i] = xi;
    y[i] = yi;
  }
  return { x, y, total_length: totalLength };
}

function computeRosseProfile({ phi = 0, t_values, params }) {
  if (!Array.isArray(t_values)) throw new Error('compute_rosse_profile: t_values must be an array');
  if (!params || typeof params !== 'object') throw new Error('compute_rosse_profile: params required');

  const prepared = prepareGeometryParams({ ...params, type: 'ROSSE' });
  const x = new Array(t_values.length);
  const y = new Array(t_values.length);
  for (let i = 0; i < t_values.length; i += 1) {
    const t = t_values[i];
    const { x: xi, y: yi } = calculateROSSE(t, phi, prepared);
    x[i] = xi;
    y[i] = yi;
  }
  // ROSSE produces x in [0, L]; total length equals the last x value when valid.
  const last = x[x.length - 1];
  return { x, y, total_length: Number.isFinite(last) ? last : NaN };
}

function computeLookupProfile({ t_values, lookup_points }) {
  if (!Array.isArray(t_values)) throw new Error('compute_lookup_profile: t_values must be an array');
  if (!Array.isArray(lookup_points) || lookup_points.length < 2) {
    throw new Error('compute_lookup_profile: lookup_points must be an array of at least two [z, r] pairs');
  }
  const zs = lookup_points.map((p) => Number(p[0]));
  const rs = lookup_points.map((p) => Number(p[1]));
  const L = zs[zs.length - 1] - zs[0];
  const z0 = zs[0];
  const x = new Array(t_values.length);
  const y = new Array(t_values.length);
  for (let i = 0; i < t_values.length; i += 1) {
    const t = t_values[i];
    const z = z0 + t * L;
    x[i] = z;
    y[i] = pchipEval(zs, rs, z);
  }
  return { x, y, total_length: L };
}

function buildInnerPoints({ params }) {
  if (!params || typeof params !== 'object') {
    throw new Error('build_inner_points: params required');
  }
  const lengthSegments = Number(params.lengthSegments);
  if (!Number.isInteger(lengthSegments) || lengthSegments < 1) {
    throw new Error('build_inner_points: params.lengthSegments must be a positive integer');
  }
  const shape = buildGeometryShape(params);
  const meshData = buildGeometryMeshFromShape(shape, {
    includeEnclosure: false,
    adaptivePhi: false,
  });
  const grid = extractPointGrid(meshData.vertices, meshData.ringCount, lengthSegments, false);
  return {
    inner_points: Array.from(grid.innerPoints),
    grid_n_phi: grid.nPhi,
    grid_n_length: grid.nLength,
    full_circle: Boolean(meshData.fullCircle),
    angle_list: Array.isArray(meshData.angleList) ? [...meshData.angleList] : null,
    slice_map: Array.isArray(meshData.sliceMap) ? [...meshData.sliceMap] : null,
  };
}

function buildPointGrid({ params }) {
  if (!params || typeof params !== 'object') {
    throw new Error('build_point_grid: params required');
  }
  const lengthSegments = Number(params.lengthSegments);
  if (!Number.isInteger(lengthSegments) || lengthSegments < 1) {
    throw new Error('build_point_grid: params.lengthSegments must be a positive integer');
  }

  const prepared = prepareGeometryParams(params, { type: params.type });
  const encDepth = Number(prepared.encDepth || 0);
  const wallThickness = Number(prepared.wallThickness || 0);
  const includeOuter = encDepth <= 0 && wallThickness > 0;
  const gridParams = encDepth > 0
    ? { ...prepared, encDepth: 0, wallThickness: 0 }
    : prepared;

  const shape = buildGeometryShape(gridParams);
  const meshData = buildGeometryMeshFromShape(shape, {
    includeEnclosure: false,
    adaptivePhi: false,
  });
  const grid = extractPointGrid(meshData.vertices, meshData.ringCount, lengthSegments, includeOuter);
  if (includeOuter && grid.outerPoints) {
    for (let i = 0; i < grid.nPhi; i += 1) {
      const throatIdx = i * (grid.nLength + 1) * 3;
      grid.outerPoints[throatIdx + 2] = grid.innerPoints[throatIdx + 2] - wallThickness;
    }
  }
  return {
    inner_points: Array.from(grid.innerPoints),
    outer_points: grid.outerPoints ? Array.from(grid.outerPoints) : null,
    grid_n_phi: grid.nPhi,
    grid_n_length: grid.nLength,
    full_circle: Boolean(meshData.fullCircle),
    angle_list: Array.isArray(meshData.angleList) ? [...meshData.angleList] : null,
    slice_map: Array.isArray(meshData.sliceMap) ? [...meshData.sliceMap] : null,
  };
}

const OPS = {
  health: () => ({ status: 'ok', version: VERSION }),
  compute_osse_profile: computeOsseProfile,
  compute_rosse_profile: computeRosseProfile,
  compute_lookup_profile: computeLookupProfile,
  build_inner_points: buildInnerPoints,
  build_point_grid: buildPointGrid,
};

function handleLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return;
  let id = null;
  try {
    const msg = JSON.parse(trimmed);
    id = msg.id ?? null;
    const op = msg.op;
    if (typeof op !== 'string' || !(op in OPS)) {
      process.stdout.write(JSON.stringify({ id, error: `unknown op: ${op}` }) + '\n');
      return;
    }
    const result = OPS[op](msg.params || {});
    process.stdout.write(JSON.stringify({ id, result }) + '\n');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    process.stdout.write(JSON.stringify({ id, error: message }) + '\n');
  }
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', handleLine);
rl.on('close', () => process.exit(0));

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));
