import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CLI_PATH = resolve(__dirname, '..', 'bin', 'geometry-cli.js');
const PACKAGE_ROOT = resolve(__dirname, '..');

function runOps(requests, { timeoutMs = 10_000 } = {}) {
  return new Promise((resolveProm, rejectProm) => {
    const proc = spawn('node', [CLI_PATH], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: PACKAGE_ROOT,
    });
    const responses = [];
    let stdoutBuf = '';
    let stderrBuf = '';
    const timer = setTimeout(() => {
      proc.kill();
      rejectProm(new Error(`CLI timeout. stderr: ${stderrBuf}`));
    }, timeoutMs);

    proc.stdout.on('data', (chunk) => {
      stdoutBuf += chunk.toString('utf8');
      let idx;
      while ((idx = stdoutBuf.indexOf('\n')) >= 0) {
        const line = stdoutBuf.slice(0, idx);
        stdoutBuf = stdoutBuf.slice(idx + 1);
        if (line.trim()) responses.push(JSON.parse(line));
      }
    });
    proc.stderr.on('data', (c) => { stderrBuf += c.toString('utf8'); });
    proc.on('error', (err) => { clearTimeout(timer); rejectProm(err); });
    proc.on('close', () => { clearTimeout(timer); resolveProm({ responses, stderr: stderrBuf }); });

    for (const req of requests) proc.stdin.write(JSON.stringify(req) + '\n');
    proc.stdin.end();
  });
}

test('geometry-cli health responds OK', async () => {
  const { responses } = await runOps([{ id: '1', op: 'health' }]);
  assert.equal(responses.length, 1);
  assert.equal(responses[0].id, '1');
  assert.equal(responses[0].result.status, 'ok');
});

test('geometry-cli compute_osse_profile returns expected mouth radius', async () => {
  const { responses } = await runOps([{
    id: 'osse',
    op: 'compute_osse_profile',
    params: {
      phi: 0,
      t_values: [0, 0.5, 1.0],
      params: { L: 120, r0: 12.7, a0: 15.5, a: 60, k: 1.0, n: 4.0, q: 0.995, s: 0.0 },
    },
  }]);
  assert.equal(responses.length, 1);
  const r = responses[0].result;
  assert.equal(r.total_length, 120);
  assert.equal(r.x[0], 0);
  assert.equal(r.x[2], 120);
  assert.ok(Math.abs(r.y[0] - 12.7) < 1e-9, `y[0] = ${r.y[0]}`);
  // Pinned 2026-05-18 from canonical WG evaluator.
  assert.ok(Math.abs(r.y[2] - 210.25359737777225) < 1e-9, `y[-1] = ${r.y[2]}`);
});

test('geometry-cli unknown op surfaces an error response', async () => {
  const { responses } = await runOps([{ id: 'bad', op: 'nonexistent' }]);
  assert.equal(responses.length, 1);
  assert.equal(responses[0].id, 'bad');
  assert.match(responses[0].error, /unknown op/);
});

test('geometry-cli rejects non-OSSE point-grid types', async () => {
  const { responses } = await runOps([{
    id: 'lookup',
    op: 'build_point_grid',
    params: {
      params: {
        type: 'LOOKUP',
        lookupPoints: [[0, 12.7], [100, 120]],
        angularSegments: 8,
        lengthSegments: 4,
      },
    },
  }]);
  assert.equal(responses.length, 1);
  assert.equal(responses[0].id, 'lookup');
  assert.match(responses[0].error, /supports only OSSE or R-OSSE/);
});
