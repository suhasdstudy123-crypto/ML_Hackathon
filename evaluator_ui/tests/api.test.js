import test from 'node:test';
import assert from 'node:assert/strict';
import {ApiClient, normalizeBase, ApiError} from '../js/api.js';

test('configurable LAN address is preserved and unsafe origins are rejected', () => {
  assert.equal(normalizeBase(' http://192.168.1.20:8000/ '), 'http://192.168.1.20:8000');
  for (const value of ['file:///tmp/test', 'javascript:alert(1)', 'http://user:password@host', 'http://host/path', 'http://host?query=1', 'wrong']) assert.throws(() => normalizeBase(value), ApiError);
});
test('typed request paths preserve trip and job IDs', async () => {
  const seen = [];
  const api = new ApiClient('http://192.168.1.20:8000', async (url, options) => {seen.push([url, options]); return new Response(JSON.stringify({schema_version: '1.0'}));});
  await api.preview('00000023'); await api.start('00000023', false); await api.result('job-1');
  assert.equal(seen[0][0], 'http://192.168.1.20:8000/api/v1/trips/00000023/preview');
  assert.deepEqual(JSON.parse(seen[1][1].body), {trip_id: '00000023', use_hmm: false});
  assert.equal(seen[2][0], 'http://192.168.1.20:8000/api/v1/reconstructions/job-1/result');
});
test('missing-model error remains structured and no fallback is used', async () => {
  const api = new ApiClient('http://host', async () => new Response(JSON.stringify({schema_version: '1.0', error: {code: 'MODEL_NOT_READY', message: 'LightGBM model not connected'}}), {status: 503}));
  await assert.rejects(api.start('00000023'), err => err.code === 'MODEL_NOT_READY' && err.status === 503);
});
test('unsupported response schema is rejected', async () => {
  const api = new ApiClient('http://host', async () => new Response('{"schema_version":"2.0"}'));
  await assert.rejects(api.health(), err => err.code === 'UNSUPPORTED_SCHEMA');
});
test('network errors provide connection guidance', async () => {
  const api = new ApiClient('http://host', async () => {throw new TypeError('Failed to fetch');});
  await assert.rejects(api.health(), /LAN address.*firewall/);
});
test('browser fetch is invoked with its global receiver', async () => {
  const api = new ApiClient('http://host', async function () {assert.equal(this, globalThis); return new Response('{"schema_version":"1.0"}');});
  await api.health();
});
