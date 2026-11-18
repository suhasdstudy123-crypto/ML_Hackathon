import test from 'node:test';
import assert from 'node:assert/strict';
import {reconstructionState, formatFraction, formatDuration, pollJob} from '../js/state.js';

test('connection, loading, missing-model and unavailable-feature states disable reconstruction', () => {
  assert.equal(reconstructionState(null, null).enabled, false);
  assert.equal(reconstructionState({model_ready: false}, {features: {ready: true}}).message, 'LightGBM model not connected');
  assert.equal(reconstructionState({model_ready: true}, null).enabled, false);
  assert.equal(reconstructionState({model_ready: true}, {features: {ready: false}}).enabled, false);
  assert.equal(reconstructionState({model_ready: true}, {features: {ready: true}}, true).enabled, false);
  assert.equal(reconstructionState({model_ready: true}, {features: {ready: true}}).enabled, true);
});
test('metrics distinguish null from genuine zero', () => {
  assert.equal(formatFraction(null), '—'); assert.equal(formatFraction(0), '0.0%'); assert.equal(formatFraction(.85), '85.0%');
  assert.equal(formatDuration(null), '—'); assert.equal(formatDuration(12.3), '12 ms');
});
test('SYNTHETIC UNIT TEST job polling follows real stages and retrieves a completed result', async () => {
  const statuses = ['QUEUED', 'SCORING_CANDIDATES', 'COMPLETED'], seen = [];
  const fixture = {schema_version: '1.0', trip_id: 'SYNTHETIC_UNIT_TEST_ONLY'};
  const api = {job: async () => ({status: statuses.shift()}), result: async () => fixture};
  assert.equal(await pollJob(api, 'test-job', j => seen.push(j.status), {wait: async () => {}}), fixture);
  assert.deepEqual(seen, ['QUEUED', 'SCORING_CANDIDATES', 'COMPLETED']);
});
test('polling retries transient reads without resubmitting inference', async () => {
  let calls = 0; const waits = [];
  const api = {job: async () => {if (++calls < 3) throw new Error('network'); return {status: 'COMPLETED'};}, result: async () => 'done'};
  assert.equal(await pollJob(api, 'test-job', () => {}, {wait: async ms => waits.push(ms)}), 'done');
  assert.deepEqual(waits, [1000, 2000]);
});
test('backend restart and failed jobs surface errors', async () => {
  await assert.rejects(pollJob({job: async () => {throw Object.assign(new Error('Job expired'), {status: 404});}}, 'old', () => {}), /expired/);
  await assert.rejects(pollJob({job: async () => ({status: 'FAILED', error: {message: 'Decoder failed'}})}, 'bad', () => {}), /Decoder failed/);
});
test('changing trips cancels old polling and bounded polling stops', async () => {
  let called = false;
  assert.equal(await pollJob({job: async () => {called = true;}}, 'old', () => {}, {isCancelled: () => true}), null); assert.equal(called, false);
  await assert.rejects(pollJob({job: async () => ({status: 'QUEUED'})}, 'long', () => {}, {maxPolls: 1, wait: async () => {}}), /Backend work continues/);
});
