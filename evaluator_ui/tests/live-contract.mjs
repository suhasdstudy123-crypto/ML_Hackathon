// Integration check against an actual API process. No synthetic routes or prediction success.
import assert from 'node:assert/strict';
import {ApiClient} from '../js/api.js';
import {toLeaflet} from '../js/map.js';
const api = new ApiClient(process.argv[2] || 'http://127.0.0.1:8000');
const health = await api.health(); assert.equal(health.schema_version, '1.0');
const {trips} = await api.trips(); assert.equal(trips.length, 10);
const trip = trips.includes('00000023') ? '00000023' : trips[0];
const [metadata, preview] = await Promise.all([api.metadata(trip), api.preview(trip)]);
assert.equal(preview.trip_id, trip);
assert.equal(preview.geometry.raw_gps.features.length, metadata.observation_count);
assert.ok(preview.geometry.road_network.features.length);
const point = preview.geometry.raw_gps.features[0].geometry.coordinates;
assert.deepEqual(toLeaflet(point), [point[1], point[0]]);
const forbidden = /ground_truth|aligned_edge|aligned_route|alignment_status|usable_for_training|"label"/i;
assert.equal(forbidden.test(JSON.stringify(preview)), false);
if (!health.model_ready) await assert.rejects(api.start(trip), err => err.status === 503 && err.code === 'MODEL_NOT_READY');
console.log(JSON.stringify({result:'PASS',trip_count:trips.length,trip_id:trip,observations:metadata.observation_count,road_preview_arcs:preview.geometry.road_network.features.length,model_ready:health.model_ready,reference_fields_exposed:false}));
