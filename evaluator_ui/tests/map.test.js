import test from 'node:test';
import assert from 'node:assert/strict';
import {RouteMap, toLeaflet} from '../js/map.js';

function syntheticLeaflet() {
  const lines = [], points = [], removed = [];
  const map = {setView() {return this;}, fitBounds(bounds) {this.bounds = bounds;}, invalidateSize() {}, removeLayer(layer) {removed.push(layer);}};
  const shape = () => ({addTo(layer) {layer.items?.push(this); return this;}, on(event, fn) {this[event] = fn; return this;}, bindTooltip() {return this;}});
  return {lines, points, removed, mapObject: map, map: () => map, control: {zoom: () => shape()}, tileLayer: () => shape(),
    layerGroup: () => ({items: [], addTo() {return this;}, clearLayers() {this.items = [];}}),
    circleMarker: (coordinates, style) => {points.push(coordinates); return shape();},
    polyline: (coordinates, style) => {lines.push(coordinates); return shape();}};
}
const fc = features => ({type: 'FeatureCollection', features});
const line = (a, b, arc) => ({type: 'Feature', geometry: {type: 'LineString', coordinates: [a, b]}, properties: {arc_id: arc, sequence_index: 0}});

test('GeoJSON coordinates convert exactly once to Leaflet order', () => {
  assert.deepEqual(toLeaflet([8.6, 50.1]), [50.1, 8.6]);
  for (const v of [[181, 50], [8, 91], [NaN, 50], [8, Infinity]]) assert.throws(() => toLeaflet(v));
});
test('SYNTHETIC UNIT TEST preview renders GPS, directed arcs, toggles and inspection', async () => {
  const L = syntheticLeaflet(); let selected;
  const map = new RouteMap('test', value => selected = value, L);
  await map.preview({geometry: {raw_gps: fc([{type: 'Feature', geometry: {type: 'Point', coordinates: [8, 50]}, properties: {observation_index: 0, timestamp: 1}}]), road_network: fc([line([8, 50], [8.1, 50.1], 0)])}});
  assert.deepEqual(L.points, [[50, 8]]); assert.deepEqual(L.lines[0], [[50, 8], [50.1, 8.1]]);
  map.layers.raw_gps.items[0].click(); assert.equal(selected.observation_index, 0);
  map.toggle('road_network', false); assert.equal(L.removed.length, 1); map.toggle('road_network', true);
  assert.deepEqual(L.mapObject.bounds, [[50, 8]]);
});
test('completed test result keeps repeated and disconnected segments separate', async () => {
  const L = syntheticLeaflet(); const map = new RouteMap('test', () => {}, L);
  const a = line([8, 50], [8.1, 50.1], 0), b = line([12, 52], [12.1, 52.1], 4);
  await map.result({geometry: {classifier_route: fc([]), hmm_route: fc([a, b, a])}});
  assert.equal(L.lines.length, 3); assert.equal(map.layers.hmm_route.items.length, 3);
  assert.deepEqual(L.lines[0], L.lines[2]); assert.equal(L.lines[1].length, 2);
  map.clearResults(); assert.equal(map.layers.hmm_route.items.length, 0);
});
