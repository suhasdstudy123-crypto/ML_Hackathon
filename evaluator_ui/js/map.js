/** Central coordinate utility. GeoJSON stays [longitude, latitude] until this boundary. */
export function toLeaflet(coordinates) {
  const [lon, lat] = coordinates;
  if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lon) > 180 || Math.abs(lat) > 90) throw new Error('Invalid GeoJSON coordinates');
  return [lat, lon];
}
export const colors = {raw_gps: '#3183f6', road_network: '#8795a7', classifier_route: '#e5a526', hmm_route: '#21a878'};
export class RouteMap {
  constructor(id, inspector, leaflet = globalThis.L) {
    if (!leaflet) throw new Error('Map library unavailable. Rebuild the dashboard (npm run build) so vendor/leaflet is included, then reload.');
    this.L = leaflet; this.inspector = inspector;
    /** @type {(() => void) | null} */ this.onTileError = null;
    this.map = leaflet.map(id, {preferCanvas: true, zoomControl: false}).setView([50.1, 8.65], 12);
    leaflet.control.zoom({position: 'topright'}).addTo(this.map);
    const tiles = leaflet.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(this.map);
    tiles.on('tileerror', () => this.onTileError?.());
    this.layers = {}; this.visibility = {}; this.token = 0;
    Object.keys(colors).forEach(key => {this.layers[key] = leaflet.layerGroup().addTo(this.map); this.visibility[key] = true;});
  }
  clearResults() {this.layers.classifier_route.clearLayers(); this.layers.hmm_route.clearLayers();}
  clear() {this.bounds = []; this.token++; Object.values(this.layers).forEach(layer => layer.clearLayers());}
  toggle(key, enabled) {
    this.visibility[key] = enabled;
    if (enabled) this.layers[key].addTo(this.map); else this.map.removeLayer(this.layers[key]);
  }
  async draw(key, collection, token) {
    const layer = this.layers[key]; layer.clearLayers();
    const features = collection.features;
    for (let start = 0; start < features.length; start += 400) {
      if (token !== this.token) return;
      for (const feature of features.slice(start, start + 400)) {
        if (feature.geometry.type === 'Point') {
          const marker = this.L.circleMarker(toLeaflet(feature.geometry.coordinates), {radius: 3, color: colors.raw_gps, weight: 1, fillOpacity: .7});
          marker.on('click', () => this.inspector(feature.properties)); marker.addTo(layer);
        } else {
          // Each individual segment is drawn separately, including repeated arcs. Never connect unrelated endpoints.
          const line = this.L.polyline(feature.geometry.coordinates.map(toLeaflet), {color: colors[key], weight: key === 'road_network' ? 1.2 : 4, opacity: key === 'road_network' ? .38 : .85, dashArray: key === 'classifier_route' ? '6 5' : undefined});
          if (key !== 'road_network') line.bindTooltip(`Predicted arc ${feature.properties.arc_id} · observation ${feature.properties.sequence_index}`);
          line.addTo(layer);
        }
      }
      if (start + 400 < features.length) await new Promise(resolve => requestAnimationFrame(resolve));
    }
  }
  async preview(data) {
    this.clear(); const token = this.token;
    await this.draw('road_network', data.geometry.road_network, token);
    await this.draw('raw_gps', data.geometry.raw_gps, token);
    if (token !== this.token) return;
    this.bounds = data.geometry.raw_gps.features.map(f => toLeaflet(f.geometry.coordinates)); this.fit();
  }
  async result(data) {
    const token = this.token;
    await this.draw('classifier_route', data.geometry.classifier_route, token);
    await this.draw('hmm_route', data.geometry.hmm_route, token);
    if (token !== this.token) return;
    const lines = [...data.geometry.classifier_route.features, ...data.geometry.hmm_route.features];
    this.bounds = lines.flatMap(f => f.geometry.coordinates.map(toLeaflet)); this.fit();
  }
  fit() {if (this.bounds?.length) this.map.fitBounds(this.bounds, {padding: [36, 36], maxZoom: 17});}
  resize() {this.map.invalidateSize(); this.fit();}
}
