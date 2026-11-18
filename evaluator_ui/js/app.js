import {ApiClient} from './api.js';
import {RouteMap} from './map.js';
import {reconstructionState, formatFraction, formatDuration, stageLabel, pollJob} from './state.js';
/** @param {string} id @returns {any} */
const el = id => document.getElementById(id);
const stored = key => {try {return localStorage.getItem(key);} catch {return null;}};
const store = (key, value) => {try {if (value == null) localStorage.removeItem(key); else localStorage.setItem(key, value);} catch {}};
/** @type {import('./api.js').ApiClient | null} */ let api = null;
/** @type {import('./types').Health | null} */ let health = null;
/** @type {import('./types').TripMetadata | null} */ let metadata = null;
let mapError = null;
let routeMap, busy = false, loading = false, epoch = 0, previewAbort, currentJob = null, checking = false;
const names = {overview: ['Fleet Overview', 'Every trip, in context.', 'Explore the road beneath your GPS observations.'], explorer: ['Trip Explorer', 'Follow the observations.', 'Inspect real GPS samples and the original directed road network.'], studio: ['Route Studio', 'From points to a path.', 'Compare classifier predictions with a directed HMM reconstruction.']};
function navigate() {
  const page = location.hash.slice(1) in names ? location.hash.slice(1) : 'overview';
  document.body.dataset.page = page;
  document.querySelectorAll('[data-page]').forEach(a => {a.classList.toggle('active', a.getAttribute('data-page') === page); if (a.getAttribute('data-page') === page) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');});
  [el('breadcrumb').textContent, el('page-title').textContent, el('page-description').textContent] = names[page];
  requestAnimationFrame(() => routeMap?.resize());
}
function showError(error) {el('error').hidden = false; el('error-message').textContent = error.message || String(error);}
function clearError() {el('error').hidden = true;}
function resetResults() {
  routeMap?.clearResults();
  for (const id of ['metric-acc', 'metric-cont', 'metric-jacc', 'metric-lat']) el(id).textContent = '—';
  el('progress').textContent = 'Metrics appear after a real reconstruction.'; el('result-state').textContent = 'AWAITING INFERENCE';
  document.querySelectorAll('[data-layer$="route"]').forEach(node => {/** @type {HTMLInputElement} */(node).disabled = true;});
}
function updateAvailability() {
  const state = loading ? {enabled: false, message: 'Loading trip preview…'} : reconstructionState(health, metadata, busy);
  el('run-btn').disabled = !state.enabled; el('pending-note').textContent = state.message; el('use-hmm').disabled = busy;
}
function markConnected(value) {if (!value) el('model-chip').textContent = '○ Backend offline'; el('status-dot').style.background = value ? '#32bf8f' : '#e2856a'; el('status-text').textContent = value ? 'Backend connected' : 'Backend disconnected';}
function updateHealth(value) {
  health = value; markConnected(true); el('model-chip').textContent = value.model_ready ? '● LightGBM connected' : '○ Model pending';
  el('trip-count').textContent = value.trip_count.toLocaleString(); updateAvailability();
}
async function refreshJobs(client = api) {
  if (!client) return;
  const jobs = await client.jobs(); if (client !== api) return;
  el('completed-count').textContent = jobs.filter(j => j.status === 'COMPLETED').length.toLocaleString(); el('recent-jobs').replaceChildren();
  if (!jobs.length) {el('recent-jobs').textContent = 'No reconstructions in this backend session yet.'; return;}
  for (const job of jobs.slice(-5).reverse()) {
    const row = document.createElement('div'); row.className = 'job-row';
    const label = document.createElement('span'); label.textContent = `Trip ${job.trip_id} · ${stageLabel(job.status)}`; row.append(label);
    if (job.status === 'COMPLETED') {
      const button = document.createElement('button'); button.textContent = 'View result →';
      button.addEventListener('click', async () => {
        if (busy) return;
        try {const resultEpoch = await selectTrip(job.trip_id); if (resultEpoch !== epoch) return;
          const result = await client.result(job.job_id); if (resultEpoch === epoch) {await renderResult(result); location.hash = 'studio';}
        } catch (error) {showError(error);}
      }); row.append(button);
    }
    el('recent-jobs').append(row);
  }
}
async function connect() {
  const token = ++epoch; previewAbort?.abort(); busy = false; metadata = null; health = null; api = null; currentJob = null;
  routeMap?.clear(); resetResults(); clearError(); updateAvailability(); markConnected(false);
  for (const id of ['trip-count','observation-count','arc-count','completed-count','trip-duration','trip-gaps']) el(id).textContent = '—';
  el('warnings').textContent = ''; el('map-title').textContent = 'Trip workspace'; el('map-subtitle').textContent = 'Select a trip to preview';
  el('recent-jobs').textContent = 'Connect to view actual backend jobs.';
  el('map-empty').hidden = false; el('trip-select').disabled = true; el('connect-btn').disabled = true;
  el('connect-btn').textContent = 'Connecting…'; el('loading').hidden = true;
  try {
    const client = new ApiClient(el('api-url').value);
    const [readiness, trips] = await Promise.all([client.health(), client.trips()]); if (token !== epoch) return;
    api = client; store('pathsnap-api', client.base); updateHealth(readiness);
    const select = el('trip-select'); select.replaceChildren();
    for (const tid of trips.trips) {const option = document.createElement('option'); option.value = tid; option.textContent = `Trip ${tid}`; select.append(option);}
    select.disabled = trips.trips.length === 0;
    if (!trips.trips.length) {showError(new Error('No permitted trips available on the backend')); return;}
    await selectTrip(trips.trips[0]);
    try {await refreshJobs(client);} catch (error) {showError(error);}
    const saved = stored('pathsnap-job');
    if (saved && client === api) {
      try {const job = JSON.parse(saved); if (job.base === client.base && trips.trips.includes(job.trip_id)) {await selectTrip(job.trip_id); await resumeJob(job.job_id);}}
      catch (error) {store('pathsnap-job', null); showError(error);}
    }
  } catch (error) {if (token === epoch || api == null) {health = null; markConnected(false); showError(error);}}
  finally {el('connect-btn').disabled = false; el('connect-btn').textContent = 'Connect backend ↗'; updateAvailability();}
}
async function selectTrip(id) {
  const token = ++epoch, client = api; previewAbort?.abort(); previewAbort = new AbortController();
  busy = false; loading = true; currentJob = null; metadata = null; el('trip-select').value = id;
  resetResults(); clearError(); updateAvailability(); el('loading').hidden = false; el('loading').textContent = 'Loading trip preview…';
  el('warnings').textContent = ''; routeMap?.clear(); el('map-empty').hidden = true;
  el('observation-count').textContent = '—'; el('arc-count').textContent = '—';
  el('inspector').textContent = 'Click a blue GPS observation on the map to inspect its index and timestamp.';
  try {
    const [preview, info] = await Promise.all([client.preview(id, previewAbort.signal), client.metadata(id, previewAbort.signal)]);
    if (token !== epoch) return token;
    if (preview.trip_id !== id || info.trip_id !== id) throw new Error('Backend returned a different trip');
    metadata = info; el('map-title').textContent = `Trip ${id}`; el('map-subtitle').textContent = `${info.observation_count.toLocaleString()} GPS observations`;
    el('observation-count').textContent = info.observation_count.toLocaleString(); el('arc-count').textContent = info.arc_count.toLocaleString();
    el('trip-duration').textContent = `${(info.duration_seconds / 60).toFixed(1)} min`; el('trip-gaps').textContent = info.large_gaps;
    el('warnings').textContent = preview.warnings.join(' '); if (routeMap) await routeMap.preview(preview); else if (mapError) showError(mapError);
  } catch (error) {if (token === epoch && error.name !== 'AbortError') {metadata = null; showError(error); el('map-empty').hidden = false;}}
  finally {if (token === epoch) {loading = false; el('loading').hidden = true; updateAvailability();}}
  return token;
}
async function renderResult(result) {
  if (result.schema_version !== '1.0' || result.trip_id !== metadata?.trip_id) throw new Error('Result schema or trip does not match this workspace');
  el('metric-acc').textContent = formatFraction(result.metrics.top1_accuracy); el('metric-cont').textContent = formatFraction(result.metrics.route_continuity);
  el('metric-jacc').textContent = formatFraction(result.metrics.sequence_jaccard); el('metric-lat').textContent = formatDuration(result.metrics.inference_time_ms);
  el('progress').textContent = `Reconstructed ${result.observations.count.toLocaleString()} observations · ${result.diagnostics.transition_breaks.length} run breaks`;
  el('result-state').textContent = 'COMPLETED'; el('warnings').textContent = result.warnings.join(' ');
  for (const key of ['classifier_route', 'hmm_route']) {const checkbox = /** @type {HTMLInputElement} */(document.querySelector(`[data-layer="${key}"]`)); checkbox.disabled = !result.geometry[key].features.length;}
  await routeMap?.result(result);
}
async function resumeJob(id) {
  const token = epoch, client = api; busy = true; currentJob = id; updateAvailability();
  try {
    const result = await pollJob(client, id, job => {el('progress').textContent = stageLabel(job.status); el('result-state').textContent = job.status.replaceAll('_', ' ');}, {isCancelled: () => token !== epoch || currentJob !== id});
    if (result && token === epoch) {await renderResult(result); store('pathsnap-job', null); await refreshJobs(client);}
  } catch (error) {
    if (token === epoch) {if (error.status === 404) store('pathsnap-job', null); showError(error); el('progress').textContent = error.message;}
  } finally {if (token === epoch) {busy = false; updateAvailability();}}
}
async function run() {
  if (!reconstructionState(health, metadata, busy || loading).enabled) return;
  const token = epoch, client = api; busy = true; clearError(); resetResults(); updateAvailability();
  try {
    const job = await client.start(metadata.trip_id, el('use-hmm').checked);
    store('pathsnap-job', JSON.stringify({base: client.base, trip_id: job.trip_id, job_id: job.job_id})); if (token !== epoch) return;
    location.hash = 'studio'; await resumeJob(job.job_id);
  } catch (error) {if (token === epoch) {showError(error); if (error.code === 'MODEL_NOT_READY') {health.model_ready = false; updateHealth(health);}}}
  finally {if (token === epoch) {busy = false; updateAvailability();}}
}
async function initialize() {
  navigate();
  try {routeMap = new RouteMap('map', properties => {el('inspector').textContent = `Observation #${properties.observation_index} · timestamp ${properties.timestamp} s`;});
    routeMap.onTileError = () => {el('warnings').textContent = 'Background tiles unavailable. GPS and road geometry remain available; check internet access.';};
  } catch (error) {mapError = error; showError(error);}
  el('api-url').value = new URLSearchParams(location.search).get('api') || stored('pathsnap-api') || '';
  el('connect-btn').addEventListener('click', connect); el('retry-btn').addEventListener('click', connect);
  el('api-url').addEventListener('keydown', event => {if (event.key === 'Enter') connect();});
  el('trip-select').addEventListener('change', event => selectTrip(event.target.value));
  el('run-btn').addEventListener('click', run); el('fit-btn').addEventListener('click', () => routeMap?.fit());
  document.querySelectorAll('[data-layer]').forEach(node => node.addEventListener('change', event => {const target = /** @type {HTMLInputElement} */(event.target); routeMap?.toggle(target.dataset.layer, target.checked);}));
  addEventListener('hashchange', navigate); addEventListener('resize', () => routeMap?.resize());
  setInterval(async () => {
    if (!api || busy || loading || checking) return;
    checking = true; const client = api;
    try {const value = await client.health(); if (client === api) updateHealth(value);}
    catch {if (client === api) {health = null; markConnected(false); updateAvailability();}}
    finally {checking = false;}
  }, 15000);
  if (el('api-url').value) await connect();
}
initialize();
