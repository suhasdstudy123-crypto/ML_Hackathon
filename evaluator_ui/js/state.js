export function reconstructionState(health, metadata, busy = false) {
  if (busy) return {enabled: false, message: 'Reconstruction in progress'};
  if (!health) return {enabled: false, message: 'Connect to your backend to explore trips'};
  if (!health.model_ready) return {enabled: false, message: 'LightGBM model not connected'};
  if (!metadata) return {enabled: false, message: 'Select and preview a trip'};
  if (!metadata.features?.ready) return {enabled: false, message: metadata.features?.reason || 'Authoritative features unavailable for this trip'};
  return {enabled: true, message: 'Ready to reconstruct this trip'};
}
export function formatFraction(value) {return value == null ? '—' : `${(value * 100).toFixed(1)}%`;}
export function formatDuration(value) {return value == null ? '—' : `${value.toFixed(0)} ms`;}
export const stageLabel = value => ({QUEUED: 'Queued', PREPARING_DATA: 'Loading trip data', PREPARING_FEATURES: 'Validating candidate features', GENERATING_CANDIDATES: 'Generating notebook candidates', SCORING_CANDIDATES: 'Scoring with LightGBM', DECODING_ROUTE: 'Decoding directed route', EVALUATING: 'Evaluating on the server', COMPLETED: 'Reconstruction complete', FAILED: 'Reconstruction failed'})[value] || value;
/** Bounded polling; retry network errors only, never repeat the heavy POST. */
/** @param {any} api @param {string} id @param {(job: import('./types').Job) => void} onStage
 * @param {{isCancelled?: () => boolean, wait?: (ms: number) => Promise<any>, maxPolls?: number}} options */
export async function pollJob(api, id, onStage, {isCancelled = () => false, wait = ms => new Promise(r => setTimeout(r, ms)), maxPolls = 300} = {}) {
  let failures = 0;
  for (let n = 0; n < maxPolls; n++) {
    if (isCancelled()) return null;
    let job;
    try {job = await api.job(id); failures = 0;}
    catch (err) {
      if (err.status >= 400 || ++failures > 3) throw err;
      await wait(1000 * 2 ** (failures - 1)); continue;
    }
    if (isCancelled()) return null;
    onStage(job);
    if (job.status === 'FAILED') throw new Error(job.error?.message || 'Reconstruction failed');
    if (job.status === 'COMPLETED') return api.result(id);
    await wait(1000);
  }
  throw new Error('Stopped checking after five minutes. Backend work continues; reconnect to resume checking this job.');
}
