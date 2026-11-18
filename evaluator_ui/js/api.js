/** @typedef {import('./types').Health} Health */
/** @typedef {import('./types').TripMetadata} TripMetadata */
/** @typedef {import('./types').Preview} Preview */
/** @typedef {import('./types').Job} Job */
/** @typedef {import('./types').Result} Result */
export class ApiError extends Error {
  constructor(message, code = 'NETWORK_ERROR', status = 0) {super(message); this.code = code; this.status = status;}
}
export function normalizeBase(value) {
  let url;
  try {url = new URL(value.trim());} catch {throw new ApiError('Enter a full backend URL, for example http://192.168.1.20:8000', 'INVALID_API_URL');}
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)) {
    throw new ApiError('Use an HTTP or HTTPS backend origin without credentials or a path', 'INVALID_API_URL');
  }
  if (globalThis.location?.protocol === 'https:' && url.protocol === 'http:') throw new ApiError('This HTTPS page needs an HTTPS backend, or serve the dashboard over HTTP on your LAN', 'MIXED_CONTENT');
  return url.origin;
}
export class ApiClient {
  /** @param {string} base @param {typeof fetch} transport */
  constructor(base, transport = globalThis.fetch) {this.base = normalizeBase(base); this.transport = transport.bind(globalThis);}
  /** @param {string} path @param {RequestInit} options @returns {Promise<any>} */
  async request(path, options = {}) {
    const timeout = AbortSignal.timeout(20000);
    const signal = options.signal ? AbortSignal.any([timeout, options.signal]) : timeout;
    let response;
    try {response = await this.transport(this.base + path, {...options, signal});}
    catch (err) {
      if (options.signal?.aborted) throw err;
      throw new ApiError('Backend unreachable or request timed out. Check its LAN address, server, firewall and trusted frontend origin.');
    }
    let body;
    try {body = await response.json();} catch {throw new ApiError('Backend returned an unreadable response', 'INVALID_RESPONSE', response.status);}
    if (!response.ok) throw new ApiError(body.error?.message || 'Backend request failed', body.error?.code || 'HTTP_ERROR', response.status);
    const items = Array.isArray(body) ? body : [body];
    if (items.some(item => item.schema_version !== '1.0')) throw new ApiError('Unsupported backend response version', 'UNSUPPORTED_SCHEMA');
    return body;
  }
  /** @returns {Promise<Health>} */ health() {return this.request('/api/v1/health/ready');}
  /** @returns {Promise<{schema_version: '1.0', trips: string[]}>} */ trips() {return this.request('/api/v1/trips');}
  /** @returns {Promise<TripMetadata>} */ metadata(id, signal) {return this.request(`/api/v1/trips/${encodeURIComponent(id)}`, {signal});}
  /** @returns {Promise<Preview>} */ preview(id, signal) {return this.request(`/api/v1/trips/${encodeURIComponent(id)}/preview`, {signal});}
  /** @returns {Promise<Job>} */ start(id, useHmm = true) {return this.request('/api/v1/reconstructions', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({trip_id: id, use_hmm: useHmm})});}
  /** @returns {Promise<Job>} */ job(id) {return this.request(`/api/v1/reconstructions/${encodeURIComponent(id)}`);}
  /** @returns {Promise<Result>} */ result(id) {return this.request(`/api/v1/reconstructions/${encodeURIComponent(id)}/result`);}
  /** @returns {Promise<Job[]>} */ jobs() {return this.request('/api/v1/reconstructions');}
}
