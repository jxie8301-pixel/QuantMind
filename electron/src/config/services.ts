/**
 * 统一服务端口配置
 * 所有服务端口的唯一配置来源
 */
export const SERVICE_PORTS = {
  // 前端服务
  FRONTEND_DEV: 3000,

  // 后端服务 (统一通过网关 8000)
  API_GATEWAY: 8000,
  MARKET_DATA: 8000,    // 原 8002
  DATA_SERVICE: 8000,   // 原 8002
  USER_SERVICE: 8000,   // 原 8011
  AI_STRATEGY: 8000,    // 原 8007
  STOCK_QUERY: 8000,    // 原 8010
  TRADING: 8000,        // 原 8004
  QLIB_SERVICE: 8000, // Qlib快速回测服务（收敛至网关）

  // WebSocket服务
  WEBSOCKET_MARKET: 8003,

  // 数据库
  REDIS: 6379,
} as const;

const ENV: Record<string, any> = typeof import.meta !== 'undefined' ? (import.meta as any).env || {} : {};
const HOST = ENV.VITE_SERVICE_HOST || '';
const HTTP_PROTOCOL = ENV.VITE_HTTP_PROTOCOL || 'http';
const WS_PROTOCOL = HTTP_PROTOCOL === 'https' ? 'wss' : 'ws';

export function normalizeBaseUrl(url: string): string {
  if (!url) return url;
  let normalized = url.replace(/\/+$/, '');
  if (normalized.endsWith('/api/v1')) {
    normalized = normalized.slice(0, -'/api/v1'.length);
  }
  return normalized;
}

const API_BASE = normalizeBaseUrl(ENV.VITE_API_BASE_URL || '');

export const SERVICE_URLS = {
  API_GATEWAY: normalizeBaseUrl(ENV.VITE_API_GATEWAY_URL) || API_BASE,
  MARKET_DATA: normalizeBaseUrl(ENV.VITE_MARKET_DATA_API_URL) || API_BASE,
  DATA_SERVICE: normalizeBaseUrl(ENV.VITE_DATA_SERVICE_API_URL) || API_BASE,
  USER_SERVICE: normalizeBaseUrl(ENV.VITE_USER_API_URL) || API_BASE,
  AI_STRATEGY: normalizeBaseUrl(ENV.VITE_AI_STRATEGY_API_URL) || API_BASE,
  STOCK_QUERY: normalizeBaseUrl(ENV.VITE_STOCK_QUERY_API_URL) || API_BASE,
  TRADING: normalizeBaseUrl(ENV.VITE_TRADING_API_URL) || API_BASE,
  QLIB_SERVICE: normalizeBaseUrl(ENV.VITE_QLIB_SERVICE_URL) || API_BASE,
  WEBSOCKET_MARKET: ENV.VITE_WS_BASE_URL || ENV.VITE_WEBSOCKET_MARKET_URL || `${WS_PROTOCOL}://${HOST}:${SERVICE_PORTS.WEBSOCKET_MARKET}/api/v1/ws/market`,
} as const;

// API路径配置
export const API_PATHS = {
  V1: '/api/v1',
  HEALTH: '/health',
  STRATEGIES: '/strategies',
  MARKET_DATA: '/market-data',
  USER: '/user',
  FILES: '/files',
} as const;

// 完整的服务端点配置
export const SERVICE_ENDPOINTS = {
  API_GATEWAY: `${SERVICE_URLS.API_GATEWAY}${API_PATHS.V1}`,
  AI_STRATEGY: `${SERVICE_URLS.AI_STRATEGY}${API_PATHS.V1}`,
  DATA_SERVICE: `${SERVICE_URLS.DATA_SERVICE}${API_PATHS.V1}`,
  USER_SERVICE: `${SERVICE_URLS.USER_SERVICE}${API_PATHS.V1}`,
  QLIB_SERVICE: `${SERVICE_URLS.QLIB_SERVICE}${API_PATHS.V1}`,
  STOCK_QUERY: `${SERVICE_URLS.STOCK_QUERY}${API_PATHS.V1}`,
  TRADING: `${SERVICE_URLS.TRADING}${API_PATHS.V1}`,
} as const;

export default {
  PORTS: SERVICE_PORTS,
  URLS: SERVICE_URLS,
  PATHS: API_PATHS,
  ENDPOINTS: SERVICE_ENDPOINTS,
};
