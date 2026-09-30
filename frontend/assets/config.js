/**
 * Runtime Deployment Configuration for UAV Digital Twin Frontend
 * 
 * Auto-detects local vs. cloud environment:
 * - Localhost / 127.0.0.1: Connects directly to local backend (same-origin / relative paths)
 * - Netlify / Cloud: Automatically routes to deployed Render backend
 * 
 * URL Override:
 *   Append ?backend=https://your-custom-backend.onrender.com to override at runtime.
 */

const isLocalhost = 
  window.location.hostname === "localhost" || 
  window.location.hostname === "127.0.0.1" || 
  window.location.hostname === "";

const PRODUCTION_RENDER_BACKEND = "https://uav-digital-twin-vzgs.onrender.com";
const PRODUCTION_RENDER_WS = "wss://uav-digital-twin-vzgs.onrender.com";

window.BACKEND_API_URL = window.BACKEND_API_URL || (isLocalhost ? "" : PRODUCTION_RENDER_BACKEND);
window.BACKEND_WS_URL = window.BACKEND_WS_URL || (isLocalhost ? "" : PRODUCTION_RENDER_WS);

try {
  const params = new URLSearchParams(window.location.search);
  const backendParam = params.get("backend");
  if (backendParam) {
    window.BACKEND_API_URL = backendParam.replace(/\/+$/, "");
    const wsProto = window.BACKEND_API_URL.startsWith("https") ? "wss://" : "ws://";
    const host = window.BACKEND_API_URL.replace(/^https?:\/\//, "");
    window.BACKEND_WS_URL = `${wsProto}${host}`;
  }
} catch (e) {
  /* ignore environment where URLSearchParams is unavailable */
}
