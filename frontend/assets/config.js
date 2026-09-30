/**
 * Runtime Deployment Configuration for UAV Digital Twin Frontend
 * 
 * Local Development:
 *   Leave values empty or set to "http://localhost:8000".
 *   Defaults automatically to same-origin / relative paths.
 * 
 * Production Deployment (Netlify -> Render):
 *   Set BACKEND_API_URL to your Render web service URL:
 *     window.BACKEND_API_URL = "https://your-backend.onrender.com";
 *     window.BACKEND_WS_URL  = "wss://your-backend.onrender.com";
 * 
 * URL Override:
 *   You can also test immediately by opening your Netlify URL with a query parameter:
 *     https://your-site.netlify.app/?backend=https://your-backend.onrender.com
 */
window.BACKEND_API_URL = window.BACKEND_API_URL || "";
window.BACKEND_WS_URL = window.BACKEND_WS_URL || "";

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
