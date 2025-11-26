# Bug Report: Intermittent Gradio UI Loading (Hydration/Timeout)

## 1. Symptoms
- **Intermittent Loading**: The UI sometimes fails to load, showing a blank screen or a "Connection Error" toast.
- **Refresh Required**: Users often have to hard refresh the page (Ctrl+Shift+R) multiple times to get the UI to appear.
- **Mobile vs. Desktop**: The issue appears to be more prevalent or noticeable on Desktop Web than on Mobile Web (possibly due to network conditions, caching, or layout differences).
- **Environment**: HuggingFace Spaces (Docker SDK).

## 2. Root Cause Analysis

Based on research into Gradio 5.x/6.x behavior on HuggingFace Spaces, this is likely due to a combination of:

### A. SSR (Server-Side Rendering) Hydration Mismatch
Gradio 5+ introduced Server-Side Rendering (SSR) to improve initial load performance. However, on HuggingFace Spaces (which uses an iframe), there can be race conditions where the server-rendered HTML doesn't match what the client-side JavaScript expects, causing a "Hydration Error". When this happens, the React/Svelte frontend crashes silently or enters an inconsistent state, requiring a full refresh.

### B. WebSocket Timeouts
HuggingFace Spaces enforces strict timeouts for WebSocket connections. If the app takes too long to initialize (e.g., loading heavy libraries or models), the initial handshake may fail.
- *Mitigation*: Our app is relatively lightweight on startup (lazy loading models), so this is secondary, but network latency can trigger it.

### C. Browser Caching
Aggressive browser caching of the main bundle can sometimes cause version mismatches if the Space was recently rebuilt/redeployed.

## 3. Proposed Solution

### Immediate Fix: Disable SSR
Forcing Client-Side Rendering (CSR) eliminates the hydration mismatch entirely. While this theoretically slightly slows down the "First Contentful Paint", it is much more robust for dynamic apps inside iframes.

**Change in `src/app.py`:**
```python
demo.launch(
    # ... other args ...
    ssr_mode=False,  # Force Client-Side Rendering to fix hydration issues
)
```

### Secondary Fixes (If needed)
- **Increase Concurrency Limits**: Ensure `max_threads` is sufficient if many users connect at once.
- **Health Check**: Add a simple lightweight endpoint to keep the Space "warm" if it sleeps aggressively.

## 4. Verification Plan
1. Apply `ssr_mode=False` to `src/app.py`.
2. Deploy to HuggingFace Spaces (`fix/gradio-ui-final` branch).
3. Test on Desktop (Chrome Incognito, Firefox) and Mobile.
4. Verify no "Connection Error" toasts appear on initial load.
