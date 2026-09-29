/*
 * PHASE 19 — Control-room dashboard client.
 *
 * Responsibilities:
 *   - render the new TOP/LEFT/CENTER/RIGHT/BOTTOM layout
 *   - drive the 8 chart canvases (RPM, EGT, CHT, OilP, Fuel,
 *     Vib+IMU, Health, RUL range)
 *   - keep an in-memory "active alerts" set, keyed by fault_type
 *     (deduplicated banner feed)
 *   - transport: WebSocket primary, REST polling fallback
 *   - scenario selector
 *
 * No aircraft-control commands are ever issued. The dashboard
 * surfaces advisory text only.
 */

(() => {
  "use strict";

  // -------- Status colour map (driven by enums, not magic numbers) ------
  const COLORS = {
    GO: "var(--green)",
    HEALTHY: "var(--green)",
    RUL_OK: "var(--green)",
    NORMAL: "var(--green)",
    CAUTION: "var(--amber)",
    DEGRADED: "var(--amber)",
    RUL_DEGRADED: "var(--amber)",
    WARN: "var(--amber)",
    RETURN_TO_BASE: "var(--orange)",
    RUL_CRITICAL: "var(--orange)",
    HIGH: "var(--orange)",
    ABORT: "var(--red)",
    CRITICAL: "var(--red)",
    SEVERE: "var(--red)",
    INSUFFICIENT_DATA: "var(--grey)",
    RUL_UNCERTAIN: "var(--grey)",
    MODEL_NOT_CALIBRATED: "var(--grey)",
  };
  const color = (token) => COLORS[token] || "var(--grey)";

  // -------- Helpers ----------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const fmt = (n, digits = 2) => {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits);
  };
  const safe = (v, digits = 2) =>
    v === null || v === undefined ? "—" : Number(v).toFixed(digits);
  const setChip = (id, token, fallbackText) => {
    const el = $(id);
    if (!el) return;
    el.textContent = token || fallbackText || "—";
    el.style.background = color(token);
  };
  const scoreToColor = (score) => {
    if (score >= 0.8) return "var(--red)";
    if (score >= 0.55) return "var(--orange)";
    if (score >= 0.25) return "var(--amber)";
    return "var(--green)";
  };

  // ===========================================================
  // PHASE 24 — Individual warning popup system + alert center.
  // ===========================================================
  //
  // The backend emits the *current* set of active alerts every
  // tick. The popup system is responsible for:
  //
  //   * **Dedup** — an alert with the same `alert_id` across
  //     ticks is the *same* condition; we update the existing
  //     popup / alert-center row in place. No new popup, no
  //     new sound.
  //   * **New alert** — an `alert_id` we've never seen
  //     produces a new popup. The popup stack is capped at
  //     3 visible; older popups collapse into the alert
  //     center.
  //   * **Escalation** — if a known alert's `tier` rises
  //     (e.g. WARNING → CRITICAL), we surface a *new*
  //     escalation event so the operator sees the change.
  //   * **Resolution** — when a previously-active alert no
  //     longer appears, we mark it "resolved" and keep it
  //     in history. Popups for resolved alerts are removed.
  //   * **Acknowledgement** — clicking ACKNOWLEDGE marks
  //     the alert as `acked: true`. The popup dims. The
  //     underlying condition is unchanged — if it is still
  //     active, the popup continues to show its values.
  //
  // The popup manager is purely client-side state. The
  // backend does not retain acknowledge state across
  // reconnects (a fresh WebSocket resets it), which matches
  // the existing PHASE 19 behaviour for the alert list.

  const MAX_VISIBLE_POPUPS = 3;

  // alertId -> {
  //   data: <last-known alert dict>,
  //   acked: <bool>,
  //   firstSeenAt: <sim-time>,
  //   lastSeenAt: <sim-time>,
  //   resolvedAt: <sim-time or null>,
  //   element: <HTMLElement or null>,     // the popup, if visible
  //   escalated: <bool>,                  // true if an escalation popup fired
  //   preEscalationTier: <int or null>,
  // }
  const alertState = new Map();
  // History of all alerts (capped to keep memory bounded).
  const alertHistory = [];
  const MAX_HISTORY = 200;

  // Currently-shown popup alertIds (oldest at index 0).
  // Bounded by MAX_VISIBLE_POPUPS; the rest live in the
  // alert center.
  const visiblePopupIds = [];

  // Currently-selected tab in the alert center.
  let acTab = "active";

  // Tones for the optional audio cue. Played on CRITICAL and
  // WARNING (once per new / escalated alert). The browser
  // may block autoplay — that is fine, the visual cue is
  // always present.
  let audioCtx = null;
  let audioUnlocked = false;
  function unlockAudio() {
    if (audioUnlocked) return;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      audioCtx = new Ctx();
      audioUnlocked = true;
    } catch (e) {
      /* ignore */
    }
  }
  function playTone(tier) {
    if (!audioUnlocked || !audioCtx) return;
    try {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.type = "sine";
      // CRITICAL = higher pitch, double-beep. WARNING = single lower.
      o.frequency.value = tier === 2 ? 880 : tier === 1 ? 660 : 440;
      g.gain.value = 0;
      o.connect(g);
      g.connect(audioCtx.destination);
      const now = audioCtx.currentTime;
      g.gain.setValueAtTime(0, now);
      g.gain.linearRampToValueAtTime(0.08, now + 0.02);
      g.gain.linearRampToValueAtTime(0, now + 0.18);
      if (tier === 2) {
        // second beep
        g.gain.setValueAtTime(0, now + 0.22);
        g.gain.linearRampToValueAtTime(0.08, now + 0.24);
        g.gain.linearRampToValueAtTime(0, now + 0.4);
      }
      o.start(now);
      o.stop(now + (tier === 2 ? 0.42 : 0.2));
    } catch (e) {
      /* ignore */
    }
  }

  function fmtTime(t) {
    if (t === null || t === undefined) return "—";
    return Number(t).toFixed(1) + " s";
  }
  function fmtClockLike(t) {
    if (t === null || t === undefined) return "—";
    // Use mission-relative seconds with a stable presentation.
    const total = Math.max(0, Math.floor(Number(t)));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return m.toString().padStart(2, "0") + ":" + s.toString().padStart(2, "0");
  }

  // -------- Popup rendering ---------------------------------------
  function ensurePopupEl(state) {
    if (state.element) return state.element;
    const el = document.createElement("div");
    el.className = "popup tier-" + (state.data.severity_label || "INFO");
    el.dataset.alertId = state.data.alert_id || "";
    el.innerHTML = `
      <div class="popup-header">
        <span class="popup-sev">${state.data.severity_label || "INFO"}</span>
        <div class="popup-title"></div>
        <div class="popup-time"></div>
      </div>
      <div class="popup-msg"></div>
      <div class="popup-stats"></div>
      <div class="popup-evidence"></div>
      <div class="popup-actions">
        <button class="btn-ack" type="button">ACKNOWLEDGE</button>
        <button class="btn-details" type="button">VIEW DETAILS</button>
      </div>`;
    el.querySelector(".btn-ack").addEventListener("click", () => {
      acknowledgeAlert(state.data.alert_id);
    });
    el.querySelector(".btn-details").addEventListener("click", () => {
      openDetail(state.data.alert_id);
    });
    state.element = el;
    return el;
  }

  function refreshPopupEl(state) {
    const el = ensurePopupEl(state);
    el.classList.toggle("acked", !!state.acked);
    el.classList.toggle(
      "tier-CRITICAL",
      (state.data.severity_label || "INFO") === "CRITICAL",
    );
    el.classList.toggle(
      "tier-WARNING",
      (state.data.severity_label || "INFO") === "WARNING",
    );
    el.classList.toggle(
      "tier-INFO",
      (state.data.severity_label || "INFO") === "INFO",
    );
    el.querySelector(".popup-title").textContent =
      state.data.title || state.data.fault_type || "—";
    el.querySelector(".popup-time").textContent = fmtClockLike(
      state.data.time_s,
    );
    el.querySelector(".popup-msg").textContent =
      state.data.message || state.data.evidence || "—";
    // stats
    const stats = el.querySelector(".popup-stats");
    stats.innerHTML = "";
    const s = state.data;
    const statRows = [
      [
        "Confidence",
        s.confidence !== undefined && s.confidence !== null
          ? (Number(s.confidence) * 100).toFixed(0) + " %"
          : "—",
      ],
    ];
    if (s.sensor) {
      const cur =
        s.current_value !== null && s.current_value !== undefined
          ? Number(s.current_value).toFixed(2)
          : "—";
      const exp =
        s.expected_value !== null && s.expected_value !== undefined
          ? Number(s.expected_value).toFixed(2)
          : "—";
      const dev =
        s.deviation !== null && s.deviation !== undefined
          ? (Number(s.deviation) >= 0 ? "+" : "") +
            Number(s.deviation).toFixed(2)
          : "—";
      statRows.push(["Sensor", s.sensor]);
      statRows.push(["Current", cur]);
      statRows.push(["Expected", exp]);
      statRows.push(["Deviation", dev]);
    }
    if (s.mission_phase) statRows.push(["Phase", s.mission_phase]);
    statRows.forEach(([k, v]) => {
      const kEl = document.createElement("div");
      kEl.className = "k";
      kEl.textContent = k;
      const vEl = document.createElement("div");
      vEl.className = "v";
      vEl.textContent = v;
      stats.appendChild(kEl);
      stats.appendChild(vEl);
    });
    // evidence
    const evEl = el.querySelector(".popup-evidence");
    evEl.innerHTML = "";
    const evidence = (s.evidence || "")
      .split(";")
      .map((x) => x.trim())
      .filter(Boolean);
    if (evidence.length === 0) {
      evEl.innerHTML = `<div class="ev-line">no extra evidence</div>`;
    } else {
      evidence.forEach((line) => {
        const d = document.createElement("div");
        d.className = "ev-line";
        d.textContent = line;
        evEl.appendChild(d);
      });
    }
    return el;
  }

  function showPopup(state) {
    const el = refreshPopupEl(state);
    if (!el.parentElement) {
      document.getElementById("popup-layer").appendChild(el);
    }
    if (!visiblePopupIds.includes(state.data.alert_id)) {
      visiblePopupIds.push(state.data.alert_id);
    }
  }
  function hidePopup(alertId) {
    const state = alertState.get(alertId);
    if (state && state.element && state.element.parentElement) {
      state.element.parentElement.removeChild(state.element);
    }
    const idx = visiblePopupIds.indexOf(alertId);
    if (idx >= 0) visiblePopupIds.splice(idx, 1);
  }

  // -------- Alert center rendering -------------------------------
  function renderAlertCenter() {
    const list = document.getElementById("alert-center-list");
    if (!list) return;
    const tab = acTab;
    const rows = [];
    alertState.forEach((state, id) => {
      const active = state.resolvedAt === null;
      if (tab === "active" && (!active || state.acked)) return;
      if (tab === "acked" && (!state.acked || !active)) return;
      if (tab === "history" && active && !state.acked) return;
      rows.push(state);
    });
    if (rows.length === 0) {
      list.innerHTML = `<div class="ac-empty">${
        tab === "active"
          ? "No active alerts."
          : tab === "acked"
            ? "No acknowledged alerts."
            : "No history yet."
      }</div>`;
      return;
    }
    rows.sort((a, b) => {
      // Most-recent first; CRITICAL before WARNING before INFO.
      const tA = a.resolvedAt !== null ? a.resolvedAt : a.lastSeenAt;
      const tB = b.resolvedAt !== null ? b.resolvedAt : b.lastSeenAt;
      if (tB !== tA) return tB - tA;
      return (b.data.tier || 0) - (a.data.tier || 0);
    });
    list.innerHTML = rows
      .map((state) => {
        const s = state.data;
        const tier = s.severity_label || "INFO";
        const title = s.title || s.fault_type || "—";
        const ts = fmtClockLike(
          state.resolvedAt !== null ? state.resolvedAt : s.time_s,
        );
        return `<div class="ac-row tier-${tier} ${state.acked ? "acked" : ""}" data-alert-id="${s.alert_id || ""}">
        <div class="dot"></div>
        <div class="sev">${tier}</div>
        <div class="title">${escapeHtml(title)}</div>
        <div class="ts">${ts}</div>
      </div>`;
      })
      .join("");
    list.querySelectorAll(".ac-row").forEach((row) => {
      row.addEventListener("click", () => {
        openDetail(row.dataset.alertId);
      });
    });
  }

  function escapeHtml(s) {
    return String(s).replace(
      /[&<>"]/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
        })[c],
    );
  }

  // -------- Detail modal -----------------------------------------
  let currentDetailId = null;
  function openDetail(alertId) {
    const state = alertState.get(alertId);
    if (!state) return;
    currentDetailId = alertId;
    const s = state.data;
    const modal = document.getElementById("detail-modal");
    const sev = document.getElementById("detail-modal-sev");
    const title = document.getElementById("detail-modal-title");
    const body = document.getElementById("detail-modal-body");
    const ackBtn = document.getElementById("detail-modal-ack");
    if (!modal) return;
    sev.textContent = s.severity_label || "INFO";
    sev.className = "popup-sev tier-" + (s.severity_label || "INFO");
    title.textContent = s.title || s.fault_type || "—";
    body.innerHTML = detailBodyHtml(s, state);
    ackBtn.textContent = state.acked ? "Acknowledged" : "Acknowledge";
    ackBtn.disabled = !!state.acked;
    modal.classList.add("open");
  }
  function closeDetail() {
    const modal = document.getElementById("detail-modal");
    if (modal) modal.classList.remove("open");
    currentDetailId = null;
  }
  function detailBodyHtml(s, state) {
    const rows = [];
    const row = (k, v) =>
      `<div class="row"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    rows.push(row("Fault / event", s.fault_type || "—"));
    rows.push(row("Severity", s.severity_label || "—"));
    rows.push(row("Active since", fmtTime(state.firstSeenAt)));
    rows.push(row("Last update", fmtTime(s.time_s)));
    rows.push(
      row(
        "Status",
        state.acked
          ? "ACKNOWLEDGED — condition active"
          : state.resolvedAt !== null
            ? "RESOLVED at " + fmtTime(state.resolvedAt)
            : "ACTIVE",
      ),
    );
    if (s.sensor) rows.push(row("Sensor", s.sensor));
    if (s.subsystem) rows.push(row("Subsystem", s.subsystem));
    if (s.current_value !== null && s.current_value !== undefined)
      rows.push(row("Current value", Number(s.current_value).toFixed(2)));
    if (s.expected_value !== null && s.expected_value !== undefined)
      rows.push(row("Expected value", Number(s.expected_value).toFixed(2)));
    if (s.deviation !== null && s.deviation !== undefined)
      rows.push(row("Deviation", Number(s.deviation).toFixed(2)));
    rows.push(
      row(
        "Confidence",
        s.confidence !== undefined && s.confidence !== null
          ? (Number(s.confidence) * 100).toFixed(0) + " %"
          : "—",
      ),
    );
    if (s.anomaly_score !== null && s.anomaly_score !== undefined)
      rows.push(row("Anomaly score", Number(s.anomaly_score).toFixed(2)));
    if (s.mission_phase) rows.push(row("Mission phase", s.mission_phase));
    if (s.altitude_m !== null && s.altitude_m !== undefined)
      rows.push(row("Altitude", Number(s.altitude_m).toFixed(0) + " m"));
    if (s.airspeed_mps !== null && s.airspeed_mps !== undefined)
      rows.push(row("Airspeed", Number(s.airspeed_mps).toFixed(1) + " m/s"));
    if (s.rpm !== null && s.rpm !== undefined)
      rows.push(row("RPM", Number(s.rpm).toFixed(0)));
    if (s.throttle !== null && s.throttle !== undefined)
      rows.push(row("Throttle", (Number(s.throttle) * 100).toFixed(0) + " %"));
    if (s.engine_load !== null && s.engine_load !== undefined)
      rows.push(
        row("Engine load", (Number(s.engine_load) * 100).toFixed(0) + " %"),
      );
    if (s.env_temp_c !== null && s.env_temp_c !== undefined)
      rows.push(row("Ambient T", Number(s.env_temp_c).toFixed(1) + " °C"));
    if (s.env_pressure_pa !== null && s.env_pressure_pa !== undefined)
      rows.push(row("Ambient P", Number(s.env_pressure_pa).toFixed(0) + " Pa"));
    if (s.turbulence_w_mps !== null && s.turbulence_w_mps !== undefined)
      rows.push(
        row("Turbulence", Number(s.turbulence_w_mps).toFixed(2) + " m/s"),
      );
    rows.push(
      row(
        "Gust active",
        s.gust_active === true ? "yes" : s.gust_active === false ? "no" : "—",
      ),
    );
    const evidence = (s.evidence || "")
      .split(";")
      .map((x) => x.trim())
      .filter(Boolean);
    const evidenceHtml =
      evidence.length === 0
        ? `<div class="ev-line">no extra evidence</div>`
        : evidence
            .map((line) => `<div class="ev-line">${escapeHtml(line)}</div>`)
            .join("");
    return `
      <div class="section">
        <h4>Identification</h4>
        ${rows.join("")}
      </div>
      <div class="section">
        <h4>Recommended action</h4>
        <div class="row"><div class="v" style="grid-column:2">${escapeHtml(s.recommendation || "—")}</div></div>
      </div>
      <div class="section">
        <h4>Evidence / contributors</h4>
        <div class="ev-list">${evidenceHtml}</div>
      </div>`;
  }

  // -------- Acknowledge ------------------------------------------
  function acknowledgeAlert(alertId) {
    const state = alertState.get(alertId);
    if (!state) return;
    state.acked = true;
    // Refresh visible popup.
    if (state.element) refreshPopupEl(state);
    // Refresh the alert center if it's open.
    if (document.getElementById("alert-center").classList.contains("open")) {
      renderAlertCenter();
    }
    // Update the detail modal if it's open on this alert.
    if (currentDetailId === alertId) {
      openDetail(alertId);
    }
  }

  // -------- Public update entry point ----------------------------
  function updateAlertSystem(currentAlerts) {
    // 1) Find the set of alert IDs that are *currently* active
    //    on the server.
    const incomingIds = new Set();
    (currentAlerts || []).forEach((a) => {
      if (!a || !a.alert_id) return; // skip malformed
      incomingIds.add(a.alert_id);
    });

    // 2) Update / create state entries for every incoming alert.
    (currentAlerts || []).forEach((a) => {
      if (!a || !a.alert_id) return;
      const id = a.alert_id;
      const tier = Number(a.tier || 0);
      const existing = alertState.get(id);
      if (!existing) {
        const state = {
          data: a,
          acked: false,
          firstSeenAt: Number(a.time_s || 0),
          lastSeenAt: Number(a.time_s || 0),
          resolvedAt: null,
          element: null,
          escalated: false,
          preEscalationTier: null,
        };
        alertState.set(id, state);
        // NEW alert — show a popup (capped at MAX_VISIBLE_POPUPS).
        if (visiblePopupIds.length < MAX_VISIBLE_POPUPS) {
          showPopup(state);
        }
        if (tier >= 1) playTone(tier);
      } else {
        // EXISTING alert. Refresh data, check for escalation.
        existing.data = a;
        existing.lastSeenAt = Number(a.time_s || 0);
        if (existing.resolvedAt !== null) existing.resolvedAt = null;
        const prevTier = Number(existing.data.tier || 0);
        if (tier > prevTier) {
          // ESCALATION — surface a fresh popup (if room) and a tone.
          if (!existing.escalated) {
            existing.preEscalationTier = prevTier;
            existing.escalated = true;
          }
          if (visiblePopupIds.includes(id)) {
            refreshPopupEl(existing);
          } else if (visiblePopupIds.length < MAX_VISIBLE_POPUPS) {
            showPopup(existing);
          } else {
            // No visible slot — bump the oldest non-critical out
            // to make room. We never bump a CRITICAL popup.
            const victim = visiblePopupIds.find(
              (vid) => (alertState.get(vid)?.data.tier || 0) < tier,
            );
            if (victim) {
              hidePopup(victim);
              showPopup(existing);
            } else {
              refreshPopupEl(existing); // still refresh in case of re-attach
            }
          }
          playTone(tier);
        } else if (visiblePopupIds.includes(id)) {
          // Same tier — just update the existing popup in place.
          refreshPopupEl(existing);
        }
      }
    });

    // 3) Mark alerts that no longer appear as resolved. Keep
    //    them in history; remove their popups.
    alertState.forEach((state, id) => {
      if (!incomingIds.has(id) && state.resolvedAt === null) {
        state.resolvedAt = state.data.time_s || state.lastSeenAt;
        // Push into bounded history.
        alertHistory.push({
          alert_id: id,
          title: state.data.title || state.data.fault_type || "—",
          tier: state.data.tier || 0,
          severity_label: state.data.severity_label || "INFO",
          firstSeenAt: state.firstSeenAt,
          resolvedAt: state.resolvedAt,
          acked: state.acked,
        });
        if (alertHistory.length > MAX_HISTORY) alertHistory.shift();
        hidePopup(id);
      }
    });

    // 4) Refresh the alert count chip + alert center.
    refreshHeaderCounts();
    renderAlertCenter();
  }

  function refreshHeaderCounts() {
    let active = 0,
      warning = 0,
      critical = 0;
    alertState.forEach((state) => {
      if (state.resolvedAt !== null) return;
      if (state.acked) return;
      active += 1;
      const tier = Number(state.data.tier || 0);
      if (tier === 2) critical += 1;
      else if (tier === 1) warning += 1;
    });
    const m = document.getElementById("meta-alert-counts");
    if (m) m.textContent = `alerts: ${active} / ${warning} / ${critical}`;
    const c = document.getElementById("alert-center-count");
    if (c) {
      c.textContent = String(active);
      c.classList.toggle("zero", active === 0);
    }
  }

  // -------- Reset for new scenario / replay -----------------------
  // Called by the WebSocket / polling path when the scenario
  // name changes, or by the user via the alert center "clear"
  // action. Clears the live state, the popups, the history,
  // and the header counts.
  function resetAlertSystem() {
    visiblePopupIds.slice().forEach(hidePopup);
    alertState.clear();
    alertHistory.length = 0;
    refreshHeaderCounts();
    renderAlertCenter();
  }

  // -------- Wire up static controls (one-shot at load) -----------
  function wireAlertUI() {
    // Unlock audio on first user gesture (browser autoplay policy).
    const unlock = () => {
      unlockAudio();
      document.removeEventListener("click", unlock);
      document.removeEventListener("keydown", unlock);
    };
    document.addEventListener("click", unlock);
    document.addEventListener("keydown", unlock);

    const toggle = document.getElementById("alert-center-toggle");
    if (toggle) {
      toggle.addEventListener("click", () => {
        document.getElementById("alert-center").classList.toggle("open");
        renderAlertCenter();
      });
    }
    const close = document.getElementById("alert-center-close");
    if (close) {
      close.addEventListener("click", () => {
        document.getElementById("alert-center").classList.remove("open");
      });
    }
    document.querySelectorAll("#alert-center .ac-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document
          .querySelectorAll("#alert-center .ac-tab")
          .forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        acTab = tab.dataset.tab || "active";
        renderAlertCenter();
      });
    });
    const dm = document.getElementById("detail-modal");
    if (dm) {
      dm.addEventListener("click", (e) => {
        if (e.target === dm) closeDetail();
      });
    }
    const dmClose = document.getElementById("detail-modal-close");
    if (dmClose) dmClose.addEventListener("click", closeDetail);
    const dmDismiss = document.getElementById("detail-modal-dismiss");
    if (dmDismiss) dmDismiss.addEventListener("click", closeDetail);
    const dmAck = document.getElementById("detail-modal-ack");
    if (dmAck)
      dmAck.addEventListener("click", () => {
        if (currentDetailId) acknowledgeAlert(currentDetailId);
      });
    // Esc closes the modal.
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (dm && dm.classList.contains("open")) closeDetail();
      }
    });
  }
  wireAlertUI();

  // -------- Active alerts (deduplicated, keyed by fault_type) ----------
  // The server emits the *current* set per tick. We keep the
  // set alive across ticks and re-use an existing entry when
  // the same fault_type recurs.
  const activeAlerts = new Map(); // fault_type -> last seen alert
  const alertOrder = []; // preserve insertion order

  function updateAlerts(currentAlerts) {
    // PHASE 24 — the popup / alert-center system is the
    // primary consumer now. The PHASE 13 banner feed still
    // updates for back-compat (existing tests rely on
    // ``alerts-list`` rendering). The popup system dedupes
    // on ``alert_id``; the legacy feed dedupes on
    // ``fault_type`` for back-compat.
    updateAlertSystem(currentAlerts);

    const seenNow = new Set();
    (currentAlerts || []).forEach((a) => {
      const key = a.fault_type;
      seenNow.add(key);
      if (!activeAlerts.has(key)) {
        alertOrder.push(key);
      }
      activeAlerts.set(key, a);
    });
    // Remove alerts that no longer appear in the current tick.
    for (let i = alertOrder.length - 1; i >= 0; i--) {
      if (!seenNow.has(alertOrder[i])) {
        activeAlerts.delete(alertOrder[i]);
        alertOrder.splice(i, 1);
      }
    }
    renderAlerts();
  }

  function renderAlerts() {
    const list = $("alerts-list");
    if (!list) return;
    if (alertOrder.length === 0) {
      list.innerHTML = `<div class="alert-empty">No active conditions.</div>`;
      return;
    }
    const html = alertOrder
      .slice(0, 5)
      .map((key) => {
        const a = activeAlerts.get(key);
        const sev = (a.severity || "INFO").toLowerCase();
        return `
        <div class="alert-banner sev-${a.severity}">
          <div class="k">FAULT TYPE</div><div class="v">${a.fault_type}</div>
          <div class="k">CONFIDENCE</div><div class="v">${fmt(a.confidence, 2)}</div>
          <div class="k">EVIDENCE</div><div class="v">${a.evidence || "—"}</div>
          <div class="k">ENV CONTEXT</div><div class="v">${a.env_context || "—"}</div>
          <div class="k">SENSOR HEALTH</div><div class="v">${a.sensor_health || "—"}</div>
          <div class="k">RECOMMENDED INVESTIGATION</div>
          <div class="v">${a.recommendation || "—"}</div>
        </div>`;
      })
      .join("");
    list.innerHTML = html;
  }

  // -------- Channels (LEFT column) -------------------------------------
  // Each channel tile shows: name, value, unit, delta arrow, sparkline.
  const CHANNELS = [
    { id: "rpm", name: "RPM", unit: "rpm", field: "rpm", digits: 0 },
    {
      id: "throttle",
      name: "Throttle",
      unit: "%",
      field: "throttle",
      digits: 1,
      scale: 100,
    },
    {
      id: "fuel",
      name: "Fuel Flow",
      unit: "L/h",
      field: "fuel_flow_lph",
      digits: 1,
    },
    {
      id: "oilp",
      name: "Oil P",
      unit: "psi",
      field: "oil_pressure_psi",
      digits: 1,
    },
    {
      id: "oilt",
      name: "Oil T",
      unit: "°C",
      field: "oil_temperature_c",
      digits: 1,
    },
  ];
  const sparkCharts = {}; // channelId -> Chart
  const lastChannelValue = {};

  function renderChannels(eng) {
    CHANNELS.forEach((ch) => {
      let v = eng[ch.field];
      if (v === null || v === undefined) v = 0;
      if (ch.scale) v = v * ch.scale;
      const valEl = $(`ch-${ch.id}-value`);
      const unitEl = $(`ch-${ch.id}-unit`);
      const deltaEl = $(`ch-${ch.id}-delta`);
      if (valEl) valEl.textContent = fmt(v, ch.digits);
      if (unitEl) unitEl.textContent = ch.unit;
      if (deltaEl) {
        const prev = lastChannelValue[ch.id];
        if (prev !== undefined) {
          const d = v - prev;
          if (Math.abs(d) < 1e-6) {
            deltaEl.textContent = "—";
            deltaEl.className = "delta";
          } else if (d > 0) {
            deltaEl.textContent = "▲";
            deltaEl.className = "delta up";
          } else {
            deltaEl.textContent = "▼";
            deltaEl.className = "delta down";
          }
        }
        lastChannelValue[ch.id] = v;
      }
      // Push to sparkline chart.
      const sc = sparkCharts[ch.id];
      if (sc) {
        sc.data.labels.push("");
        sc.data.datasets[0].data.push(v);
        if (sc.data.labels.length > 60) {
          sc.data.labels.shift();
          sc.data.datasets[0].data.shift();
        }
        sc.update("none");
      }
    });
  }

  // -------- Charts (CENTER column) ------------------------------------
  // 8 canvases: rpm, egt, cht, oilp, fuel, vib-imu, health, rul
  const MAX_POINTS = 200;
  const mainCharts = {};

  function lineOptions(label, borderColor) {
    const isGreen = borderColor === "#22c55e" || borderColor === "var(--green)";
    const fillColor = isGreen
      ? "rgba(16, 185, 129, 0.06)"
      : "rgba(56, 189, 248, 0.06)";
    return {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label,
            data: [],
            borderColor,
            backgroundColor: fillColor,
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.2,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { display: false },
          y: {
            beginAtZero: false,
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: {
              color: "rgba(148, 163, 184, 0.65)",
              font: { family: "'JetBrains Mono', monospace", size: 9 },
            },
          },
        },
        plugins: { legend: { display: false } },
      },
    };
  }

  function dualLineOptions(label1, label2) {
    return {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: label1,
            data: [],
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.06)",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.2,
            fill: true,
            yAxisID: "y",
          },
          {
            label: label2,
            data: [],
            borderColor: "#f59e0b",
            backgroundColor: "transparent",
            borderWidth: 1.4,
            pointRadius: 0,
            tension: 0.2,
            fill: false,
            yAxisID: "y1",
            borderDash: [3, 3],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { display: false },
          y: {
            type: "linear",
            position: "left",
            beginAtZero: false,
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: {
              color: "rgba(148, 163, 184, 0.65)",
              font: { family: "'JetBrains Mono', monospace", size: 9 },
            },
          },
          y1: {
            type: "linear",
            position: "right",
            beginAtZero: true,
            grid: { display: false },
            ticks: {
              color: "rgba(245, 158, 11, 0.7)",
              font: { family: "'JetBrains Mono', monospace", size: 9 },
            },
          },
        },
        plugins: { legend: { display: false } },
      },
    };
  }

  function bandedLineOptions(label) {
    return {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "upper",
            data: [],
            borderColor: "transparent",
            backgroundColor: "rgba(56, 189, 248, 0.12)",
            borderWidth: 0,
            pointRadius: 0,
            fill: "+1",
            tension: 0.2,
          },
          {
            label: "central",
            data: [],
            borderColor: "#38bdf8",
            backgroundColor: "transparent",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.2,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { display: false },
          y: {
            beginAtZero: false,
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: {
              color: "rgba(148, 163, 184, 0.65)",
              font: { family: "'JetBrains Mono', monospace", size: 9 },
            },
          },
        },
        plugins: { legend: { display: false } },
      },
    };
  }

  function initCharts() {
    if (typeof Chart === "undefined") {
      document.body.insertAdjacentHTML(
        "beforeend",
        `<div class="insufficient">Chart.js not loaded — charts disabled</div>`,
      );
      return;
    }
    Chart.defaults.color = "rgba(148, 163, 184, 0.7)";
    Chart.defaults.borderColor = "rgba(255, 255, 255, 0.05)";
    Chart.defaults.font.family =
      "'JetBrains Mono', monospace, -apple-system, sans-serif";

    // Sparklines for LEFT column
    CHANNELS.forEach((ch) => {
      const canvas = $(`ch-${ch.id}-spark`);
      if (!canvas) return;
      sparkCharts[ch.id] = new Chart(canvas, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              data: [],
              borderColor: "#38bdf8",
              backgroundColor: "rgba(56, 189, 248, 0.08)",
              borderWidth: 1.4,
              pointRadius: 0,
              tension: 0.25,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: { x: { display: false }, y: { display: false } },
          plugins: { legend: { display: false } },
        },
      });
    });

    // 8 main charts
    mainCharts.rpm = new Chart(
      $("chart-rpm"),
      lineOptions("RPM actual", "#38bdf8"),
    );
    mainCharts.egt = new Chart(
      $("chart-egt"),
      lineOptions("EGT actual", "#38bdf8"),
    );
    mainCharts.cht = new Chart($("chart-cht"), lineOptions("CHT", "#38bdf8"));
    mainCharts.oilp = new Chart(
      $("chart-oilp"),
      lineOptions("Oil pressure", "#38bdf8"),
    );
    mainCharts.fuel = new Chart(
      $("chart-fuel"),
      lineOptions("Fuel flow", "#38bdf8"),
    );
    mainCharts.vibImu = new Chart(
      $("chart-vib-imu"),
      dualLineOptions("Vibration (g)", "Vert. accel (m/s²)"),
    );
    mainCharts.health = new Chart(
      $("chart-health"),
      lineOptions("Health", "#10b981"),
    );
    mainCharts.rul = new Chart($("chart-rul"), bandedLineOptions("RUL range"));
  }

  function pushMain(actual, expected, chId) {
    const c = mainCharts[chId];
    if (!c) return;
    c.data.labels.push("");
    c.data.datasets[0].data.push(actual);
    if (
      c.data.datasets.length > 1 &&
      expected !== undefined &&
      expected !== null
    ) {
      c.data.datasets[1].data.push(expected);
    } else if (c.data.datasets.length > 1) {
      // Keep the bands aligned: push null so the line breaks cleanly.
      c.data.datasets[1].data.push(null);
    }
    if (c.data.labels.length > MAX_POINTS) {
      c.data.labels.shift();
      c.data.datasets.forEach((d) => d.data.shift());
    }
    c.update("none");
  }

  function pushHealth(healthScore) {
    const c = mainCharts.health;
    if (!c) return;
    c.data.labels.push("");
    c.data.datasets[0].data.push(healthScore);
    if (c.data.labels.length > MAX_POINTS) {
      c.data.labels.shift();
      c.data.datasets[0].data.shift();
    }
    c.update("none");
  }

  function pushRul(central, lower, upper) {
    const c = mainCharts.rul;
    if (!c) return;
    c.data.labels.push("");
    c.data.datasets[0].data.push(upper);
    c.data.datasets[1].data.push(central);
    // The fill="+1" dataset fills to the next dataset (lower).
    if (!c.data.datasets[2]) {
      c.data.datasets.push({
        label: "lower",
        data: [],
        borderColor: "transparent",
        backgroundColor: "transparent",
        borderWidth: 0,
        pointRadius: 0,
        fill: false,
        tension: 0.15,
      });
    }
    c.data.datasets[2].data.push(lower);
    if (c.data.labels.length > MAX_POINTS) {
      c.data.labels.shift();
      c.data.datasets.forEach((d) => d.data.shift());
    }
    c.update("none");
  }

  // -------- Snapshot update -------------------------------------------
  function update(snap) {
    if (!snap) return;
    const eng = snap.engine_state || {};
    const env = snap.environment || {};
    const risk = snap.risk || {};
    const health = snap.health || {};
    const rul = snap.rul || {};
    const anomaly = snap.anomaly || {};
    const fc = snap.fault_classification;
    const residual = snap.residual || {};
    const twinExp = snap.twin_expected || {};

    // -------- Header / time
    $("meta-time").textContent = "t = " + fmt(snap.time_s, 1) + " s";
    $("meta-mission-phase").textContent =
      "phase: " + (risk.mission_phase || "—");

    // -------- TOP strip tiles
    // 1. Mission Risk in percentage
    const rawRisk = Number(risk.risk_score ?? 0);
    const riskPct = rawRisk <= 1.0 ? rawRisk * 100 : rawRisk;
    setChip("tile-risk-chip", risk.status || (riskPct < 25 ? "GO" : (riskPct < 60 ? "CAUTION" : "ABORT")), "—");
    const riskValEl = $("tile-risk-value");
    riskValEl.textContent = `${riskPct.toFixed(1)}%`;
    riskValEl.style.color = riskPct >= 60 ? "var(--red)" : (riskPct >= 25 ? "var(--amber)" : "var(--fg)");
    $("tile-risk-sub").textContent = "level: " + (risk.risk_level || "—");
    const riskBarEl = $("tile-risk-bar");
    if (riskBarEl) {
      riskBarEl.style.width = `${Math.min(100, Math.max(0, riskPct))}%`;
      riskBarEl.style.background = riskPct >= 60 ? "linear-gradient(90deg, #ef4444, #dc2626)" : (riskPct >= 25 ? "linear-gradient(90deg, #f59e0b, #ea580c)" : "linear-gradient(90deg, #10b981, #059669)");
    }

    // 2. Health Index in percentage
    const rawHealth = Number(health.overall_score ?? 1.0);
    const healthPct = rawHealth <= 1.0 ? rawHealth * 100 : rawHealth;
    setChip("tile-health-chip", health.overall_label || "HEALTHY", "—");
    const healthValEl = $("tile-health-value");
    healthValEl.textContent = `${healthPct.toFixed(1)}%`;
    $("tile-health-sub").textContent = "trend: " + (health.trend || "—");
    const healthBarEl = $("tile-health-bar");
    if (healthBarEl) {
      healthBarEl.style.width = `${Math.min(100, Math.max(0, healthPct))}%`;
    }

    // 3. Engine Status in percentage (performance score with RPM & MAP in subtext)
    const perfScore = health["sub.PERFORMANCE.score"] !== undefined 
      ? Number(health["sub.PERFORMANCE.score"]) 
      : Number(health.overall_score || 1.0);
    const engineStatusPct = Math.max(0, Math.min(100, (perfScore <= 1.0 ? perfScore * 100 : perfScore)));
    const engToken = engineStatusPct >= 90 ? "NORMAL" : (engineStatusPct >= 75 ? "DEGRADED" : "CRITICAL");
    setChip("tile-engine-chip", engToken, "NORMAL");
    const engValEl = $("tile-engine-value");
    engValEl.textContent = `${engineStatusPct.toFixed(1)}%`;
    engValEl.style.color = engineStatusPct < 75 ? "var(--red)" : (engineStatusPct < 90 ? "var(--amber)" : "var(--fg)");
    $("tile-engine-sub").textContent = `${fmt(eng.rpm, 0)} rpm · MAP ${fmt(eng.manifold_pressure_inhg, 1)} inHg`;
    const engBarEl = $("tile-engine-bar");
    if (engBarEl) {
      engBarEl.style.width = `${engineStatusPct}%`;
      engBarEl.style.background = engineStatusPct < 75 ? "linear-gradient(90deg, #ef4444, #dc2626)" : (engineStatusPct < 90 ? "linear-gradient(90deg, #f59e0b, #ea580c)" : "linear-gradient(90deg, #38bdf8, #0284c7)");
    }

    // 4. Remaining Useful Life
    setChip("tile-rul-chip", rul.status, "—");
    $("tile-rul-value").textContent =
      rul.tte_hours_central === null || rul.tte_hours_central === undefined
        ? "— h"
        : fmt(rul.tte_hours_central, 1) + " h";
    $("tile-rul-sub").textContent =
      rul.tte_hours_lower === null || rul.tte_hours_lower === undefined
        ? "—"
        : `L ${fmt(rul.tte_hours_lower, 1)} h · U ${fmt(rul.tte_hours_upper, 1)} h`;
    const rulBarEl = $("tile-rul-bar");
    if (rulBarEl && rul.tte_hours_central !== null && rul.tte_hours_central !== undefined) {
      rulBarEl.style.width = `${Math.min(100, Math.max(0, (Number(rul.tte_hours_central) / 100) * 100))}%`;
    }

    // -------- LEFT column
    renderChannels(eng);

    // -------- CENTER column — push data to all 8 charts
    pushMain(eng.rpm, twinExp.rpm, "rpm");
    pushMain(eng.egt_c, twinExp.egt, "egt");
    pushMain(eng.cht_c, null, "cht");
    pushMain(eng.oil_pressure_psi, twinExp.oil_pressure, "oilp");
    pushMain(eng.fuel_flow_lph, twinExp.fuel_flow, "fuel");
    pushMain(eng.vibration_rms_g, env.vertical_accel_mps2, "vibImu");
    pushHealth(health.overall_score);
    pushRul(rul.tte_hours_central, rul.tte_hours_lower, rul.tte_hours_upper);

    // -------- RIGHT column
    // Current diagnosis — drivers (kept from PHASE 13) + subsystem mini-bars.
    const drvBody = $("risk-drivers-body");
    if (drvBody) {
      drvBody.innerHTML = "";
      (risk.drivers || []).slice(0, 5).forEach((d) => {
        const sevTok =
          d.severity === "INFO"
            ? "GO"
            : d.severity === "WARNING"
              ? "CAUTION"
              : d.severity === "ABORT"
                ? "ABORT"
                : d.severity === "RETURN_TO_BASE"
                  ? "RETURN_TO_BASE"
                  : "GO";
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${d.signal || "—"}</td>
          <td class="num">${safe(d.value, 3)}</td>
          <td class="num">${safe(d.contribution, 3)}</td>
          <td><span class="chip" style="background:${color(sevTok)}">${d.severity || "—"}</span></td>`;
        drvBody.appendChild(tr);
      });
      if ((risk.drivers || []).length === 0) {
        drvBody.innerHTML = `<tr><td colspan="4" class="insufficient">no driver data</td></tr>`;
      }
    }

    const subs = $("subsystems");
    if (subs) {
      subs.innerHTML = "";
      const subKeys = Object.keys(health).filter((k) => k.startsWith("sub."));
      subKeys.forEach((k) => {
        const sub = health[k] || {};
        const score = Number(sub.score || 0);
        const div = document.createElement("div");
        div.className = "subsystem";
        div.innerHTML = `<div class="name">${sub.subsystem || k}</div>
          <div class="bar"><div class="fill" style="width:${(score * 100).toFixed(1)}%; background:${scoreToColor(score)}"></div></div>
          <div class="val">${fmt(score, 2)}</div>`;
        subs.appendChild(div);
      });
      if (subKeys.length === 0) {
        subs.innerHTML = `<div class="insufficient">no subsystem data</div>`;
      }
    }

    // Fault probabilities
    const fpBody = $("fault-probabilities-body");
    if (fpBody) {
      fpBody.innerHTML = "";
      if (!fc) {
        fpBody.innerHTML = `<tr><td colspan="2" class="insufficient">no classifier output</td></tr>`;
      } else {
        const probs = Object.entries(fc.probabilities || {}).sort(
          (a, b) => b[1] - a[1],
        );
        probs.forEach(([k, v]) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${k}</td><td class="num">${fmt(v, 2)}</td>`;
          fpBody.appendChild(tr);
        });
      }
    }

    // Sensor health
    const shBody = $("sensor-health-body");
    if (shBody) {
      shBody.innerHTML = "";
      // Per-channel noise mode from the sample (if any). Fall
      // back to "—" when not provided.
      const sample = snap.sample || {};
      const sensorHealth = sample.sensor_health || {};
      const channels = Object.keys(sensorHealth).sort();
      if (channels.length === 0) {
        shBody.innerHTML = `<div class="row"><span class="k">All channels</span><span class="v" style="color:var(--green)">NORMAL</span></div>`;
      } else {
        channels.forEach((ch) => {
          const mode = (sensorHealth[ch] && sensorHealth[ch].mode) || "—";
          const tag = mode === "NORMAL" ? "OK" : mode;
          const tok = mode === "NORMAL" ? "GO" : "CAUTION";
          shBody.insertAdjacentHTML(
            "beforeend",
            `<div class="row"><span class="k">${ch}</span>
              <span class="v" style="color:${color(tok)}">${tag}</span></div>`,
          );
        });
      }
    }

    // Environmental condition
    $("env-alt").textContent = fmt(env.altitude_m, 0) + " m";
    $("env-airspeed").textContent = fmt(env.airspeed_mps, 1) + " m/s";
    $("env-temp").textContent = fmt(env.temperature_c, 1) + " °C";
    $("env-pressure").textContent = fmt(env.pressure_pa, 0) + " Pa";
    $("env-wind").textContent = fmt(env.total_w_mps, 2) + " m/s";
    $("env-gust").textContent = env.is_gust_active ? "yes" : "no";
    $("env-vac").textContent = fmt(env.vertical_accel_mps2, 2) + " m/s²";

    // -------- BOTTOM strip
    // Mission timeline
    const tNow = snap.time_s || 0;
    const missionEnd = (risk.hours_to_destination || 0) * 3600 + tNow;
    const tFrac =
      missionEnd > 0 ? Math.max(0, Math.min(1, tNow / missionEnd)) : 0;
    const tl = $("timeline-fill");
    if (tl) tl.style.width = (tFrac * 100).toFixed(1) + "%";
    $("timeline-phase").textContent = risk.mission_phase || "—";

    // Alerts
    updateAlerts(snap.alerts);

    // Events log
    const events = $("events-list");
    if (events) {
      const evts = (snap.events || []).slice(-8).reverse();
      if (evts.length === 0) {
        events.innerHTML = `<div class="insufficient">No events yet</div>`;
      } else {
        events.innerHTML = evts
          .map(
            (e) => `
          <div class="event">
            <div class="t">t=${fmt(e.time_s, 0)}s</div>
            <div class="k">${e.kind || "—"}</div>
            <div class="d">${e.description || "—"}</div>
          </div>`,
          )
          .join("");
      }
    }

    // Model confidence
    const conf = Number(snap.model_confidence || 0);
    $("model-conf-fill").style.width = (conf * 100).toFixed(1) + "%";
    $("model-conf-value").textContent = fmt(conf, 2);

    // Latency
    const lat = snap.latency || {};
    const stages = ["ingestion", "digital_twin", "ai", "risk"];
    const latEl = $("latency-body");
    if (latEl) {
      latEl.innerHTML =
        stages
          .map((s) => {
            const v = lat[s];
            return `<div class="latency-row">
          <span class="k">${s}</span>
          <span class="v">${v === undefined || v === null ? "—" : fmt(v * 1000, 1) + " ms"}</span>
        </div>`;
          })
          .join("") +
        `
        <div class="latency-row">
          <span class="k">end-to-end</span>
          <span class="v">${lat.end_to_end_s === undefined || lat.end_to_end_s === null ? "—" : fmt(lat.end_to_end_s * 1000, 1) + " ms"}</span>
        </div>`;
    }

    // -------- Predictive maintenance panel
    renderMaintenance(rul, health, risk);
  }

  // -------- Predictive Maintenance Panel ------------------------------
  // ponytail: dynamic maintenance tasks projected linearly from RUL estimate;
  // ceiling: does not parse discrete aircraft logbook records; upgrade: ingest maintenance log API.
  function renderMaintenance(rul, health, risk) {
    const tteEl = $("maint-tte-val");
    if (!tteEl) return;

    const tte = rul.tte_hours_central;
    tteEl.textContent =
      tte === null || tte === undefined ? "— h" : fmt(tte, 1) + " h";
    $("maint-tte-bounds").textContent =
      `Confidence Bounds: Lower ${safe(rul.tte_hours_lower, 1)} h · Upper ${safe(rul.tte_hours_upper, 1)} h`;
    $("maint-model-status").textContent = rul.model_status || "CLOSED_FORM";
    $("maint-wear-rate").textContent =
      rul.wear_rate_per_hour !== undefined && rul.wear_rate_per_hour !== null
        ? fmt(rul.wear_rate_per_hour * 100, 3) + " %/h"
        : "— %/h";
    $("maint-trend").textContent = rul.trend || "STABLE";
    $("maint-confidence").textContent = fmt(rul.confidence || 0, 2);

    // Advisory Box
    const box = $("maint-action-box");
    const title = $("maint-action-title");
    const desc = $("maint-action-desc");
    const isCritical =
      rul.status === "RUL_CRITICAL" ||
      risk.risk_level === "CRITICAL" ||
      (tte !== null && tte !== undefined && tte < 10);
    const isWarning =
      rul.status === "RUL_DEGRADED" ||
      risk.risk_level === "HIGH" ||
      (tte !== null && tte !== undefined && tte < 30);

    if (box && title && desc) {
      if (isCritical) {
        box.className = "maint-action-box action-critical";
        title.textContent =
          "CRITICAL ACTION REQUIRED — ENGINE TEARDOWN RECOMMENDED";
        desc.textContent =
          "Remaining useful life threshold reached. High risk of in-flight component failure. Ground aircraft immediately for borescope and overhaul inspection.";
      } else if (isWarning) {
        box.className = "maint-action-box action-warning";
        title.textContent = "PRECAUTIONARY SERVICE — ACCELERATED WEAR DETECTED";
        desc.textContent =
          "Elevated wear progression observed by the digital twin. Schedule oil filter analysis and cylinder inspection within 15 flight hours.";
      } else {
        box.className = "maint-action-box action-nominal";
        title.textContent =
          "NOMINAL — PROGRESSIVE MAINTENANCE CYCLE ON SCHEDULE";
        desc.textContent =
          "Component wear accumulation is within certified flight envelopes. Proceed with standard progressive service intervals.";
      }
    }

    // Subsystems wear accumulation list
    const subList = $("maint-subsystems-list");
    if (subList) {
      subList.innerHTML = "";
      const subKeys = Object.keys(health).filter((k) => k.startsWith("sub."));
      if (subKeys.length === 0) {
        subList.innerHTML = `<div class="insufficient">Awaiting subsystem wear telemetry…</div>`;
      } else {
        subKeys.forEach((k) => {
          const sub = health[k] || {};
          const score = Number(sub.score || 0);
          const wear = Math.max(0, Math.min(1, 1 - score));
          const div = document.createElement("div");
          div.className = "subsystem";
          div.innerHTML = `<div class="name">${sub.subsystem || k}</div>
            <div class="bar"><div class="fill" style="width:${(wear * 100).toFixed(1)}%; background:${scoreToColor(1 - wear)}"></div></div>
            <div class="val">${(wear * 100).toFixed(0)}% wear</div>`;
          subList.appendChild(div);
        });
      }
    }

    // Dynamic tasks due update based on RUL
    const tteVal = tte !== null && tte !== undefined ? Number(tte) : 50;
    const oilDue = Math.max(0, Math.min(50, Math.floor(tteVal)));
    const plugsDue = Math.max(0, Math.min(100, Math.floor(tteVal * 1.5)));
    const compDue = Math.max(0, Math.min(100, Math.floor(tteVal * 1.8)));
    const valveDue = Math.max(0, Math.min(200, Math.floor(tteVal * 2.2)));

    $("maint-task-oil").textContent = `~${oilDue} h`;
    $("maint-task-plugs").textContent = `~${plugsDue} h`;
    $("maint-task-comp").textContent = `~${compDue} h`;
    $("maint-task-valve").textContent = `~${valveDue} h`;

    // Progress bar fills (ratio of remaining interval)
    const setBar = (id, cur, max) => {
      const el = $(id);
      if (el)
        el.style.width =
          Math.min(100, Math.max(0, (cur / max) * 100)).toFixed(1) + "%";
    };
    setBar("maint-task-oil-bar", oilDue, 50);
    setBar("maint-task-plugs-bar", plugsDue, 100);
    setBar("maint-task-comp-bar", compDue, 100);
    setBar("maint-task-valve-bar", valveDue, 200);

    const setTaskCard = (statusId, cardId, due) => {
      const statusEl = $(statusId);
      const cardEl = $(cardId);
      if (!statusEl) return;
      if (due <= 10) {
        statusEl.textContent = "EXPEDITE";
        statusEl.style.background = "var(--red)";
        if (cardEl) cardEl.className = "maint-card-featured urgent";
      } else if (due <= 25) {
        statusEl.textContent = "UPCOMING";
        statusEl.style.background = "var(--amber)";
        if (cardEl) cardEl.className = "maint-card-featured warning";
      } else {
        statusEl.textContent = "ON SCHEDULE";
        statusEl.style.background = "var(--green)";
        if (cardEl) cardEl.className = "maint-card-featured";
      }
    };
    setTaskCard("maint-task-oil-status", "card-task-oil", oilDue);
    setTaskCard("maint-task-plugs-status", "card-task-plugs", plugsDue);
    setTaskCard("maint-task-comp-status", "card-task-comp", compDue);
    setTaskCard("maint-task-valve-status", "card-task-valve", valveDue);
  }

  // -------- Transport: WebSocket primary, REST fallback ----------------
  let ws = null;
  function connect() {
    const url =
      (location.protocol === "https:" ? "wss://" : "ws://") +
      location.host +
      "/api/stream";
    try {
      ws = new WebSocket(url);
    } catch (e) {
      $("conn-status").textContent = "ws error, falling back to polling";
      return startPolling();
    }
    ws.onopen = () => {
      $("conn-status").textContent = "ws: connected";
    };
    ws.onmessage = (ev) => {
      try {
        update(JSON.parse(ev.data));
      } catch (e) {
        /* ignore */
      }
    };
    ws.onclose = () => {
      $("conn-status").textContent = "ws: closed, retrying in 1s…";
      setTimeout(connect, 1000);
    };
    ws.onerror = () => {
      $("conn-status").textContent = "ws: error, falling back to polling";
      startPolling();
    };
  }
  function startPolling() {
    setInterval(async () => {
      try {
        const r = await fetch("/api/snapshot/latest");
        const j = await r.json();
        if (j.snapshot) update(j.snapshot);
      } catch (e) {
        /* ignore */
      }
    }, 200);
  }

  // -------- Scenarios dropdown ----------------------------------------
  async function loadScenarios() {
    try {
      const r = await fetch("/api/scenarios");
      const j = await r.json();
      const sel = $("scenario-select");
      sel.innerHTML = "";
      (j.scenarios || []).forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s.name;
        opt.textContent = s.name + (s.description ? " — " + s.description : "");
        sel.appendChild(opt);
      });
      const current = await (await fetch("/api/scenario")).json();
      sel.value = current.name;
      sel.addEventListener("change", async () => {
        const r = await fetch("/api/scenario", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: sel.value }),
        });
        if (r.ok) {
          // Reload to get a clean WS connection + fresh chart history.
          window.location.reload();
        } else {
          $("conn-status").textContent = "scenario switch failed: " + r.status;
        }
      });
    } catch (e) {
      $("conn-status").textContent = "scenarios load failed: " + e.message;
    }
  }

  // -------- View / Section navigation ---------------------------------
  function initViewNavigation() {
    const navItems = document.querySelectorAll("#sidebar-nav .nav-item");
    const sections = document.querySelectorAll(".view-section");
    if (!navItems.length || !sections.length) return;

    function switchSection(targetId) {
      navItems.forEach((btn) => {
        const isActive = btn.dataset.section === targetId;
        btn.classList.toggle("active", isActive);
      });
      sections.forEach((sec) => {
        const isActive = sec.id === `view-${targetId}`;
        sec.classList.toggle("active", isActive);
      });

      // When switching to charts, overview, or replay, resize Chart.js/Canvas instances so they adapt cleanly
      if (targetId === "charts" || targetId === "overview") {
        setTimeout(() => {
          Object.values(mainCharts).forEach((c) => {
            if (c && typeof c.resize === "function") c.resize();
          });
          Object.values(sparkCharts).forEach((c) => {
            if (c && typeof c.resize === "function") c.resize();
          });
        }, 40);
      } else if (targetId === "replay") {
        setTimeout(() => {
          window.dispatchEvent(new Event("resize"));
        }, 40);
      }
    }

    navItems.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.section;
        if (target) {
          switchSection(target);
          try {
            history.replaceState(null, "", `#${target}`);
          } catch (e) {
            /* ignore */
          }
        }
      });
    });

    // Check URL hash on load
    const hash = (window.location.hash || "").replace("#", "");
    if (hash && document.getElementById(`view-${hash}`)) {
      switchSection(hash);
    }
  }

  // ------------------------------------------------------------------
  // SECTION: MISSION REPLAY (MVP) ENGINE
  // ponytail: Lightweight client-side simulation engine, zero extra frameworks.
  // ------------------------------------------------------------------
  const MISSIONS_DATA = {
    mission_07: {
      id: "MALE-UAV-07",
      name: "Sortie #07: Reconnaissance Patrol (CHT & Valve Anomaly)",
      durationStr: "04h 18m",
      totalSeconds: 15480, // 4h 18m
      distance: "582 km",
      maxAlt: "14,200 ft",
      avgSpeed: "135 kts",
      status: "ANOMALY LOGGED",
      statusType: "warning",
      statusSub: "Post-Flight Inspection Required",
      anomalyPct: 56, // t = 02h 24m
      ai: {
        faultType: "Exhaust Valve Sticking / CHT Thermal Excursion",
        confidence: "94.8%",
        healthPct: 68,
        rulHours: "34.5",
        desc: "Exhaust valve thermal buildup detected at t = 02h 24m. Digital Twin acoustic-thermal residual exceeded threshold (ε = +18.4%). Projected degradation accelerates if continuous cruise above 5,400 RPM is sustained.",
        action:
          "Optical Borescope Inspection of Cylinder #3 exhaust seat prior to next deployment.",
      },
      waypoints: [
        {
          name: "WP-00 (Base)",
          x: 70,
          y: 250,
          lat: "28.6139° N",
          lon: "77.2090° E",
          alt: 500,
          phase: "TAKEOFF",
        },
        {
          name: "WP-01 (Climb)",
          x: 170,
          y: 160,
          lat: "28.8402° N",
          lon: "77.4810° E",
          alt: 12000,
          phase: "CLIMB",
        },
        {
          name: "WP-02 (Orbit N)",
          x: 330,
          y: 80,
          lat: "29.2150° N",
          lon: "77.9200° E",
          alt: 14200,
          phase: "SURVEILLANCE",
        },
        {
          name: "WP-03 (Orbit S)",
          x: 450,
          y: 140,
          lat: "29.0800° N",
          lon: "78.3500° E",
          alt: 14200,
          phase: "SURVEILLANCE (ANOMALY)",
        },
        {
          name: "WP-04 (Corridor)",
          x: 360,
          y: 240,
          lat: "28.7900° N",
          lon: "77.8900° E",
          alt: 8500,
          phase: "RTB INGRESS",
        },
        {
          name: "WP-05 (Touchdown)",
          x: 75,
          y: 255,
          lat: "28.6145° N",
          lon: "77.2105° E",
          alt: 500,
          phase: "TOUCHDOWN",
        },
      ],
      events: [
        {
          pct: 0,
          time: "00:00:00",
          phase: "WP-00",
          desc: "Takeoff roll initiated, engine warmup completed",
          sev: "INFO",
          sensor: "RPM: 5,400",
          action: "Normal",
        },
        {
          pct: 18,
          time: "00:46:12",
          phase: "WP-01",
          desc: "Level-off at FL120, auto-mixture cruise engaged",
          sev: "INFO",
          sensor: "Alt: 12,000 ft",
          action: "Monitoring",
        },
        {
          pct: 42,
          time: "01:48:30",
          phase: "WP-02",
          desc: "Surveillance orbit active, electro-optical feed nominal",
          sev: "INFO",
          sensor: "CHT: 184°C",
          action: "Patrol",
        },
        {
          pct: 56,
          time: "02:24:40",
          phase: "WP-03",
          desc: "Cylinder #3 CHT thermal excursion beyond threshold (>214°C)",
          sev: "WARNING",
          sensor: "CHT: 214°C",
          action: "Throttle 92%",
        },
        {
          pct: 62,
          time: "02:40:15",
          phase: "WP-03",
          desc: "Digital Twin acoustic residual alert (RMS: 3.4 mm/s)",
          sev: "CRITICAL",
          sensor: "Vib: 3.4 mm/s",
          action: "Advisory Alert",
        },
        {
          pct: 78,
          time: "03:21:10",
          phase: "WP-04",
          desc: "RTB corridor established, gradual descent profile",
          sev: "INFO",
          sensor: "TAS: 128 kts",
          action: "Recovery",
        },
        {
          pct: 100,
          time: "04:18:00",
          phase: "WP-05",
          desc: "Safe runway recovery & engine shutdown complete",
          sev: "SUCCESS",
          sensor: "RPM: 1,820",
          action: "Log Debrief",
        },
      ],
      telemetryAt: (p) => {
        // p from 0 to 100
        let rpm = 5400,
          chttemp = 180,
          egttemp = 790,
          oilp = 4.2,
          vib = 1.2,
          fuel = Math.max(12, Math.round(100 - p * 0.72));
        if (p < 18) {
          rpm = Math.round(4800 + (p / 18) * 700);
          chttemp = Math.round(150 + (p / 18) * 32);
          egttemp = Math.round(720 + (p / 18) * 60);
          oilp = (4.4 - (p / 18) * 0.2).toFixed(1);
          vib = (1.0 + (p / 18) * 0.3).toFixed(1);
        } else if (p < 56) {
          rpm = 5380 + Math.round(Math.sin(p * 2) * 40);
          chttemp = 182 + Math.round(Math.cos(p) * 4);
          egttemp = 785 + Math.round(Math.sin(p) * 6);
          oilp = (4.2 + Math.sin(p) * 0.1).toFixed(1);
          vib = (1.2 + Math.sin(p * 3) * 0.15).toFixed(1);
        } else if (p < 78) {
          // Anomaly zone
          const ap = (p - 56) / 22;
          rpm = 5350 + Math.round(Math.sin(p * 4) * 90);
          chttemp = Math.min(
            222,
            Math.round(186 + ap * 32 + Math.sin(p * 2) * 5),
          );
          egttemp = Math.min(875, Math.round(790 + ap * 65));
          oilp = Math.max(3.2, (4.1 - ap * 0.7).toFixed(1));
          vib = (1.4 + ap * 2.1 + Math.random() * 0.2).toFixed(1);
        } else {
          // Descent & landing
          const dp = (p - 78) / 22;
          rpm = Math.round(5200 - dp * 3380);
          chttemp = Math.round(214 - dp * 48);
          egttemp = Math.round(840 - dp * 150);
          oilp = (3.5 + dp * 0.5).toFixed(1);
          vib = Math.max(1.1, (3.2 - dp * 2.0).toFixed(1));
        }
        return {
          rpm,
          chttemp,
          egttemp,
          oilp: Number(oilp),
          vib: Number(vib),
          fuel,
        };
      },
    },
    mission_12: {
      id: "MALE-UAV-07",
      name: "Sortie #12: High-Altitude Ingress (Thermal & Oil P Drop)",
      durationStr: "03h 45m",
      totalSeconds: 13500,
      distance: "490 km",
      maxAlt: "16,800 ft",
      avgSpeed: "142 kts",
      status: "THERMAL ALERT",
      statusType: "danger",
      statusSub: "Oil Cooler By-pass Valve Degradation",
      anomalyPct: 52,
      ai: {
        faultType: "Oil Cooler Degradation & Fluid Scavenge Loss",
        confidence: "91.4%",
        healthPct: 59,
        rulHours: "18.2",
        desc: "Oil pressure decay detected concurrently with CHT elevation beyond 215°C at flight level FL165. Scavenge pump volumetric efficiency down 14%.",
        action:
          "Oil Spectral Analysis & Filter Cut (50h) immediately scheduled. Check scavenge pump screen.",
      },
      waypoints: [
        {
          name: "WP-00 (Base)",
          x: 70,
          y: 250,
          lat: "28.6139° N",
          lon: "77.2090° E",
          alt: 500,
          phase: "TAKEOFF",
        },
        {
          name: "WP-01 (Climb)",
          x: 190,
          y: 150,
          lat: "28.8900° N",
          lon: "77.5200° E",
          alt: 14000,
          phase: "RAPID CLIMB",
        },
        {
          name: "WP-02 (Sector H)",
          x: 380,
          y: 70,
          lat: "29.3500° N",
          lon: "78.1000° E",
          alt: 16800,
          phase: "HIGH CRUISE",
        },
        {
          name: "WP-03 (Anomaly)",
          x: 490,
          y: 120,
          lat: "29.2000° N",
          lon: "78.5000° E",
          alt: 16500,
          phase: "THERMAL ALARM",
        },
        {
          name: "WP-04 (Descent)",
          x: 320,
          y: 230,
          lat: "28.8000° N",
          lon: "77.8000° E",
          alt: 9000,
          phase: "EMERGENCY DESCENT",
        },
        {
          name: "WP-05 (Touchdown)",
          x: 75,
          y: 255,
          lat: "28.6145° N",
          lon: "77.2105° E",
          alt: 500,
          phase: "TOUCHDOWN",
        },
      ],
      events: [
        {
          pct: 0,
          time: "00:00:00",
          phase: "WP-00",
          desc: "Climb power set, high altitude sortie profile",
          sev: "INFO",
          sensor: "RPM: 5,500",
          action: "Climb",
        },
        {
          pct: 24,
          time: "00:54:00",
          phase: "WP-01",
          desc: "Crossing FL140, turbocharger wastegate locked at 84%",
          sev: "INFO",
          sensor: "MAP: 38 inHg",
          action: "Cruise",
        },
        {
          pct: 52,
          time: "01:57:00",
          phase: "WP-03",
          desc: "Oil pressure low limit alert (< 2.8 bar) with CHT surge",
          sev: "CRITICAL",
          sensor: "Oil P: 2.7 bar",
          action: "Emergency Throttle",
        },
        {
          pct: 72,
          time: "02:42:00",
          phase: "WP-04",
          desc: "Expedited step-down descent initiated by flight computer",
          sev: "WARNING",
          sensor: "Rate: -1800 fpm",
          action: "Descent",
        },
        {
          pct: 100,
          time: "03:45:00",
          phase: "WP-05",
          desc: "Expedited landing completed without power loss",
          sev: "SUCCESS",
          sensor: "RPM: 1,800",
          action: "Maintenance Hold",
        },
      ],
      telemetryAt: (p) => {
        let rpm = 5500,
          chttemp = 190,
          egttemp = 810,
          oilp = 4.2,
          vib = 1.3,
          fuel = Math.max(18, Math.round(100 - p * 0.8));
        if (p >= 52 && p <= 75) {
          oilp = (2.6 + Math.random() * 0.2).toFixed(1);
          chttemp = Math.min(226, 205 + Math.round((p - 52) * 0.9));
          vib = 2.8;
          rpm = 5100;
        } else if (p > 75) {
          oilp = 3.2;
          chttemp = 175;
          rpm = 3400;
        }
        return {
          rpm,
          chttemp,
          egttemp,
          oilp: Number(oilp),
          vib: Number(vib),
          fuel,
        };
      },
    },
    mission_03: {
      id: "MALE-UAV-04",
      name: "Sortie #03: Boundary Surveillance (Harmonic Vibration Spike)",
      durationStr: "05h 10m",
      totalSeconds: 18600,
      distance: "640 km",
      maxAlt: "13,500 ft",
      avgSpeed: "128 kts",
      status: "VIB WARNING",
      statusType: "warning",
      statusSub: "Propeller Governor / Gearbox Resonance",
      anomalyPct: 60,
      ai: {
        faultType: "Reduction Gearbox Spline Harmonic Vibration",
        confidence: "88.6%",
        healthPct: 74,
        rulHours: "48.0",
        desc: "High-frequency vibration spike recorded across 2.4x fundamental propeller harmonics. Gearbox backlash deviation detected.",
        action:
          "Cylinder Differential Compression & Gearbox Backlash verification at 100h.",
      },
      waypoints: [
        {
          name: "WP-00 (Base)",
          x: 70,
          y: 250,
          lat: "28.6139° N",
          lon: "77.2090° E",
          alt: 500,
          phase: "TAKEOFF",
        },
        {
          name: "WP-01 (Climb)",
          x: 160,
          y: 180,
          lat: "28.7500° N",
          lon: "77.4000° E",
          alt: 11000,
          phase: "CLIMB",
        },
        {
          name: "WP-02 (West Sector)",
          x: 300,
          y: 120,
          lat: "29.1000° N",
          lon: "77.8000° E",
          alt: 13500,
          phase: "BORDER PATROL",
        },
        {
          name: "WP-03 (Turnpoint)",
          x: 440,
          y: 160,
          lat: "28.9500° N",
          lon: "78.2500° E",
          alt: 13500,
          phase: "RESONANCE DETECTED",
        },
        {
          name: "WP-04 (Return)",
          x: 320,
          y: 240,
          lat: "28.7000° N",
          lon: "77.8000° E",
          alt: 7500,
          phase: "RTB",
        },
        {
          name: "WP-05 (Touchdown)",
          x: 75,
          y: 255,
          lat: "28.6145° N",
          lon: "77.2105° E",
          alt: 500,
          phase: "TOUCHDOWN",
        },
      ],
      events: [
        {
          pct: 0,
          time: "00:00:00",
          phase: "WP-00",
          desc: "Sortie launch, automated pre-flight twin calibration passed",
          sev: "INFO",
          sensor: "RPM: 5,350",
          action: "Patrol",
        },
        {
          pct: 60,
          time: "03:06:00",
          phase: "WP-03",
          desc: "Acoustic vibration RMS spiked above 4.1 mm/s at 5,300 RPM",
          sev: "WARNING",
          sensor: "Vib: 4.1 mm/s",
          action: "RPM Shift -150",
        },
        {
          pct: 100,
          time: "05:10:00",
          phase: "WP-05",
          desc: "Mission recovery completed safely",
          sev: "SUCCESS",
          sensor: "RPM: 1,800",
          action: "Routine",
        },
      ],
      telemetryAt: (p) => {
        let rpm = 5300,
          chttemp = 180,
          egttemp = 770,
          oilp = 4.3,
          vib = 1.2,
          fuel = Math.max(10, Math.round(100 - p * 0.65));
        if (p >= 60 && p <= 82) {
          vib = (3.8 + Math.sin(p * 5) * 0.6).toFixed(1);
          rpm = 5150;
        }
        return { rpm, chttemp, egttemp, oilp, vib: Number(vib), fuel };
      },
    },
    mission_01: {
      id: "MALE-UAV-07",
      name: "Sortie #01: Standard Transit (Nominal Baseline)",
      durationStr: "03h 30m",
      totalSeconds: 12600,
      distance: "460 km",
      maxAlt: "12,000 ft",
      avgSpeed: "132 kts",
      status: "ALL SYSTEMS NOMINAL",
      statusType: "success",
      statusSub: "Twin Health Index 99% - Zero Faults",
      anomalyPct: 999, // No anomaly
      ai: {
        faultType: "None (Nominal Twin Match)",
        confidence: "99.1%",
        healthPct: 98,
        rulHours: "142.0",
        desc: "All physical sensors matched Digital Twin aero-thermal boundary estimates within ±1.2% tolerance. No mechanical wear acceleration identified.",
        action:
          "Follow standard maintenance schedule. Next: Oil Spectral Analysis at 50h.",
      },
      waypoints: [
        {
          name: "WP-00 (Base)",
          x: 70,
          y: 250,
          lat: "28.6139° N",
          lon: "77.2090° E",
          alt: 500,
          phase: "TAKEOFF",
        },
        {
          name: "WP-01 (Climb)",
          x: 180,
          y: 170,
          lat: "28.8000° N",
          lon: "77.4500° E",
          alt: 10000,
          phase: "CLIMB",
        },
        {
          name: "WP-02 (Waypoint Alpha)",
          x: 320,
          y: 110,
          lat: "29.1500° N",
          lon: "77.9000° E",
          alt: 12000,
          phase: "TRANSIT",
        },
        {
          name: "WP-03 (Waypoint Bravo)",
          x: 460,
          y: 150,
          lat: "29.0500° N",
          lon: "78.3500° E",
          alt: 12000,
          phase: "TRANSIT",
        },
        {
          name: "WP-04 (Waypoint Charlie)",
          x: 350,
          y: 240,
          lat: "28.7500° N",
          lon: "77.9000° E",
          alt: 7000,
          phase: "DESCENT",
        },
        {
          name: "WP-05 (Touchdown)",
          x: 75,
          y: 255,
          lat: "28.6145° N",
          lon: "77.2105° E",
          alt: 500,
          phase: "TOUCHDOWN",
        },
      ],
      events: [
        {
          pct: 0,
          time: "00:00:00",
          phase: "WP-00",
          desc: "Standard sortie start, digital twin sync confirmed",
          sev: "INFO",
          sensor: "RPM: 5,400",
          action: "Nominal",
        },
        {
          pct: 45,
          time: "01:34:30",
          phase: "WP-02",
          desc: "Midpoint transit check, fuel burn within 1% of predicted model",
          sev: "INFO",
          sensor: "EGT: 780°C",
          action: "Nominal",
        },
        {
          pct: 100,
          time: "03:30:00",
          phase: "WP-05",
          desc: "Nominal landing, zero advisories logged",
          sev: "SUCCESS",
          sensor: "RPM: 1,810",
          action: "Nominal",
        },
      ],
      telemetryAt: (p) => {
        return {
          rpm: p < 15 ? 5200 : p > 85 ? 3200 : 5400,
          chttemp: p < 15 ? 165 : 180,
          egttemp: 775,
          oilp: 4.3,
          vib: 1.1,
          fuel: Math.max(22, Math.round(100 - p * 0.7)),
        };
      },
    },
  };

  function initMissionReplay() {
    let currentMissionKey = "mission_07";
    let isPlaying = false;
    let playbackSpeed = 1;
    let currentPct = 56; // start at anomaly for immediate user wow-factor
    let animFrameId = null;
    let lastTime = null;

    const missionSelect = $("replay-mission-select");
    const slider = $("replay-slider");
    const playBtn = $("btn-replay-play");
    const iconPlay = $("icon-replay-play");
    const iconPause = $("icon-replay-pause");
    const speedBtns = document.querySelectorAll(".speed-btn");
    const milestoneNodes = document.querySelectorAll(".milestone-node");
    const canvas = $("replay-radar-canvas");
    const btnReport = $("btn-download-report");
    const reportModal = $("report-modal");
    const reportModalClose = $("report-modal-close");
    const reportPrintBtn = $("report-print-btn");
    const reportDownloadBtn = $("report-raw-download-btn");

    if (!slider || !canvas) return; // Not on page or missing

    // Format seconds to HH:MM:SS
    function formatTime(sec) {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    // Load Mission Summary UI
    function loadMission(key) {
      currentMissionKey = key;
      const m = MISSIONS_DATA[key];
      if (!m) return;

      $("rep-uav-id").textContent = m.id;
      $("rep-duration").textContent = m.durationStr;
      $("rep-distance").textContent = m.distance;
      $("rep-max-alt").textContent = m.maxAlt;
      $("rep-avg-speed").textContent = m.avgSpeed;

      const statusText = $("rep-status-text");
      const statusSub = $("rep-status-sub");
      const pulseDot = document.querySelector(".status-card .pulse-dot");
      if (statusText) statusText.textContent = m.status;
      if (statusSub) statusSub.textContent = m.statusSub;
      if (pulseDot) {
        pulseDot.className = `pulse-dot ${m.statusType === "success" ? "" : m.statusType === "danger" ? "danger" : "warning"}`;
      }

      // Populate AI Analysis Card
      $("ai-fault-type").textContent = m.ai.faultType;
      $("ai-confidence").textContent = m.ai.confidence;
      $("ai-health-pct").textContent = `${m.ai.healthPct}% Health`;
      $("ai-health-bar").style.width = `${m.ai.healthPct}%`;
      $("ai-rul-hours").innerHTML =
        `${m.ai.rulHours} <span class="unit">Flight Hours</span>`;
      $("ai-rul-desc").textContent = m.ai.desc;
      $("ai-action-callout").innerHTML =
        `<strong>Recommended Action:</strong> ${m.ai.action}`;

      // Populate Events Log Table
      const tbody = $("replay-events-tbody");
      if (tbody) {
        tbody.innerHTML = "";
        m.events.forEach((ev, idx) => {
          const tr = document.createElement("tr");
          tr.id = `rep-ev-row-${idx}`;
          const sevClass =
            ev.sev === "CRITICAL"
              ? "badge-danger"
              : ev.sev === "WARNING"
                ? "badge-warning"
                : ev.sev === "SUCCESS"
                  ? "badge-success"
                  : "badge-info";
          tr.innerHTML = `
            <td class="font-mono">${ev.time}</td>
            <td style="font-weight:600;">${ev.phase}</td>
            <td>${ev.desc}</td>
            <td><span class="badge ${sevClass}">${ev.sev}</span></td>
            <td class="font-mono">${ev.sensor}</td>
            <td style="color:var(--fg-dim);">${ev.action}</td>
          `;
          tbody.appendChild(tr);
        });
        $("rep-events-count").textContent = `${m.events.length} Events Logged`;
      }

      // Seek to current percentage
      seek(currentPct);
    }

    // Seek / Scrub engine
    function seek(pct) {
      currentPct = Math.max(0, Math.min(100, pct));
      slider.value = currentPct;

      const m = MISSIONS_DATA[currentMissionKey];
      if (!m) return;

      const currentSec = Math.round((currentPct / 100) * m.totalSeconds);
      $("rep-current-time").textContent = formatTime(currentSec);
      $("rep-total-time").textContent = formatTime(m.totalSeconds);
      $("rep-pct-badge").textContent = `${Math.round(currentPct)}%`;

      // Update Telemetry Values
      const tel = m.telemetryAt(currentPct);
      $("rep-tel-rpm").textContent = tel.rpm.toLocaleString();
      $("rep-bar-rpm").style.width =
        `${Math.min(100, Math.max(0, ((tel.rpm - 1800) / 4000) * 100))}%`;

      const tempEl = $("rep-tel-temp");
      tempEl.textContent = `${tel.chttemp}°C`;
      $("rep-tel-egt").textContent = `${tel.egttemp}°C`;
      $("rep-bar-temp").style.width =
        `${Math.min(100, Math.max(0, ((tel.chttemp - 100) / 140) * 100))}%`;
      const tempCard = $("tel-card-temp");
      if (tel.chttemp >= 200) {
        tempCard.className = "card tel-stat-card alert-active";
        tempEl.className = "tel-val font-mono text-danger";
      } else if (tel.chttemp >= 185) {
        tempCard.className = "card tel-stat-card warn-active";
        tempEl.className = "tel-val font-mono text-warning";
      } else {
        tempCard.className = "card tel-stat-card";
        tempEl.className = "tel-val font-mono";
      }

      $("rep-tel-oilp").textContent = tel.oilp.toFixed(1);
      $("rep-bar-oilp").style.width =
        `${Math.min(100, Math.max(0, (tel.oilp / 6.0) * 100))}%`;
      const oilCard = $("tel-card-oilp");
      if (tel.oilp <= 3.0) {
        oilCard.className = "card tel-stat-card alert-active";
      } else {
        oilCard.className = "card tel-stat-card";
      }

      const vibEl = $("rep-tel-vib");
      vibEl.textContent = tel.vib.toFixed(1);
      $("rep-bar-vib").style.width =
        `${Math.min(100, Math.max(0, (tel.vib / 5.0) * 100))}%`;
      const vibCard = $("tel-card-vib");
      if (tel.vib >= 2.5) {
        vibCard.className = "card tel-stat-card alert-active";
        vibEl.className = "tel-val font-mono text-danger";
      } else {
        vibCard.className = "card tel-stat-card";
        vibEl.className = "tel-val font-mono";
      }

      $("rep-tel-fuel").textContent = `${tel.fuel}%`;
      $("rep-bar-fuel").style.width = `${tel.fuel}%`;

      // Highlight corresponding event in table
      m.events.forEach((ev, idx) => {
        const row = $(`rep-ev-row-${idx}`);
        if (!row) return;
        const nextEv = m.events[idx + 1];
        const nextPct = nextEv ? nextEv.pct : 101;
        if (currentPct >= ev.pct && currentPct < nextPct) {
          row.className =
            ev.sev === "CRITICAL" ? "anomaly-event-row" : "active-event-row";
        } else {
          row.className = "";
        }
      });

      // Render 2D Canvas Radar
      renderRadar(m, currentPct);
    }

    // 2D Tactical Vector Radar Canvas Renderer
    function renderRadar(mission, pct) {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.parentElement.clientWidth || 620;
      const h = canvas.parentElement.clientHeight || 310;

      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.save();
      ctx.scale(dpr, dpr);

      // Clear dark background
      ctx.fillStyle = "#050913";
      ctx.fillRect(0, 0, w, h);

      // Draw subtle tactical coordinate grid
      ctx.strokeStyle = "rgba(56, 189, 248, 0.06)";
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Radar Concentric Range Rings centered at Base
      const base = mission.waypoints[0];
      const cx = base.x;
      const cy = base.y;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.12)";
      ctx.lineWidth = 1;
      [80, 160, 260, 380].forEach((r) => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      });

      // Scale waypoints to current canvas aspect
      const scaleX = w / 580;
      const scaleY = h / 310;
      const pts = mission.waypoints.map((wp) => ({
        ...wp,
        px: wp.x * scaleX,
        py: wp.y * scaleY,
      }));

      // Draw Planned Route Path (Glow line)
      ctx.beginPath();
      ctx.moveTo(pts[0].px, pts[0].py);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].px, pts[i].py);
      }
      ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Active flown segment (Solid cyan)
      const totalSegments = pts.length - 1;
      const globalT = (pct / 100) * totalSegments;
      const segIndex = Math.min(Math.floor(globalT), totalSegments - 1);
      const segT = globalT - segIndex;

      ctx.beginPath();
      ctx.moveTo(pts[0].px, pts[0].py);
      for (let i = 1; i <= segIndex; i++) {
        ctx.lineTo(pts[i].px, pts[i].py);
      }
      const curX =
        pts[segIndex].px + (pts[segIndex + 1].px - pts[segIndex].px) * segT;
      const curY =
        pts[segIndex].py + (pts[segIndex + 1].py - pts[segIndex].py) * segT;
      ctx.lineTo(curX, curY);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(56, 189, 248, 0.8)";
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0; // reset

      // Draw Waypoint nodes
      pts.forEach((pt, i) => {
        const isAnomalyWp = i === 3 && mission.anomalyPct < 100;
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = isAnomalyWp ? "#ef4444" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Node label
        ctx.font = "9.5px 'JetBrains Mono', monospace";
        ctx.fillStyle = isAnomalyWp ? "#fca5a5" : "rgba(255, 255, 255, 0.75)";
        ctx.fillText(pt.name, pt.px + 8, pt.py - 6);
      });

      // Draw Anomaly Pulsing Sector if past trigger
      if (pct >= mission.anomalyPct && mission.anomalyPct < 100) {
        const anWp = pts[3];
        const pulseR = 14 + Math.sin(Date.now() / 150) * 6;
        ctx.beginPath();
        ctx.arc(anWp.px, anWp.py, pulseR, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "rgba(239, 68, 68, 0.2)";
        ctx.fill();

        // Anomaly Tag
        ctx.font = "bold 9px sans-serif";
        ctx.fillStyle = "#ef4444";
        ctx.fillText("⚠️ CHT FAULT (t = 02:24)", anWp.px - 45, anWp.py + 24);
      }

      // Calculate Drone Heading Angle
      const dx = pts[segIndex + 1].px - pts[segIndex].px;
      const dy = pts[segIndex + 1].py - pts[segIndex].py;
      const angle = Math.atan2(dy, dx);
      const headingDeg = Math.round(((angle * 180) / Math.PI + 360) % 360);

      // Update HUD Overlay text
      const hudHeading = $("hud-heading");
      const hudCoords = $("hud-coords");
      const hudPhase = $("hud-phase");
      if (hudHeading)
        hudHeading.textContent = `${String(headingDeg).padStart(3, "0")}°`;
      if (hudCoords)
        hudCoords.textContent = pts[segIndex].lat + ", " + pts[segIndex].lon;
      if (hudPhase) hudPhase.textContent = pts[segIndex].phase;

      // Draw Drone Icon at (curX, curY)
      ctx.save();
      ctx.translate(curX, curY);
      ctx.rotate(angle);

      // Drone shadow/glow ring
      ctx.beginPath();
      ctx.arc(0, 0, 12, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
      ctx.fill();

      // Drone Chevron Silhouette
      ctx.beginPath();
      ctx.moveTo(10, 0);
      ctx.lineTo(-8, -6);
      ctx.lineTo(-4, 0);
      ctx.lineTo(-8, 6);
      ctx.closePath();
      ctx.fillStyle = "#10b981";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.restore();
      ctx.restore();
    }

    // Playback loop
    function step(timestamp) {
      if (!lastTime) lastTime = timestamp;
      const dt = (timestamp - lastTime) / 1000;
      lastTime = timestamp;

      if (isPlaying) {
        // Full run takes ~40 seconds at 1x
        const advance = (dt / 40) * 100 * playbackSpeed;
        currentPct += advance;
        if (currentPct >= 100) {
          currentPct = 100;
          pause();
        }
        seek(currentPct);
      }
      if (isPlaying) {
        animFrameId = requestAnimationFrame(step);
      }
    }

    function play() {
      isPlaying = true;
      if (currentPct >= 100) currentPct = 0;
      iconPlay.style.display = "none";
      iconPause.style.display = "inline";
      lastTime = null;
      animFrameId = requestAnimationFrame(step);
    }

    function pause() {
      isPlaying = false;
      iconPlay.style.display = "inline";
      iconPause.style.display = "none";
      if (animFrameId) cancelAnimationFrame(animFrameId);
    }

    // Event Listeners
    playBtn.addEventListener("click", () => {
      if (isPlaying) pause();
      else play();
    });

    slider.addEventListener("input", (e) => {
      pause();
      seek(parseFloat(e.target.value));
    });

    speedBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        speedBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        playbackSpeed = parseFloat(btn.dataset.speed || 1);
      });
    });

    milestoneNodes.forEach((node) => {
      node.addEventListener("click", () => {
        pause();
        const timePct = parseFloat(node.dataset.time || 0);
        seek(timePct);
      });
    });

    missionSelect.addEventListener("change", (e) => {
      pause();
      currentPct = 0;
      loadMission(e.target.value);
    });

    // Mission Report Modal Generator
    function generateReportHTML(mission) {
      return `
        <div class="report-sheet">
          <div class="rep-header">
            <div class="rep-title">
              <h1>UAV Mission Debrief & Engine Health Certificate</h1>
              <p>MALE Uav Piston Digital Twin Post-Flight Audit | Sortie Ref: ${mission.id} / ${mission.name.split(":")[0]}</p>
            </div>
            <div class="rep-badge">STATUS: ${mission.status}</div>
          </div>

          <div class="rep-grid-2">
            <div class="rep-box">
              <h4>Mission Profile Summary</h4>
              <div class="rep-meta-row"><span class="k">Airframe & Engine:</span> <span class="v">${mission.id} (Rotax 914 Twin-Turbo)</span></div>
              <div class="rep-meta-row"><span class="k">Total Flight Duration:</span> <span class="v">${mission.durationStr}</span></div>
              <div class="rep-meta-row"><span class="k">Distance Traversed:</span> <span class="v">${mission.distance}</span></div>
              <div class="rep-meta-row"><span class="k">Peak Altitude:</span> <span class="v">${mission.maxAlt}</span></div>
              <div class="rep-meta-row"><span class="k">Average TAS:</span> <span class="v">${mission.avgSpeed}</span></div>
            </div>

            <div class="rep-box">
              <h4>Digital Twin AI Diagnostics</h4>
              <div class="rep-meta-row"><span class="k">Localized Fault:</span> <span class="v" style="color:#b91c1c;">${mission.ai.faultType}</span></div>
              <div class="rep-meta-row"><span class="k">Model Confidence:</span> <span class="v">${mission.ai.confidence}</span></div>
              <div class="rep-meta-row"><span class="k">Engine Health Degradation:</span> <span class="v">${mission.ai.healthPct}% Nominal</span></div>
              <div class="rep-meta-row"><span class="k">Remaining Useful Life (RUL):</span> <span class="v">${mission.ai.rulHours} Flight Hours</span></div>
              <div class="rep-meta-row"><span class="k">Anomaly Trigger Point:</span> <span class="v">t = ${mission.anomalyPct < 100 ? "02h 24m (WP-03)" : "None"}</span></div>
            </div>
          </div>

          <div class="rep-diag-alert">
            <strong>AI Engineering Assessment & Digital Twin Residuals:</strong>
            ${mission.ai.desc}
          </div>

          <div class="rep-section-title">Chronological Flight Log & Severity Classification</div>
          <table class="rep-table">
            <thead>
              <tr>
                <th>Time (UTC)</th>
                <th>Phase / Waypoint</th>
                <th>Operational Event</th>
                <th>Severity</th>
                <th>Sensor Telemetry Snapshot</th>
              </tr>
            </thead>
            <tbody>
              ${mission.events
                .map(
                  (e) => `
                <tr>
                  <td>${e.time}</td>
                  <td><strong>${e.phase}</strong></td>
                  <td>${e.desc}</td>
                  <td><span style="font-weight:700; color:${e.sev === "CRITICAL" ? "#dc2626" : e.sev === "WARNING" ? "#d97706" : "#059669"}">${e.sev}</span></td>
                  <td><code>${e.sensor}</code></td>
                </tr>
              `,
                )
                .join("")}
            </tbody>
          </table>

          <div class="rep-section-title">Mandatory Maintenance Directive</div>
          <div style="background:#f8fafc; border:1px solid #e2e8f0; padding:12px; border-radius:6px; font-size:11.5px; color:#1e293b;">
            <strong>Immediate Action:</strong> ${mission.ai.action}<br>
            <strong>Scheduled Intervals:</strong> Spark Plugs Gap (100h) · Cylinder Differential Compression (100h) · Oil Filter Cut (50h).
          </div>

          <div class="rep-signatures">
            <div class="rep-sig-box">
              <strong>Chief Flight Operations Officer</strong><br>
              Digital Twin Autonomous Telemetry Verified
            </div>
            <div class="rep-sig-box">
              <strong>Lead Propulsion Maintenance Engineer</strong><br>
              Sortie Airworthiness Review & Release
            </div>
          </div>
        </div>
      `;
    }

    btnReport.addEventListener("click", () => {
      const m = MISSIONS_DATA[currentMissionKey];
      $("report-modal-body").innerHTML = generateReportHTML(m);
      reportModal.style.display = "flex";
    });

    reportModalClose.addEventListener("click", () => {
      reportModal.style.display = "none";
    });

    reportPrintBtn.addEventListener("click", () => {
      window.print();
    });

    reportDownloadBtn.addEventListener("click", () => {
      const m = MISSIONS_DATA[currentMissionKey];
      const htmlContent = `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Mission_Report_${m.id}.html</title><style>body{font-family:sans-serif;margin:20px;background:#f1f5f9;color:#0f172a;} .report-sheet{background:#fff;padding:30px;border-radius:8px;max-width:860px;margin:auto;} table{width:100%;border-collapse:collapse;margin:16px 0;} th,td{border:1px solid #cbd5e1;padding:6px 8px;font-size:12px;text-align:left;} th{background:#e2e8f0;}</style></head><body>${generateReportHTML(m)}</body></html>`;
      const blob = new Blob([htmlContent], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Mission_Debrief_${m.id}_${currentMissionKey}.html`;
      a.click();
      URL.revokeObjectURL(url);
    });

    // Window resize handler for radar canvas
    window.addEventListener("resize", () => {
      const m = MISSIONS_DATA[currentMissionKey];
      if (m) renderRadar(m, currentPct);
    });

    // Initial load
    loadMission("mission_07");
  }

  // ------------------------------------------------------------------
  // SECTION: GARUD-DT ALERT MANAGEMENT & AI DIAGNOSTICS ENGINE
  // ------------------------------------------------------------------
  let garudAlerts = [
    {
      id: "alt-01",
      time: "09:41:17",
      uav: "UAV-01",
      parameter: "Vibration",
      alertName: "High Vibration Detected",
      currentVal: "1.6 mm/s",
      normalRange: "< 1.0 mm/s",
      severity: "WARNING",
      status: "Active",
      duration: "2 min 14 sec",
      affectedParam: "Acoustic Vibration (RMS)",
      aiAssessment: "Possible excessive mechanical vibration. Spectral peak localized at 2.4x propeller order harmonics.",
      faultProbability: "91%",
      engineHealth: 78,
      trend: "Increasing ↗",
      confidence: "High (Ensemble)",
      recommendation: "Inspect relevant mechanical/rotating components after mission. Verify propeller governor backlash at 100h interval."
    },
    {
      id: "alt-02",
      time: "09:43:02",
      uav: "UAV-01",
      parameter: "CHT",
      alertName: "High Cylinder Temperature",
      currentVal: "186°C",
      normalRange: "< 180°C",
      severity: "CRITICAL",
      status: "Active",
      duration: "1 min 08 sec",
      affectedParam: "Cylinder #3 Head Temperature",
      aiAssessment: "Exhaust valve thermal buildup detected. Digital Twin thermal boundary exceeded with acoustic residual correlation.",
      faultProbability: "96%",
      engineHealth: 64,
      trend: "Accelerating ⇈",
      confidence: "Very High (Physics-Informed)",
      recommendation: "Follow approved UAV emergency/maintenance procedure. Step-down throttle to 65% and prepare for optical borescope inspection."
    },
    {
      id: "alt-03",
      time: "09:45:11",
      uav: "UAV-01",
      parameter: "Oil Pressure",
      alertName: "Low Lubrication Pressure",
      currentVal: "1.4 bar",
      normalRange: "> 2.0 bar",
      severity: "WARNING",
      status: "Resolved",
      duration: "45 sec",
      affectedParam: "Main Gallery Lubrication Pressure",
      aiAssessment: "Transient pressure drop during rapid altitude climb step. Pressure stabilized after scavenge valve re-seat.",
      faultProbability: "42%",
      engineHealth: 88,
      trend: "Stabilizing →",
      confidence: "High",
      recommendation: "Oil Spectral Analysis & Filter Cut (50h) scheduled. Monitor pressure transient response."
    }
  ];

  let garudHistory = [
    {
      timestamp: "Today, 08:30:15",
      timeFilter: "today",
      uav: "UAV-01",
      parameter: "Fuel Flow",
      classification: "Mixture Imbalance",
      peakVal: "38.4 L/h",
      severity: "WARNING",
      status: "Resolved",
      faultType: "Degradation",
      rootCause: "Auto-mixture servo trim calibration adjusted +2.5% during cruise."
    },
    {
      timestamp: "Today, 07:12:44",
      timeFilter: "today",
      uav: "UAV-04",
      parameter: "RPM",
      classification: "Governor Overshoot",
      peakVal: "5,840 RPM",
      severity: "CRITICAL",
      status: "Resolved",
      faultType: "Vibration",
      rootCause: "Transient overshoot on rapid climb command. PID governor gains verified."
    },
    {
      timestamp: "Yesterday, 18:22:10",
      timeFilter: "7days",
      uav: "UAV-07",
      parameter: "EGT",
      classification: "Exhaust Thermal Spike",
      peakVal: "845°C",
      severity: "WARNING",
      status: "Resolved",
      faultType: "Thermal",
      rootCause: "Thermal soak during extended maximum endurance loiter. Fuel cooling profile applied."
    },
    {
      timestamp: "3 days ago, 14:10",
      timeFilter: "7days",
      uav: "UAV-01",
      parameter: "Vibration",
      classification: "Harmonic Resonance",
      peakVal: "2.8 mm/s",
      severity: "WARNING",
      status: "Resolved",
      faultType: "Vibration",
      rootCause: "Propeller pitch governor spline clearance checked and retorqued."
    },
    {
      timestamp: "5 days ago, 11:05",
      timeFilter: "7days",
      uav: "UAV-04",
      parameter: "Oil Temp",
      classification: "Cooler Bypass Sticking",
      peakVal: "128°C",
      severity: "CRITICAL",
      status: "Resolved",
      faultType: "Lubrication",
      rootCause: "Replaced thermostatic bypass valve assembly during 100h scheduled servicing."
    },
    {
      timestamp: "18 days ago, 09:40",
      timeFilter: "30days",
      uav: "UAV-07",
      parameter: "Manifold Pressure",
      classification: "Wastegate Duty Lag",
      peakVal: "36.2 inHg",
      severity: "WARNING",
      status: "Resolved",
      faultType: "Degradation",
      rootCause: "Turbocharger actuator arm linkage cleaned and lubricated."
    }
  ];

  let selectedAlertId = "alt-01";

  function initGarudAlerts() {
    renderAlertsSummary();
    renderActiveAlertsTable();
    selectAlert(selectedAlertId);
    renderHistoryTable();

    const btnAck = $("btn-ack-selected");
    const btnResolve = $("btn-resolve-selected");
    const btnAckAll = $("btn-ack-all-alerts");
    const btnClearResolved = $("btn-clear-resolved-alerts");

    if (btnAck) {
      btnAck.addEventListener("click", () => {
        acknowledgeAlert(selectedAlertId);
      });
    }

    if (btnResolve) {
      btnResolve.addEventListener("click", () => {
        resolveAlert(selectedAlertId);
      });
    }

    if (btnAckAll) {
      btnAckAll.addEventListener("click", () => {
        garudAlerts.forEach((a) => {
          if (a.status === "Active") a.status = "Acknowledged";
        });
        renderAlertsSummary();
        renderActiveAlertsTable();
        selectAlert(selectedAlertId);
      });
    }

    if (btnClearResolved) {
      btnClearResolved.addEventListener("click", () => {
        const resolved = garudAlerts.filter((a) => a.status === "Resolved");
        resolved.forEach((r) => {
          garudHistory.unshift({
            timestamp: `Today, ${r.time}`,
            timeFilter: "today",
            uav: r.uav,
            parameter: r.parameter,
            classification: r.alertName,
            peakVal: r.currentVal,
            severity: r.severity,
            status: "Resolved",
            faultType: r.parameter.includes("Vib")
              ? "Vibration"
              : r.parameter.includes("Temp") || r.parameter.includes("CHT")
                ? "Thermal"
                : "Lubrication",
            rootCause: r.recommendation,
          });
        });
        garudAlerts = garudAlerts.filter((a) => a.status !== "Resolved");
        if (
          !garudAlerts.find((a) => a.id === selectedAlertId) &&
          garudAlerts.length > 0
        ) {
          selectedAlertId = garudAlerts[0].id;
        }
        renderAlertsSummary();
        renderActiveAlertsTable();
        selectAlert(selectedAlertId);
        renderHistoryTable();
      });
    }

    const timeFilterBtns = document.querySelectorAll(
      ".alert-history-filter-bar .filter-btn",
    );
    timeFilterBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        timeFilterBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderHistoryTable();
      });
    });

    [
      "filter-uav-select",
      "filter-severity-select",
      "filter-type-select",
    ].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", renderHistoryTable);
    });
  }

  function renderAlertsSummary() {
    const critCount = garudAlerts.filter(
      (a) => a.severity === "CRITICAL" && a.status !== "Resolved",
    ).length;
    const warnCount = garudAlerts.filter(
      (a) => a.severity === "WARNING" && a.status !== "Resolved",
    ).length;
    const resolvedCount =
      garudAlerts.filter((a) => a.status === "Resolved").length +
      garudHistory.length;
    const totalCount = garudAlerts.length + garudHistory.length;

    const critEl = $("alert-kpi-critical");
    const warnEl = $("alert-kpi-warning");
    const resEl = $("alert-kpi-resolved");
    const totEl = $("alert-kpi-total");

    if (critEl) critEl.textContent = critCount;
    if (warnEl) warnEl.textContent = warnCount;
    if (resEl) resEl.textContent = resolvedCount;
    if (totEl) totEl.textContent = totalCount;

    const badgeCrit = $("badge-critical-alerts");
    const badgeWarn = $("badge-warning-alerts");
    const activeCountEl = $("active-alerts-count");
    const navBadge = $("nav-alerts-badge");

    if (badgeCrit) {
      badgeCrit.textContent = `${critCount} CRITICAL`;
      badgeCrit.style.display = critCount > 0 ? "inline-block" : "none";
    }
    if (badgeWarn) {
      badgeWarn.textContent = `${warnCount} WARNING`;
      badgeWarn.style.display = warnCount > 0 ? "inline-block" : "none";
    }
    if (activeCountEl) {
      activeCountEl.textContent = `${critCount + warnCount} Active Conditions`;
    }
    if (navBadge) {
      navBadge.textContent = `${critCount + warnCount} ACT`;
    }
  }

  function renderActiveAlertsTable() {
    const tbody = $("active-alerts-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (garudAlerts.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="insufficient">Zero active alerts. All propulsion parameters nominal.</td></tr>`;
      return;
    }

    garudAlerts.forEach((alert) => {
      const tr = document.createElement("tr");
      tr.id = `alert-row-${alert.id}`;
      if (alert.id === selectedAlertId) {
        tr.className = `selected-alert-row ${alert.severity === "CRITICAL" ? "is-critical" : ""}`;
      }

      const isCrit = alert.severity === "CRITICAL";
      const sevBadge = isCrit
        ? `<span class="badge badge-danger">🔴 Critical</span>`
        : `<span class="badge badge-warning">⚠️ Warning</span>`;

      const statusBadge =
        alert.status === "Resolved"
          ? `<span class="badge badge-success">Resolved</span>`
          : alert.status === "Acknowledged"
            ? `<span class="badge badge-info">Acknowledged</span>`
            : `<span class="badge badge-warning">Active</span>`;

      tr.innerHTML = `
        <td class="font-mono">${alert.time}</td>
        <td><strong>${alert.uav}</strong></td>
        <td>${alert.parameter}</td>
        <td style="font-weight:600;">${alert.alertName}</td>
        <td class="font-mono ${isCrit ? "text-danger" : "text-warning"}">${alert.currentVal}</td>
        <td>${sevBadge}</td>
        <td>${statusBadge}</td>
        <td>
          <button class="action-btn-pill" onclick="event.stopPropagation(); window.__resolveAlertById('${alert.id}')" ${alert.status === "Resolved" ? "disabled" : ""}>
            ${alert.status === "Resolved" ? "Done" : "Resolve"}
          </button>
        </td>
      `;

      tr.addEventListener("click", () => {
        selectAlert(alert.id);
      });

      tbody.appendChild(tr);
    });
  }

  function selectAlert(id) {
    selectedAlertId = id;
    const alert = garudAlerts.find((a) => a.id === id) || garudAlerts[0];
    if (!alert) return;

    const rows = document.querySelectorAll("#active-alerts-tbody tr");
    rows.forEach((r) => {
      const isSel = r.id === `alert-row-${alert.id}`;
      r.className = isSel
        ? `selected-alert-row ${alert.severity === "CRITICAL" ? "is-critical" : ""}`
        : "";
    });

    const titleEl = $("detail-alert-title");
    const elapsedEl = $("detail-alert-elapsed");
    const uavEl = $("detail-uav-id");
    const timeEl = $("detail-time");
    const valEl = $("detail-current-val");
    const rangeEl = $("detail-normal-range");
    const sevEl = $("detail-severity");
    const paramEl = $("detail-parameter");
    const badgeEl = $("detail-card-badge");

    if (titleEl) titleEl.textContent = alert.alertName;
    if (elapsedEl) elapsedEl.textContent = `Duration: ${alert.duration}`;
    if (uavEl) uavEl.textContent = alert.uav;
    if (timeEl) timeEl.textContent = `${alert.time} UTC`;
    if (valEl) {
      valEl.textContent = alert.currentVal;
      valEl.className =
        alert.severity === "CRITICAL"
          ? "detail-v font-mono text-danger"
          : "detail-v font-mono text-warning";
    }
    if (rangeEl) rangeEl.textContent = alert.normalRange;
    if (sevEl) {
      sevEl.innerHTML =
        alert.severity === "CRITICAL"
          ? `<span class="badge badge-danger">CRITICAL</span>`
          : `<span class="badge badge-warning">WARNING</span>`;
    }
    if (paramEl) paramEl.textContent = alert.affectedParam;
    if (badgeEl) {
      badgeEl.textContent = alert.severity;
      badgeEl.className =
        alert.severity === "CRITICAL"
          ? "badge badge-danger"
          : "badge badge-warning";
    }

    const btnAck = $("btn-ack-selected");
    const btnRes = $("btn-resolve-selected");
    if (btnAck) {
      if (alert.status === "Acknowledged") {
        btnAck.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> <span>Acknowledged ✓</span>`;
        btnAck.disabled = true;
      } else if (alert.status === "Resolved") {
        btnAck.innerHTML = `<span>Acknowledged ✓</span>`;
        btnAck.disabled = true;
      } else {
        btnAck.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> <span>Acknowledge Alert</span>`;
        btnAck.disabled = false;
      }
    }
    if (btnRes) {
      if (alert.status === "Resolved") {
        btnRes.innerHTML = `<span>Resolved ✓</span>`;
        btnRes.disabled = true;
      } else {
        btnRes.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> <span>Resolve Condition</span>`;
        btnRes.disabled = false;
      }
    }

    const quoteEl = $("ai-assessment-quote");
    const probEl = $("ai-fault-prob");
    const healthEl = $("ai-engine-health");
    const meterEl = $("ai-health-meter");
    const trendEl = $("ai-fault-trend");
    const confEl = $("ai-confidence-val");
    const recEl = $("ai-recommended-action");
    const recPanel = $("ai-recommendation-panel");

    if (quoteEl) quoteEl.textContent = `"${alert.aiAssessment}"`;
    if (probEl) probEl.textContent = alert.faultProbability;
    if (healthEl) healthEl.textContent = `${alert.engineHealth} / 100`;
    if (meterEl) {
      meterEl.style.width = `${alert.engineHealth}%`;
      meterEl.className =
        alert.engineHealth < 70
          ? "stat-pod-fill fill-danger"
          : "stat-pod-fill fill-warning";
    }
    if (trendEl) trendEl.textContent = alert.trend;
    if (confEl) confEl.textContent = alert.confidence;
    if (recEl) recEl.textContent = alert.recommendation;
    if (recPanel) {
      recPanel.className =
        alert.severity === "CRITICAL"
          ? "ai-recommendation-panel rec-critical"
          : "ai-recommendation-panel";
    }
  }

  function acknowledgeAlert(id) {
    const alert = garudAlerts.find((a) => a.id === id);
    if (!alert) return;
    alert.status = "Acknowledged";
    renderAlertsSummary();
    renderActiveAlertsTable();
    selectAlert(id);
  }

  function resolveAlert(id) {
    const alert = garudAlerts.find((a) => a.id === id);
    if (!alert) return;
    alert.status = "Resolved";
    renderAlertsSummary();
    renderActiveAlertsTable();
    selectAlert(id);
  }

  window.__resolveAlertById = (id) => {
    resolveAlert(id);
  };

  function renderHistoryTable() {
    const tbody = $("history-alerts-tbody");
    if (!tbody) return;

    const activeTimeBtn = document.querySelector(
      ".alert-history-filter-bar .filter-btn.active",
    );
    const timeFilter = activeTimeBtn
      ? activeTimeBtn.dataset.timeFilter
      : "today";
    const uavFilter = $("filter-uav-select")
      ? $("filter-uav-select").value
      : "all";
    const sevFilter = $("filter-severity-select")
      ? $("filter-severity-select").value
      : "all";
    const typeFilter = $("filter-type-select")
      ? $("filter-type-select").value
      : "all";

    const filtered = garudHistory.filter((item) => {
      if (timeFilter === "today" && item.timeFilter !== "today") return false;
      if (timeFilter === "7days" && item.timeFilter === "30days") return false;
      if (uavFilter !== "all" && item.uav !== uavFilter) return false;
      if (sevFilter !== "all" && item.severity !== sevFilter) return false;
      if (typeFilter !== "all" && item.faultType !== typeFilter) return false;
      return true;
    });

    tbody.innerHTML = "";
    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="insufficient">No historical alerts match the active filter criteria.</td></tr>`;
      return;
    }

    filtered.forEach((h) => {
      const tr = document.createElement("tr");
      const isCrit = h.severity === "CRITICAL";
      tr.innerHTML = `
        <td class="font-mono">${h.timestamp}</td>
        <td><strong>${h.uav}</strong></td>
        <td>${h.parameter}</td>
        <td>${h.classification}</td>
        <td class="font-mono">${h.peakVal}</td>
        <td><span class="badge ${isCrit ? "badge-danger" : "badge-warning"}">${h.severity}</span></td>
        <td><span class="badge badge-success">${h.status}</span></td>
        <td style="color:var(--fg-dim); font-size:11px;">${h.rootCause}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // -------- Boot -------------------------------------------------------
  initViewNavigation();
  initCharts();
  initMissionReplay();
  initGarudAlerts();
  loadScenarios();
  // Seed the time-series from the history endpoint so the
  // charts aren't empty on first load.
  fetch("/api/history?limit=" + MAX_POINTS)
    .then((r) => r.json())
    .then((j) => {
      (j.snapshots || []).forEach((s) => update(s));
    })
    .catch(() => {});
  connect();
})();
