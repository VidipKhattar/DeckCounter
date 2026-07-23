(function () {
  const statusLabel = document.getElementById("status-label");
  const guideBox = document.getElementById("guide-box");
  const stopButton = document.getElementById("cta-stop");
  const resetButton = document.getElementById("cta-action-reset");
  const resultsSection = document.getElementById("results-section");
  const capturedHeading = document.getElementById("captured-heading");
  const capturedTokens = document.getElementById("captured-tokens");
  const missingHeading = document.getElementById("missing-heading");
  const missingTokens = document.getElementById("missing-tokens");
  const metricAccuracy = document.getElementById("metric-accuracy");
  const metricSamples = document.getElementById("metric-samples");
  const metricLatency = document.getElementById("metric-latency");
  const onboardingOverlay = document.getElementById("onboarding-overlay");
  const onboardingDismiss = document.getElementById("onboarding-dismiss");
  const settingsToggle = document.getElementById("settings-toggle");
  const settingsPanel = document.getElementById("settings-panel");
  const settingsApply = document.getElementById("settings-apply");
  const settingConfidence = document.getElementById("setting-confidence");
  const settingMaybeThreshold = document.getElementById("setting-maybe-threshold");
  const settingMinConsensus = document.getElementById("setting-min-consensus");
  const settingPresenceConfidence = document.getElementById("setting-presence-confidence");
  const settingAugment = document.getElementById("setting-augment");
  const cardPreviewModal = document.getElementById("card-preview-modal");
  const cardPreviewImage = document.getElementById("card-preview-image");
  const cardPreviewClose = document.getElementById("card-preview-close");
  const videoFeed = document.getElementById("video-feed");
  const captureCanvas = document.getElementById("capture-canvas");
  const cameraContainer = document.getElementById("camera-container");
  const recordingBadge = document.getElementById("recording-badge");
  const analyzingOverlay = document.getElementById("analyzing-overlay");
  const analyzingPercent = document.getElementById("analyzing-percent");
  const maybeBlock = document.getElementById("maybe-block");
  const maybeHeading = document.getElementById("maybe-heading");
  const maybeTokens = document.getElementById("maybe-tokens");
  const cameraSwitchButton = document.getElementById("camera-switch");

  const SUIT_NAMES = { S: "Spades", H: "Hearts", D: "Diamonds", C: "Clubs" };
  // Send frames far more often while recording so a card that only flashes past
  // for a fraction of a second still lands in several frames (more chances to
  // catch a sharp one between motion blur). Recording just buffers frames with
  // no inference, so a high rate here is cheap.
  const FRAME_INTERVAL_MS = 55;
  const JPEG_QUALITY = 0.92;

  let activeSocket = null;

  function describeCard(card) {
    const suit = card.slice(-1);
    const rank = card.slice(0, -1);
    return `${rank} of ${SUIT_NAMES[suit] || suit}`;
  }

  function setState(type) {
    guideBox.classList.toggle("recording", type === "recording");
    cameraContainer.classList.toggle("is-recording", type === "recording");
    recordingBadge.classList.toggle("show", type === "recording");
    stopButton.classList.toggle("hidden", type !== "recording");
    analyzingOverlay.classList.toggle("show", type === "processing");
    // The full-screen overlay covers everything during processing, so the
    // small on-video label is hidden entirely rather than duplicating text
    // on top of the camera feed.
    statusLabel.classList.toggle("hidden", type === "processing");
    resultsSection.style.display = type === "done" ? "grid" : "none";
    statusLabel.classList.toggle("status-done", type === "done");
    statusLabel.classList.toggle("status-recording", type === "recording");

    if (type === "waiting") {
      statusLabel.textContent = "Place a card";
    } else if (type === "recording") {
      statusLabel.textContent = "Recording — flip through the deck";
    } else if (type === "done") {
      statusLabel.textContent = "Analysis complete";
    } else if (type === "error") {
      statusLabel.textContent = "Camera unavailable";
    }
  }

  function renderTokenGrid(container, cards, extraClass) {
    container.innerHTML = "";
    cards.forEach((card) => {
      const span = document.createElement("span");
      span.className = `token token-captured ${extraClass}`.trim();
      span.textContent = card;
      span.title = `View the frame ${describeCard(card)} was recognized in`;
      span.addEventListener("click", () => openCardPreview(card));
      container.appendChild(span);
    });
  }

  function openCardPreview(card) {
    cardPreviewImage.src = `/api/card/${card}/frame?t=${Date.now()}`;
    cardPreviewImage.alt = `Best captured frame for ${describeCard(card)}`;
    cardPreviewModal.classList.add("open");
  }

  function closeCardPreview() {
    cardPreviewModal.classList.remove("open");
    cardPreviewImage.src = "";
  }

  function renderMissing(cards) {
    missingTokens.innerHTML = "";
    cards.forEach((card) => {
      const wrap = document.createElement("div");
      wrap.className = "flex flex-col gap-1.5";

      const value = document.createElement("span");
      value.className = "text-4xl font-light tracking-tight text-[#8b5a5a]";
      value.textContent = card;

      const label = document.createElement("span");
      label.className = "text-[9px] uppercase tracking-widest text-zinc-500 font-bold";
      label.textContent = describeCard(card);

      wrap.appendChild(value);
      wrap.appendChild(label);
      missingTokens.appendChild(wrap);
    });
  }

  function handleMessage(msg) {
    if (msg.type === "settings") {
      applySettingsToForm(msg);
      return;
    }

    setState(msg.type);

    if (msg.type === "done") {
      const maybe = msg.maybe || [];
      missingHeading.textContent = `Critical: Missing Components (${msg.missing.length})`;
      renderMissing(msg.missing);
      maybeBlock.style.display = maybe.length ? "block" : "none";
      maybeHeading.textContent = `Uncertain — Worth a Look (${maybe.length})`;
      renderTokenGrid(maybeTokens, maybe, "token-maybe");
      capturedHeading.textContent = `Detected Inventory (${msg.seen.length})`;
      renderTokenGrid(capturedTokens, msg.seen, "");
      metricAccuracy.textContent = `${msg.metrics.reliability}%`;
      metricSamples.textContent = msg.metrics.frame_count;
      metricLatency.textContent = `${msg.metrics.latency_ms}ms`;
    } else if (msg.type === "recording") {
      statusLabel.textContent = `Recording — ${msg.frame_count} frames captured`;
    } else if (msg.type === "processing") {
      // No text on the video itself during processing — the full-screen
      // analyzing overlay (opaque, off the camera) is the single source of
      // truth for progress now, so it always reads clearly regardless of
      // what's behind the camera feed.
      const pct = msg.total ? Math.round((msg.current / msg.total) * 100) : 0;
      analyzingPercent.textContent = `${pct}%`;
    }
  }

  function applySettingsToForm(values) {
    settingConfidence.value = values.confidence;
    settingMaybeThreshold.value = values.maybe_threshold;
    settingMinConsensus.value = values.min_consensus;
    settingPresenceConfidence.value = values.presence_confidence;
    settingAugment.checked = Boolean(values.augment);
  }

  function connect() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";
    activeSocket = ws;
    ws.onclose = () => {
      if (activeSocket === ws) {
        activeSocket = null;
      }
      setTimeout(connect, 1000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  }

  // "user" (front) is the default — the app is designed to be propped up
  // facing you while you hold cards up to it, same as the laptop-webcam
  // workflow — but phones have a back camera too, so a switch button lets
  // people flip to "environment" if that framing works better for them.
  let currentFacingMode = "user";

  async function requestCameraStream(facingMode) {
    // Request a high resolution so the model has more pixels to work with on
    // soft/small cards; the browser gives the closest it can.
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
    } catch (err) {
      return await navigator.mediaDevices.getUserMedia({ video: true });
    }
  }

  async function switchCamera() {
    const nextFacingMode = currentFacingMode === "user" ? "environment" : "user";
    const previousStream = videoFeed.srcObject;
    try {
      const stream = await requestCameraStream(nextFacingMode);
      videoFeed.srcObject = stream;
      await videoFeed.play();
      currentFacingMode = nextFacingMode;
      if (previousStream) {
        previousStream.getTracks().forEach((track) => track.stop());
      }
    } catch (err) {
      // Couldn't get the other camera (e.g. device only has one) — keep the
      // stream that was already running.
    }
  }

  async function startCamera() {
    try {
      const stream = await requestCameraStream(currentFacingMode);
      videoFeed.srcObject = stream;
      await videoFeed.play();
    } catch (err) {
      statusLabel.textContent = "Camera unavailable — check browser permissions";
      return;
    }

    const context = captureCanvas.getContext("2d");
    setInterval(() => {
      if (!activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;
      if (!videoFeed.videoWidth || !videoFeed.videoHeight) return;
      // Backpressure: if frames aren't draining to the server fast enough, skip
      // this one rather than letting the send buffer grow without bound.
      if (activeSocket.bufferedAmount > 1_000_000) return;

      captureCanvas.width = videoFeed.videoWidth;
      captureCanvas.height = videoFeed.videoHeight;
      context.drawImage(videoFeed, 0, 0, captureCanvas.width, captureCanvas.height);
      captureCanvas.toBlob(
        (blob) => {
          if (blob && activeSocket && activeSocket.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => activeSocket.send(buf));
          }
        },
        "image/jpeg",
        JPEG_QUALITY
      );
    }, FRAME_INTERVAL_MS);
  }

  stopButton.addEventListener("click", () => fetch("/api/stop", { method: "POST" }));
  resetButton.addEventListener("click", () => fetch("/api/reset", { method: "POST" }));
  document.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "q") {
      fetch("/api/stop", { method: "POST" });
    }
  });

  // Onboarding overlay: shown every visit (not just the first), so nobody
  // lands on a stale previous run without a reminder of how it works.
  onboardingDismiss.addEventListener("click", () => {
    onboardingOverlay.classList.add("hidden");
  });

  // Settings panel.
  settingsToggle.addEventListener("click", () => {
    settingsPanel.classList.toggle("open");
  });
  document.addEventListener("click", (event) => {
    if (
      settingsPanel.classList.contains("open") &&
      !settingsPanel.contains(event.target) &&
      event.target !== settingsToggle
    ) {
      settingsPanel.classList.remove("open");
    }
  });
  settingsApply.addEventListener("click", () => {
    fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confidence: parseFloat(settingConfidence.value),
        maybe_threshold: parseFloat(settingMaybeThreshold.value),
        min_consensus: parseInt(settingMinConsensus.value, 10),
        presence_confidence: parseFloat(settingPresenceConfidence.value),
        augment: settingAugment.checked,
      }),
    })
      .then((res) => res.json())
      .then(applySettingsToForm);
  });

  fetch("/api/settings")
    .then((res) => res.json())
    .then(applySettingsToForm);

  // Card preview modal.
  cardPreviewClose.addEventListener("click", closeCardPreview);
  cardPreviewModal.addEventListener("click", (event) => {
    if (event.target === cardPreviewModal) {
      closeCardPreview();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && cardPreviewModal.classList.contains("open")) {
      closeCardPreview();
    }
  });

  cameraSwitchButton.addEventListener("click", () => switchCamera());

  // Every fresh page load starts clean — no leftover results or in-progress
  // session from a previous visit. Reset before opening the WebSocket so the
  // very first state this page sees is "waiting", not a stale prior run.
  fetch("/api/reset", { method: "POST" }).finally(() => {
    connect();
    startCamera();
  });
})();
