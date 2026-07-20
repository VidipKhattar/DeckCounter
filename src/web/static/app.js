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
  const settingMinConsensus = document.getElementById("setting-min-consensus");
  const settingPresenceConfidence = document.getElementById("setting-presence-confidence");
  const cardPreviewModal = document.getElementById("card-preview-modal");
  const cardPreviewImage = document.getElementById("card-preview-image");
  const cardPreviewClose = document.getElementById("card-preview-close");
  const videoFeed = document.getElementById("video-feed");
  const captureCanvas = document.getElementById("capture-canvas");

  const SUIT_NAMES = { S: "Spades", H: "Hearts", D: "Diamonds", C: "Clubs" };
  const ONBOARDED_KEY = "deckcounter_onboarded";
  const FRAME_INTERVAL_MS = 120;

  let activeSocket = null;

  function describeCard(card) {
    const suit = card.slice(-1);
    const rank = card.slice(0, -1);
    return `${rank} of ${SUIT_NAMES[suit] || suit}`;
  }

  function setState(type) {
    guideBox.classList.toggle("recording", type === "recording");
    stopButton.classList.toggle("hidden", type !== "recording");
    resultsSection.style.display = type === "done" ? "grid" : "none";
    statusLabel.classList.toggle("status-done", type === "done");
    statusLabel.classList.toggle("status-recording", type === "recording");

    if (type === "waiting") {
      statusLabel.textContent = "Place a card";
    } else if (type === "recording") {
      statusLabel.textContent = "Recording — flip through the deck";
    } else if (type === "processing") {
      statusLabel.textContent = "Analyzing…";
    } else if (type === "done") {
      statusLabel.textContent = "Analysis complete";
    } else if (type === "error") {
      statusLabel.textContent = "Camera unavailable";
    }
  }

  function renderCaptured(cards) {
    capturedTokens.innerHTML = "";
    cards.forEach((card) => {
      const span = document.createElement("span");
      span.className = "token token-captured";
      span.textContent = card;
      span.title = `View best captured frame for ${describeCard(card)}`;
      span.addEventListener("click", () => openCardPreview(card));
      capturedTokens.appendChild(span);
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
      missingHeading.textContent = `Critical: Missing Components (${msg.missing.length})`;
      renderMissing(msg.missing);
      capturedHeading.textContent = `Detected Inventory (${msg.seen.length})`;
      renderCaptured(msg.seen);
      metricAccuracy.textContent = `${msg.metrics.reliability}%`;
      metricSamples.textContent = msg.metrics.frame_count;
      metricLatency.textContent = `${msg.metrics.latency_ms}ms`;
    } else if (msg.type === "recording") {
      statusLabel.textContent = `Recording — ${msg.frame_count} frames captured`;
    } else if (msg.type === "processing") {
      const pct = msg.total ? Math.round((msg.current / msg.total) * 100) : 0;
      statusLabel.textContent = `Analyzing… ${pct}%`;
    }
  }

  function applySettingsToForm(values) {
    settingConfidence.value = values.confidence;
    settingMinConsensus.value = values.min_consensus;
    settingPresenceConfidence.value = values.presence_confidence;
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

  async function startCamera() {
    try {
      // Prefer the rear camera on phones/tablets (front camera is the wrong
      // choice for scanning cards); laptops with only one camera ignore this.
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
        });
      } catch (err) {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }
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
        0.7
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

  // Onboarding overlay: shown once per browser until dismissed.
  if (localStorage.getItem(ONBOARDED_KEY)) {
    onboardingOverlay.classList.add("hidden");
  }
  onboardingDismiss.addEventListener("click", () => {
    onboardingOverlay.classList.add("hidden");
    localStorage.setItem(ONBOARDED_KEY, "1");
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
        min_consensus: parseInt(settingMinConsensus.value, 10),
        presence_confidence: parseFloat(settingPresenceConfidence.value),
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

  connect();
  startCamera();
})();
