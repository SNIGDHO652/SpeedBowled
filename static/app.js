"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  calibration: null,
  calibrationBlob: null,
  calibrationStream: null,

  videoStream: null,
  videoBlob: null,
  videoFilename: "bowling_clip.webm",
  videoObjectUrl: null,
  videoDurationS: null,

  recorder: null,
  recordingStartedAtMs: null,
  recorderMimeType: "",
  chunks: [],
  cancelRecording: false,

  chartPayload: null,
  bounces: [],
};

let toastTimer = null;


function showToast(message, type = "success") {
  const toast = $("toast");

  window.clearTimeout(toastTimer);

  toast.classList.remove(
    "error",
    "warning",
  );

  if (
    type === "error"
    || type === "warning"
  ) {
    toast.classList.add(type);
  }

  $("toastIcon").textContent =
    type === "error"
      ? "!"
      : type === "warning"
        ? "△"
        : "✓";

  $("toastText").textContent = message;

  toast.classList.add("show");

  toastTimer = window.setTimeout(
    () => {
      toast.classList.remove("show");
    },
    2000,
  );
}


async function readError(response) {
  try {
    const payload = await response.json();

    return (
      payload.error
      || "The request failed."
    );
  } catch {
    return (
      `Request failed with status ${response.status}.`
    );
  }
}


function stopStream(stream) {
  if (!stream) {
    return;
  }

  for (const track of stream.getTracks()) {
    track.stop();
  }
}


function getTrackingSettings() {
  const diameterCm = Number(
    $("ballDiameterInput").value,
  );

  const hueTolerancePercent = Number(
    $("hueToleranceInput").value,
  );

  const saturationTolerancePercent = Number(
    $("saturationToleranceInput").value,
  );

  const valueTolerancePercent = Number(
    $("valueToleranceInput").value,
  );

  if (
    !Number.isFinite(
      diameterCm,
    )
    || diameterCm < 1
    || diameterCm > 50
  ) {
    throw new Error(
      "Ball diameter must be between 1 and 50 cm.",
    );
  }

  for (
    const [
      label,
      value,
    ]
    of [
      [
        "H tolerance",
        hueTolerancePercent,
      ],
      [
        "S tolerance",
        saturationTolerancePercent,
      ],
      [
        "V tolerance",
        valueTolerancePercent,
      ],
    ]
  ) {
    if (
      !Number.isFinite(
        value,
      )
      || value < 0
      || value > 100
    ) {
      throw new Error(
        `${label} must be between 0% and 100%.`,
      );
    }
  }

  return {
    diameterCm,
    hueTolerancePercent,
    saturationTolerancePercent,
    valueTolerancePercent,
  };
}


function updateSpecReadout() {
  try {
    const settings = getTrackingSettings();

    $("specReadout").textContent =
      `${settings.diameterCm.toFixed(2)} cm`;
  } catch {
    $("specReadout").textContent =
      "Check value";
  }
}


function resetResults() {
  $("resultsSection").hidden = true;
  state.chartPayload = null;
  state.bounces = [];
}


function markStep(stepNumber) {
  for (
    let index = 1;
    index <= 3;
    index += 1
  ) {
    const element = $(
      `stepIndicator${index}`,
    );

    element.classList.remove(
      "active",
      "done",
    );

    if (index < stepNumber) {
      element.classList.add("done");
    } else if (index === stepNumber) {
      element.classList.add("active");
    }
  }
}


function invalidateCalibration() {
  updateSpecReadout();

  if (!state.calibration) {
    return;
  }

  state.calibration = null;

  $("calibrationStatus").textContent =
    "Recalibration needed";

  $("calibrationStatus").classList.remove(
    "success",
  );

  $("videoSection").classList.add(
    "locked",
  );

  $("openVideoCameraButton").disabled =
    true;

  $("uploadVideoButton").disabled =
    true;

  $("analyseButton").disabled =
    true;

  $("videoStatus").textContent =
    "Waiting for calibration";

  $("videoStatus").classList.remove(
    "success",
  );

  $("focalReadout").textContent =
    "—";

  resetResults();
  markStep(1);

  showToast(
    "Ball diameter changed. Recalibrate before analysis.",
    "warning",
  );
}


function unlockVideoStep() {
  $("videoSection").classList.remove(
    "locked",
  );

  $("openVideoCameraButton").disabled =
    false;

  $("uploadVideoButton").disabled =
    false;

  $("videoStatus").textContent =
    "Ready";

  $("videoStatus").classList.add(
    "success",
  );

  markStep(2);
}


async function startCalibrationCamera() {
  try {
    getTrackingSettings();

    closeCalibrationCamera(
      false,
    );

    state.calibrationStream =
      await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: {
            ideal: "environment",
          },
        },
        audio: false,
      });

    const video = $(
      "calibrationCamera",
    );

    video.srcObject =
      state.calibrationStream;

    video.hidden = false;

    $("captureCalibrationButton").hidden =
      false;

    $("closeCalibrationCamera").hidden =
      false;

    $("calibrationEmpty").hidden =
      true;

    $("calibrationCanvasWrap").hidden =
      true;

    showToast(
      "Camera ready. Keep the ball exactly 1 m away.",
    );
  } catch (error) {
    showToast(
      error.message
      || "Camera access failed. Upload an image instead.",
      "error",
    );
  }
}


function closeCalibrationCamera(
  notify = true,
) {
  stopStream(
    state.calibrationStream,
  );

  state.calibrationStream =
    null;

  const video = $(
    "calibrationCamera",
  );

  video.srcObject = null;
  video.hidden = true;

  $("captureCalibrationButton").hidden =
    true;

  $("closeCalibrationCamera").hidden =
    true;

  if (
    $("calibrationCanvasWrap").hidden
  ) {
    $("calibrationEmpty").hidden =
      false;
  }

  if (notify) {
    showToast(
      "Calibration camera closed.",
    );
  }
}


function canvasToBlob(canvas) {
  return new Promise(
    (resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(
              new Error(
                "Could not create image blob.",
              ),
            );
          }
        },
        "image/jpeg",
        0.92,
      );
    },
  );
}


async function captureCalibrationFrame() {
  const video = $(
    "calibrationCamera",
  );

  if (
    !video.videoWidth
    || !video.videoHeight
  ) {
    showToast(
      "The camera frame is not ready yet.",
      "warning",
    );

    return;
  }

  const canvas = $(
    "calibrationCanvas",
  );

  canvas.width =
    video.videoWidth;

  canvas.height =
    video.videoHeight;

  canvas
    .getContext("2d")
    .drawImage(
      video,
      0,
      0,
    );

  try {
    state.calibrationBlob =
      await canvasToBlob(
        canvas,
      );

    state.calibration =
      null;

    closeCalibrationCamera(
      false,
    );

    $("calibrationCanvasWrap").hidden =
      false;

    $("calibrationEmpty").hidden =
      true;

    resetResults();

    showToast(
      "Frame captured. Click the centre of the ball.",
    );
  } catch {
    showToast(
      "Could not capture the camera frame.",
      "error",
    );
  }
}


function loadImageFile(file) {
  return new Promise(
    (resolve, reject) => {
      const image =
        new Image();

      const objectUrl =
        URL.createObjectURL(
          file,
        );

      image.onload = () => {
        URL.revokeObjectURL(
          objectUrl,
        );

        resolve(image);
      };

      image.onerror = () => {
        URL.revokeObjectURL(
          objectUrl,
        );

        reject(
          new Error(
            "Image could not be loaded.",
          ),
        );
      };

      image.src =
        objectUrl;
    },
  );
}


async function handleCalibrationUpload(
  file,
) {
  if (!file) {
    return;
  }

  try {
    getTrackingSettings();

    const image =
      await loadImageFile(
        file,
      );

    const canvas = $(
      "calibrationCanvas",
    );

    canvas.width =
      image.naturalWidth;

    canvas.height =
      image.naturalHeight;

    canvas
      .getContext("2d")
      .drawImage(
        image,
        0,
        0,
      );

    state.calibrationBlob =
      file;

    state.calibration =
      null;

    closeCalibrationCamera(
      false,
    );

    $("calibrationCanvasWrap").hidden =
      false;

    $("calibrationEmpty").hidden =
      true;

    resetResults();

    showToast(
      "Image loaded. Click the centre of the ball.",
    );
  } catch (error) {
    showToast(
      error.message
      || "That image could not be opened.",
      "error",
    );
  }
}


async function calibrateAtPoint(
  event,
) {
  if (!state.calibrationBlob) {
    showToast(
      "Capture or upload an image first.",
      "warning",
    );

    return;
  }

  let settings;

  try {
    settings =
      getTrackingSettings();
  } catch (error) {
    showToast(
      error.message,
      "error",
    );

    return;
  }

  const canvas = $(
    "calibrationCanvas",
  );

  const bounds =
    canvas.getBoundingClientRect();

  const x = Math.round(
    (
      event.clientX
      - bounds.left
    )
    * (
      canvas.width
      / bounds.width
    ),
  );

  const y = Math.round(
    (
      event.clientY
      - bounds.top
    )
    * (
      canvas.height
      / bounds.height
    ),
  );

  const form =
    new FormData();

  form.append(
    "image",
    state.calibrationBlob,
    "calibration.jpg",
  );

  form.append(
    "x",
    String(x),
  );

  form.append(
    "y",
    String(y),
  );

  form.append(
    "ball_diameter_cm",
    String(
      settings.diameterCm,
    ),
  );

  form.append(
    "hue_tolerance_percent",
    String(
      settings.hueTolerancePercent,
    ),
  );

  form.append(
    "saturation_tolerance_percent",
    String(
      settings.saturationTolerancePercent,
    ),
  );

  form.append(
    "value_tolerance_percent",
    String(
      settings.valueTolerancePercent,
    ),
  );

  $("calibrationStatus").textContent =
    "Calibrating…";

  try {
    const response =
      await fetch(
        "/api/calibrate",
        {
          method: "POST",
          body: form,
        },
      );

    if (!response.ok) {
      throw new Error(
        await readError(
          response,
        ),
      );
    }

    const payload =
      await response.json();

    state.calibration =
      payload;

    $("focalReadout").textContent =
      `${payload.focal_length_px.toFixed(0)} px`;

    $("diameterReadout").textContent =
      `${payload.ball_diameter_px.toFixed(1)} px`;

    $("specReadout").textContent =
      `${payload.ball_diameter_cm.toFixed(2)} cm`;

    const hsv =
      payload.color.hsv;

    $("colourReadout").textContent =
      `HSV ${hsv[0]}, ${hsv[1]}, ${hsv[2]}`;

    const rgb =
      payload.color.rgb;

    $("colourSwatch").style.background =
      `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

    $("circleReadout").textContent =
      `${
        Math.max(
          0,
          payload.circularity
          * 100,
        ).toFixed(0)
      }%`;

    $("calibrationStatus").textContent =
      "Calibrated";

    $("calibrationStatus").classList.add(
      "success",
    );

    if (
      payload.preview_data_uri
    ) {
      const image =
        new Image();

      image.onload = () => {
        canvas.width =
          image.naturalWidth;

        canvas.height =
          image.naturalHeight;

        canvas
          .getContext("2d")
          .drawImage(
            image,
            0,
            0,
          );
      };

      image.src =
        payload.preview_data_uri;
    }

    unlockVideoStep();

    if (
      payload.warnings
      && payload.warnings.length
    ) {
      showToast(
        payload.warnings[0],
        "warning",
      );
    } else {
      showToast(
        "Camera and ball colour calibrated.",
      );
    }
  } catch (error) {
    $("calibrationStatus").textContent =
      "Calibration failed";

    $("calibrationStatus").classList.remove(
      "success",
    );

    showToast(
      error.message,
      "error",
    );
  }
}


function chooseRecorderMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];

  for (
    const candidate
    of candidates
  ) {
    if (
      typeof MediaRecorder
      !== "undefined"
      && MediaRecorder.isTypeSupported(
        candidate,
      )
    ) {
      return candidate;
    }
  }

  return "";
}


async function openVideoCamera() {
  if (
    state.recorder
    && state.recorder.state !== "inactive"
  ) {
    showToast(
      "Stop or close the current recording before reopening the camera.",
      "warning",
    );
    return;
  }

  if (!state.calibration) {
    showToast(
      "Calibrate the camera first.",
      "warning",
    );

    return;
  }

  if (
    typeof MediaRecorder
    === "undefined"
  ) {
    showToast(
      "This browser cannot record here. Upload a clip instead.",
      "error",
    );

    return;
  }

  try {
    closeVideoCamera(
      false,
    );

    state.videoStream =
      await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: {
            ideal: "environment",
          },
        },
        audio: false,
      });

    const camera = $(
      "analysisCamera",
    );

    camera.srcObject =
      state.videoStream;

    camera.hidden =
      false;

    $("videoPreview").hidden =
      true;

    $("videoEmpty").hidden =
      true;

    $("recordControls").hidden =
      false;

    $("startRecordingButton").hidden =
      false;

    $("stopRecordingButton").hidden =
      true;

    $("recordingStatusText").textContent =
      "Camera ready";

    $("recordDot").classList.add(
      "idle",
    );

    $("videoStatus").textContent =
      "Camera ready";

    showToast(
      "Video camera opened.",
    );
  } catch {
    showToast(
      "Camera could not open. Upload a video instead.",
      "error",
    );
  }
}


function startRecording() {
  if (!state.videoStream) {
    showToast(
      "Open the camera first.",
      "warning",
    );

    return;
  }

  const mimeType =
    chooseRecorderMimeType();

  state.recorderMimeType =
    mimeType;

  state.chunks = [];
  state.cancelRecording =
    false;

  state.recorder =
    mimeType
      ? new MediaRecorder(
          state.videoStream,
          {
            mimeType,
          },
        )
      : new MediaRecorder(
          state.videoStream,
        );

  state.recorder.addEventListener(
    "dataavailable",
    (event) => {
      if (
        event.data
        && event.data.size > 0
      ) {
        state.chunks.push(
          event.data,
        );
      }
    },
  );

  state.recorder.addEventListener(
    "stop",
    finishRecording,
    {
      once: true,
    },
  );

  state.recordingStartedAtMs =
    performance.now();

  state.recorder.start(
    250,
  );

  $("startRecordingButton").hidden =
    true;

  $("stopRecordingButton").hidden =
    false;

  $("recordingStatusText").textContent =
    "Recording";

  $("recordDot").classList.remove(
    "idle",
  );

  $("videoStatus").textContent =
    "Recording…";

  $("analyseButton").disabled =
    true;

  showToast(
    "Recording started.",
    "warning",
  );
}


function finishRecording() {
  const cancelled =
    state.cancelRecording;

  if (!cancelled) {
    const recordedType =
      state.recorder?.mimeType
      || state.recorderMimeType
      || "video/webm";

    state.videoBlob =
      new Blob(
        state.chunks,
        {
          type: recordedType,
        },
      );

    state.videoFilename =
      recordedType.includes(
        "mp4",
      )
        ? "speedbowled_recording.mp4"
        : "speedbowled_recording.webm";

    if (
      Number.isFinite(
        state.recordingStartedAtMs,
      )
    ) {
      state.videoDurationS = Math.max(
        0,
        (
          performance.now()
          - state.recordingStartedAtMs
        )
        / 1000,
      );
    }
  }

  state.recordingStartedAtMs =
    null;

  state.recorder =
    null;

  stopStream(
    state.videoStream,
  );

  state.videoStream =
    null;

  const camera = $(
    "analysisCamera",
  );

  camera.srcObject =
    null;

  camera.hidden =
    true;

  $("recordControls").hidden =
    true;

  if (cancelled) {
    if (state.videoBlob) {
      showVideoPreview(
        state.videoBlob,
      );
    } else {
      $("videoEmpty").hidden =
        false;
    }

    $("videoStatus").textContent =
      state.videoBlob
        ? "Clip ready"
        : "Ready";

    $("analyseButton").disabled =
      !state.videoBlob;

    showToast(
      "Camera closed. Recording discarded.",
      "warning",
    );

    return;
  }

  showVideoPreview(
    state.videoBlob,
    state.videoDurationS,
  );

  $("videoStatus").textContent =
    "Clip ready";

  $("analyseButton").disabled =
    false;

  resetResults();

  showToast(
    "Recording ready for analysis.",
  );
}


function stopRecording() {
  if (
    state.recorder
    && state.recorder.state
    !== "inactive"
  ) {
    state.cancelRecording =
      false;

    state.recorder.stop();
  }
}


function closeVideoCamera(
  notify = true,
) {
  if (
    state.recorder
    && state.recorder.state
    !== "inactive"
  ) {
    state.cancelRecording =
      true;

    state.recorder.stop();

    return;
  }

  stopStream(
    state.videoStream,
  );

  state.videoStream =
    null;

  const camera = $(
    "analysisCamera",
  );

  camera.srcObject =
    null;

  camera.hidden =
    true;

  $("recordControls").hidden =
    true;

  if (state.videoBlob) {
    showVideoPreview(
      state.videoBlob,
    );

    $("videoStatus").textContent =
      "Clip ready";

    $("analyseButton").disabled =
      false;
  } else {
    $("videoPreview").hidden =
      true;

    $("videoEmpty").hidden =
      false;

    $("videoStatus").textContent =
      state.calibration
        ? "Ready"
        : "Waiting for calibration";
  }

  if (notify) {
    showToast(
      "Video camera closed.",
    );
  }
}


function applyVideoDuration(
  durationS,
) {
  if (
    !Number.isFinite(
      durationS,
    )
    || durationS <= 0
  ) {
    state.videoDurationS =
      null;

    $("analysisStartInput").disabled =
      false;

    $("analysisEndInput").disabled =
      false;

    $("analysisStartInput").value =
      "0.000";

    $("analysisEndInput").value =
      "";

    $("videoDurationReadout").textContent =
      "Duration unavailable — end defaults to video end";

    return;
  }

  state.videoDurationS =
    durationS;

  $("analysisStartInput").disabled =
    false;

  $("analysisEndInput").disabled =
    false;

  $("analysisStartInput").max =
    durationS.toFixed(
      3,
    );

  $("analysisEndInput").max =
    durationS.toFixed(
      3,
    );

  $("analysisStartInput").value =
    "0.000";

  $("analysisEndInput").value =
    durationS.toFixed(
      3,
    );

  $("videoDurationReadout").textContent =
    `Video duration: ${durationS.toFixed(3)} s`;
}


function showVideoPreview(
  blob,
  knownDurationS = null,
) {
  if (
    state.videoObjectUrl
  ) {
    URL.revokeObjectURL(
      state.videoObjectUrl,
    );
  }

  state.videoObjectUrl =
    URL.createObjectURL(
      blob,
    );

  const preview = $(
    "videoPreview",
  );

  preview.onloadedmetadata =
    () => {
      const metadataDuration =
        Number(
          preview.duration,
        );

      if (
        Number.isFinite(
          metadataDuration,
        )
        && metadataDuration > 0
      ) {
        applyVideoDuration(
          metadataDuration,
        );
      } else if (
        Number.isFinite(
          knownDurationS,
        )
        && knownDurationS > 0
      ) {
        applyVideoDuration(
          knownDurationS,
        );
      } else {
        applyVideoDuration(
          null,
        );
      }
    };

  preview.src =
    state.videoObjectUrl;

  preview.hidden =
    false;

  $("videoEmpty").hidden =
    true;

  $("analysisCamera").hidden =
    true;

  if (
    Number.isFinite(
      knownDurationS,
    )
    && knownDurationS > 0
  ) {
    applyVideoDuration(
      knownDurationS,
    );
  }
}


function handleVideoUpload(
  file,
) {
  if (!file) {
    return;
  }

  closeVideoCamera(
    false,
  );

  state.videoBlob =
    file;

  state.videoDurationS =
    null;

  state.videoFilename =
    file.name
    || "bowling_clip.mp4";

  showVideoPreview(
    file,
  );

  $("videoStatus").textContent =
    "Clip ready";

  $("analyseButton").disabled =
    false;

  resetResults();

  showToast(
    "Video loaded.",
  );
}


function getAnalysisWindow() {
  const startRaw =
    $("analysisStartInput").value.trim();

  const endRaw =
    $("analysisEndInput").value.trim();

  const startS =
    startRaw === ""
      ? 0
      : Number(
          startRaw,
        );

  const endS =
    endRaw === ""
      ? null
      : Number(
          endRaw,
        );

  if (
    !Number.isFinite(
      startS,
    )
    || startS < 0
  ) {
    throw new Error(
      "Analysis start time must be zero or greater.",
    );
  }

  if (
    endS !== null
    && (
      !Number.isFinite(
        endS,
      )
      || endS <= startS
    )
  ) {
    throw new Error(
      "Analysis end time must be greater than the start time.",
    );
  }

  if (
    state.videoDurationS !== null
    && startS
    >= state.videoDurationS
  ) {
    throw new Error(
      "Analysis start time must be before the end of the video.",
    );
  }

  if (
    state.videoDurationS !== null
    && endS !== null
    && endS
    > state.videoDurationS
    + 0.001
  ) {
    throw new Error(
      "Analysis end time cannot exceed the video duration.",
    );
  }

  return {
    startS,
    endS,
  };
}


function setAnalyseBusy(
  busy,
) {
  $("analyseButton").disabled =
    busy;

  $("analyseSpinner").hidden =
    !busy;

  $("analyseButtonText").textContent =
    busy
      ? "Analysing delivery…"
      : "Analyse delivery";
}


function metricText(
  value,
  digits,
) {
  const number =
    Number(
      value,
    );

  return Number.isFinite(
    number,
  )
    ? number.toFixed(
        digits,
      )
    : "—";
}


function renderMetric(
  prefix,
  payload,
  digits,
) {
  $(
    `${prefix}Avg`
  ).textContent = metricText(
    payload.avg,
    digits,
  );

  $(
    `${prefix}Min`
  ).textContent = metricText(
    payload.min,
    digits,
  );

  $(
    `${prefix}Max`
  ).textContent = metricText(
    payload.max,
    digits,
  );
}


function renderWarnings(
  warnings,
) {
  const area =
    $("warningArea");

  area.innerHTML =
    "";

  for (
    const warning
    of warnings || []
  ) {
    const element =
      document.createElement(
        "div",
      );

    element.className =
      "warning-message";

    element.textContent =
      warning;

    area.appendChild(
      element,
    );
  }
}


function renderBounces(
  bounces,
) {
  const summary =
    $("bounceSummary");

  const list =
    $("bounceList");

  list.innerHTML =
    "";

  $("bounceCount").textContent =
    String(
      bounces.length,
    );

  $("bounceSummaryTitle").textContent =
    bounces.length === 1
      ? "1 bounce / abrupt change detected"
      : `${bounces.length} bounce / abrupt changes detected`;

  if (!bounces.length) {
    summary.hidden =
      true;

    return;
  }

  summary.hidden =
    false;

  for (
    const bounce
    of bounces
  ) {
    const item =
      document.createElement(
        "div",
      );

    item.className =
      "bounce-item";

    item.innerHTML = `
      <strong>${bounce.label}</strong>
      <span>${bounce.time_s.toFixed(3)} s</span>
      <span>${bounce.angle_deg.toFixed(1)}° direction change</span>
      <span>${bounce.speed_before_kmph.toFixed(1)} → ${bounce.speed_after_kmph.toFixed(1)} km/h</span>
    `;

    list.appendChild(
      item,
    );
  }
}


function buildChartGeometry(
  width,
  height,
  points,
) {
  const padding = {
    left: 48,
    right: 16,
    top: 25,
    bottom: 30,
  };

  const chartWidth = (
    width
    - padding.left
    - padding.right
  );

  const chartHeight = (
    height
    - padding.top
    - padding.bottom
  );

  const times = points.map(
    (point) => Number(
      point.t,
    ),
  );

  const values = points.map(
    (point) => Number(
      point.v,
    ),
  );

  let minTime =
    Math.min(
      ...times,
    );

  let maxTime =
    Math.max(
      ...times,
    );

  let minValue =
    Math.min(
      ...values,
    );

  let maxValue =
    Math.max(
      ...values,
    );

  if (
    Math.abs(
      maxTime
      - minTime,
    )
    < 1e-9
  ) {
    maxTime += 1;
  }

  if (
    Math.abs(
      maxValue
      - minValue,
    )
    < 1e-9
  ) {
    maxValue += 1;
    minValue -= 1;
  }

  const valuePadding = (
    maxValue
    - minValue
  ) * 0.12;

  minValue -=
    valuePadding;

  maxValue +=
    valuePadding;

  const xPosition =
    (time) => (
      padding.left
      + (
        (
          time
          - minTime
        )
        / (
          maxTime
          - minTime
        )
      )
      * chartWidth
    );

  const yPosition =
    (value) => (
      padding.top
      + (
        1
        - (
          (
            value
            - minValue
          )
          / (
            maxValue
            - minValue
          )
        )
      )
      * chartHeight
    );

  return {
    padding,
    chartWidth,
    chartHeight,
    minTime,
    maxTime,
    minValue,
    maxValue,
    xPosition,
    yPosition,
  };
}


function drawChart(
  canvas,
  points,
  unit,
  bounces,
) {
  const width = Math.max(
    canvas.clientWidth,
    280,
  );

  const height = Math.max(
    canvas.clientHeight,
    230,
  );

  const pixelRatio =
    window.devicePixelRatio
    || 1;

  canvas.width =
    Math.round(
      width
      * pixelRatio,
    );

  canvas.height =
    Math.round(
      height
      * pixelRatio,
    );

  const context =
    canvas.getContext(
      "2d",
    );

  context.setTransform(
    pixelRatio,
    0,
    0,
    pixelRatio,
    0,
    0,
  );

  context.clearRect(
    0,
    0,
    width,
    height,
  );

  if (
    !points
    || points.length < 2
  ) {
    context.fillStyle =
      "#829789";

    context.font =
      "12px system-ui";

    context.textAlign =
      "center";

    context.fillText(
      "Not enough proper samples",
      width / 2,
      height / 2,
    );

    canvas._chartState =
      null;

    return;
  }

  const geometry =
    buildChartGeometry(
      width,
      height,
      points,
    );

  const {
    padding,
    chartHeight,
    minTime,
    maxTime,
    minValue,
    maxValue,
    xPosition,
    yPosition,
  } = geometry;

  context.strokeStyle =
    "rgba(172,255,208,0.10)";

  context.fillStyle =
    "#71867a";

  context.lineWidth =
    1;

  context.font =
    "10px system-ui";

  for (
    let index = 0;
    index <= 4;
    index += 1
  ) {
    const ratio =
      index / 4;

    const y = (
      padding.top
      + ratio
      * chartHeight
    );

    context.beginPath();

    context.moveTo(
      padding.left,
      y,
    );

    context.lineTo(
      width
      - padding.right,
      y,
    );

    context.stroke();

    context.textAlign =
      "right";

    context.fillText(
      (
        maxValue
        - ratio
        * (
          maxValue
          - minValue
        )
      ).toFixed(1),
      padding.left - 8,
      y + 3,
    );
  }

  for (
    const bounce
    of bounces || []
  ) {
    if (
      bounce.time_s < minTime
      || bounce.time_s > maxTime
    ) {
      continue;
    }

    const x =
      xPosition(
        bounce.time_s,
      );

    context.save();

    context.strokeStyle =
      "rgba(255,179,91,0.85)";

    context.setLineDash(
      [5, 5],
    );

    context.lineWidth =
      1.5;

    context.beginPath();

    context.moveTo(
      x,
      padding.top,
    );

    context.lineTo(
      x,
      height
      - padding.bottom,
    );

    context.stroke();

    context.setLineDash(
      [],
    );

    context.fillStyle =
      "#ffbd72";

    context.font =
      "700 10px system-ui";

    context.textAlign =
      "center";

    context.fillText(
      `B${bounce.number}`,
      x,
      13,
    );

    context.restore();
  }

  const gradient =
    context.createLinearGradient(
      0,
      padding.top,
      0,
      height
      - padding.bottom,
    );

  gradient.addColorStop(
    0,
    "rgba(168,255,96,0.25)",
  );

  gradient.addColorStop(
    1,
    "rgba(70,223,218,0.01)",
  );

  context.beginPath();

  points.forEach(
    (point, index) => {
      const x =
        xPosition(
          Number(
            point.t,
          ),
        );

      const y =
        yPosition(
          Number(
            point.v,
          ),
        );

      if (index === 0) {
        context.moveTo(
          x,
          y,
        );
      } else {
        context.lineTo(
          x,
          y,
        );
      }
    },
  );

  context.lineTo(
    xPosition(
      Number(
        points[
          points.length - 1
        ].t,
      ),
    ),
    height
    - padding.bottom,
  );

  context.lineTo(
    xPosition(
      Number(
        points[0].t,
      ),
    ),
    height
    - padding.bottom,
  );

  context.closePath();

  context.fillStyle =
    gradient;

  context.fill();

  context.beginPath();

  points.forEach(
    (point, index) => {
      const x =
        xPosition(
          Number(
            point.t,
          ),
        );

      const y =
        yPosition(
          Number(
            point.v,
          ),
        );

      if (index === 0) {
        context.moveTo(
          x,
          y,
        );
      } else {
        context.lineTo(
          x,
          y,
        );
      }
    },
  );

  context.strokeStyle =
    "#a8ff60";

  context.lineWidth =
    2.2;

  context.lineJoin =
    "round";

  context.lineCap =
    "round";

  context.stroke();

  points.forEach(
    (point) => {
      if (
        !point.excluded_from_summary
      ) {
        return;
      }

      const x =
        xPosition(
          Number(
            point.t,
          ),
        );

      const y =
        yPosition(
          Number(
            point.v,
          ),
        );

      context.beginPath();

      context.arc(
        x,
        y,
        3,
        0,
        Math.PI * 2,
      );

      context.fillStyle =
        "#ffb35b";

      context.fill();
    },
  );

  if (
    canvas._selectionTimeRange
  ) {
    const startTime =
      Math.max(
        minTime,
        Math.min(
          canvas._selectionTimeRange.start,
          canvas._selectionTimeRange.end,
        ),
      );

    const endTime =
      Math.min(
        maxTime,
        Math.max(
          canvas._selectionTimeRange.start,
          canvas._selectionTimeRange.end,
        ),
      );

    if (
      endTime > startTime
    ) {
      const startX =
        xPosition(
          startTime,
        );

      const endX =
        xPosition(
          endTime,
        );

      context.save();

      context.fillStyle =
        "rgba(70,223,218,0.12)";

      context.strokeStyle =
        "rgba(70,223,218,0.72)";

      context.lineWidth =
        1.2;

      context.fillRect(
        startX,
        padding.top,
        endX - startX,
        chartHeight,
      );

      context.strokeRect(
        startX,
        padding.top,
        endX - startX,
        chartHeight,
      );

      context.restore();
    }
  }

  context.fillStyle =
    "#9eb3a4";

  context.textAlign =
    "left";

  context.fillText(
    `${minTime.toFixed(2)} s`,
    padding.left,
    height - 8,
  );

  context.textAlign =
    "right";

  context.fillText(
    `${maxTime.toFixed(2)} s`,
    width - padding.right,
    height - 8,
  );

  context.fillText(
    unit,
    width - padding.right,
    11,
  );

  canvas._chartState = {
    width,
    height,
    points,
    unit,
    geometry,
  };
}


function nearestChartPoint(
  canvas,
  clientX,
) {
  const chartState =
    canvas._chartState;

  if (!chartState) {
    return null;
  }

  const bounds =
    canvas.getBoundingClientRect();

  const localX = (
    clientX
    - bounds.left
  );

  let nearest =
    null;

  let nearestDistance =
    Infinity;

  for (
    const point
    of chartState.points
  ) {
    const x =
      chartState.geometry.xPosition(
        Number(
          point.t,
        ),
      );

    const distance =
      Math.abs(
        x
        - localX,
      );

    if (
      distance
      < nearestDistance
    ) {
      nearestDistance =
        distance;

      nearest = {
        point,
        x,
        y: (
          chartState.geometry.yPosition(
            Number(
              point.v,
            ),
          )
        ),
      };
    }
  }

  return nearest;
}


function chartTimeFromClientX(
  canvas,
  clientX,
) {
  const chartState =
    canvas._chartState;

  if (!chartState) {
    return null;
  }

  const bounds =
    canvas.getBoundingClientRect();

  const localX = Math.min(
    Math.max(
      clientX
      - bounds.left,
      chartState.geometry.padding.left,
    ),
    chartState.width
    - chartState.geometry.padding.right,
  );

  const ratio = (
    localX
    - chartState.geometry.padding.left
  ) / chartState.geometry.chartWidth;

  return (
    chartState.geometry.minTime
    + ratio
    * (
      chartState.geometry.maxTime
      - chartState.geometry.minTime
    )
  );
}


function showChartTooltip(
  canvas,
  tooltip,
  event,
) {
  const nearest =
    nearestChartPoint(
      canvas,
      event.clientX,
    );

  if (!nearest) {
    tooltip.hidden =
      true;

    return;
  }

  const chartState =
    canvas._chartState;

  const excludedNote =
    nearest.point.excluded_from_summary
      ? "<small>Excluded from speed summaries near bounce</small>"
      : "";

  tooltip.innerHTML = `
    <strong>${Number(nearest.point.v).toFixed(2)} ${chartState.unit}</strong>
    <span>${Number(nearest.point.t).toFixed(3)} s</span>
    ${excludedNote}
  `;

  tooltip.hidden =
    false;

  const parent =
    canvas.parentElement;

  const parentWidth =
    parent.clientWidth;

  const left = Math.min(
    Math.max(
      nearest.x,
      70,
    ),
    parentWidth - 70,
  );

  tooltip.style.left =
    `${left}px`;

  tooltip.style.top =
    `${Math.max(
      10,
      nearest.y - 62,
    )}px`;
}


function rangeStatistics(
  points,
  startTime,
  endTime,
  excludeBounceAdjacent,
) {
  const low = Math.min(
    startTime,
    endTime,
  );

  const high = Math.max(
    startTime,
    endTime,
  );

  const intervalPoints =
    points.filter(
      (point) => (
        Number(
          point.t,
        )
        >= low
        && Number(
          point.t,
        )
        <= high
      ),
    );

  const selected =
    intervalPoints.filter(
      (point) => (
        !excludeBounceAdjacent
        || !point.excluded_from_summary
      ),
    );

  const values =
    selected
      .map(
        (point) => Number(
          point.v,
        ),
      )
      .filter(
        (value) => (
          Number.isFinite(
            value,
          )
        ),
      );

  if (!values.length) {
    return {
      low,
      high,
      count: 0,
      excluded: (
        intervalPoints.length
      ),
    };
  }

  const sum =
    values.reduce(
      (
        total,
        value,
      ) => total + value,
      0,
    );

  return {
    low,
    high,
    count: values.length,
    excluded: (
      intervalPoints.length
      - values.length
    ),
    min: Math.min(
      ...values,
    ),
    max: Math.max(
      ...values,
    ),
    avg: (
      sum
      / values.length
    ),
  };
}


function updateRangeReadout(
  canvas,
) {
  const config =
    canvas._chartConfig;

  if (!config) {
    return;
  }

  const readout =
    $(config.rangeReadoutId);

  if (!canvas._selectionTimeRange) {
    readout.textContent =
      "Drag across the graph to inspect an interval.";

    return;
  }

  const statistics =
    rangeStatistics(
      config.points,
      canvas._selectionTimeRange.start,
      canvas._selectionTimeRange.end,
      config.excludeBounceAdjacent,
    );

  if (!statistics.count) {
    readout.innerHTML = `
      <strong>${statistics.low.toFixed(3)}–${statistics.high.toFixed(3)} s</strong>
      <span>No valid summary samples in this interval.</span>
    `;

    return;
  }

  const exclusionText =
    statistics.excluded > 0
      ? `<small>${statistics.excluded} bounce-adjacent speed sample(s) omitted.</small>`
      : "";

  readout.innerHTML = `
    <strong>${statistics.low.toFixed(3)}–${statistics.high.toFixed(3)} s</strong>
    <span>MIN <b>${statistics.min.toFixed(2)}</b> ${config.unit}</span>
    <span>MAX <b>${statistics.max.toFixed(2)}</b> ${config.unit}</span>
    <span>AVG <b>${statistics.avg.toFixed(2)}</b> ${config.unit}</span>
    <span>N <b>${statistics.count}</b></span>
    ${exclusionText}
  `;
}


function redrawChart(
  canvas,
) {
  const config =
    canvas._chartConfig;

  if (!config) {
    return;
  }

  drawChart(
    canvas,
    config.points,
    config.unit,
    config.bounces,
  );

  updateRangeReadout(
    canvas,
  );
}


function configureChart(
  canvas,
  points,
  unit,
  bounces,
  rangeReadoutId,
  excludeBounceAdjacent = false,
) {
  canvas._chartConfig = {
    points,
    unit,
    bounces,
    rangeReadoutId,
    excludeBounceAdjacent,
  };

  canvas._selectionTimeRange =
    null;

  canvas._isSelecting =
    false;

  drawChart(
    canvas,
    points,
    unit,
    bounces,
  );

  updateRangeReadout(
    canvas,
  );
}


function attachChartInteraction(
  canvas,
  tooltip,
) {
  canvas.addEventListener(
    "pointerdown",
    (event) => {
      const time =
        chartTimeFromClientX(
          canvas,
          event.clientX,
        );

      if (time === null) {
        return;
      }

      canvas._isSelecting =
        true;

      canvas._selectionStartClientX =
        event.clientX;

      canvas._selectionTimeRange = {
        start: time,
        end: time,
      };

      tooltip.hidden =
        true;

      canvas.setPointerCapture?.(
        event.pointerId,
      );

      redrawChart(
        canvas,
      );
    },
  );

  canvas.addEventListener(
    "pointermove",
    (event) => {
      if (
        canvas._isSelecting
        && canvas._selectionTimeRange
      ) {
        const time =
          chartTimeFromClientX(
            canvas,
            event.clientX,
          );

        if (time !== null) {
          canvas._selectionTimeRange.end =
            time;

          redrawChart(
            canvas,
          );
        }

        return;
      }

      showChartTooltip(
        canvas,
        tooltip,
        event,
      );
    },
  );

  const finishSelection =
    (event) => {
      if (
        !canvas._isSelecting
      ) {
        return;
      }

      const movementPx =
        Math.abs(
          event.clientX
          - (
            canvas._selectionStartClientX
            ?? event.clientX
          )
        );

      const time =
        chartTimeFromClientX(
          canvas,
          event.clientX,
        );

      canvas._isSelecting =
        false;

      if (
        movementPx < 5
      ) {
        canvas._selectionTimeRange =
          null;

        redrawChart(
          canvas,
        );

        showChartTooltip(
          canvas,
          tooltip,
          event,
        );

        return;
      }

      if (
        time !== null
        && canvas._selectionTimeRange
      ) {
        canvas._selectionTimeRange.end =
          time;
      }

      redrawChart(
        canvas,
      );
    };


  canvas.addEventListener(
    "pointerup",
    finishSelection,
  );

  canvas.addEventListener(
    "pointercancel",
    finishSelection,
  );

  canvas.addEventListener(
    "pointerleave",
    () => {
      if (
        !canvas._isSelecting
      ) {
        tooltip.hidden =
          true;
      }
    },
  );

  canvas.addEventListener(
    "dblclick",
    () => {
      canvas._selectionTimeRange =
        null;

      redrawChart(
        canvas,
      );
    },
  );
}


function renderCharts(
  series,
) {
  state.chartPayload =
    series;

  configureChart(
    $("speedChart"),
    series.speed,
    "km/h",
    state.bounces,
    "speedRangeReadout",
    true,
  );

  configureChart(
    $("accelerationChart"),
    series.acceleration,
    "m/s²",
    state.bounces,
    "accelerationRangeReadout",
    false,
  );

  configureChart(
    $("swingChart"),
    series.swing,
    "°/s",
    state.bounces,
    "swingRangeReadout",
    false,
  );
}


function renderResults(
  payload,
) {
  renderMetric(
    "speed",
    payload.summary.speed,
    1,
  );

  renderMetric(
    "acceleration",
    payload.summary.acceleration,
    2,
  );

  renderMetric(
    "swing",
    payload.summary.swing,
    1,
  );

  renderWarnings(
    payload.warnings,
  );

  state.bounces =
    payload.bounces || [];

  renderBounces(
    state.bounces,
  );

  $("qualityDetections").textContent =
    `${payload.video.detections}/${payload.video.frames_read}`;

  $("qualitySpeedGraphSamples").textContent =
    String(
      payload.quality.speed_graph_frames,
    );

  $("qualitySpeedSamples").textContent =
    String(
      payload.quality.speed_summary_frames,
    );

  $("qualityExcludedSamples").textContent =
    String(
      payload.quality.speed_samples_excluded_near_bounces,
    );

  $("qualityAccelerationSamples").textContent =
    String(
      payload.quality.acceleration_summary_frames,
    );

  $("qualityFocal").textContent =
    `${payload.calibration.focal_length_px.toFixed(0)} px`;

  $("resultsSection").hidden =
    false;

  markStep(3);

  requestAnimationFrame(
    () => {
      renderCharts(
        payload.series,
      );
    },
  );

  $("resultsSection").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}


async function analyseDelivery() {
  if (!state.calibration) {
    showToast(
      "Calibrate the camera first.",
      "warning",
    );

    return;
  }

  if (!state.videoBlob) {
    showToast(
      "Record or upload a video first.",
      "warning",
    );

    return;
  }

  let settings;
  let analysisWindow;

  try {
    settings =
      getTrackingSettings();

    analysisWindow =
      getAnalysisWindow();
  } catch (error) {
    showToast(
      error.message,
      "error",
    );

    return;
  }

  if (
    Math.abs(
      settings.diameterCm
      - state.calibration.ball_diameter_cm,
    )
    > 1e-9
    || Math.abs(
      settings.hueTolerancePercent
      - state.calibration.tolerances_percent.h
    )
    > 1e-9
    || Math.abs(
      settings.saturationTolerancePercent
      - state.calibration.tolerances_percent.s
    )
    > 1e-9
    || Math.abs(
      settings.valueTolerancePercent
      - state.calibration.tolerances_percent.v
    )
    > 1e-9
  ) {
    invalidateCalibration();

    return;
  }

  setAnalyseBusy(
    true,
  );

  resetResults();

  const form =
    new FormData();

  form.append(
    "video",
    state.videoBlob,
    state.videoFilename,
  );

  form.append(
    "focal_length_px",
    String(
      state.calibration.focal_length_px,
    ),
  );

  form.append(
    "ball_diameter_cm",
    String(
      settings.diameterCm,
    ),
  );

  form.append(
    "hue_tolerance_percent",
    String(
      settings.hueTolerancePercent,
    ),
  );

  form.append(
    "saturation_tolerance_percent",
    String(
      settings.saturationTolerancePercent,
    ),
  );

  form.append(
    "value_tolerance_percent",
    String(
      settings.valueTolerancePercent,
    ),
  );

  form.append(
    "analysis_start_s",
    String(
      analysisWindow.startS,
    ),
  );

  if (
    analysisWindow.endS
    !== null
  ) {
    form.append(
      "analysis_end_s",
      String(
        analysisWindow.endS,
      ),
    );
  }

  const hsv =
    state.calibration.color.hsv;

  form.append(
    "hue",
    String(
      hsv[0],
    ),
  );

  form.append(
    "saturation",
    String(
      hsv[1],
    ),
  );

  form.append(
    "value",
    String(
      hsv[2],
    ),
  );

  try {
    const response =
      await fetch(
        "/api/analyze",
        {
          method: "POST",
          body: form,
        },
      );

    if (!response.ok) {
      throw new Error(
        await readError(
          response,
        ),
      );
    }

    const payload =
      await response.json();

    renderResults(
      payload,
    );

    if (
      payload.warnings
      && payload.warnings.length
    ) {
      showToast(
        "Analysis complete with a quality note.",
        "warning",
      );
    } else {
      showToast(
        "Delivery analysis complete.",
      );
    }
  } catch (error) {
    showToast(
      error.message,
      "error",
    );
  } finally {
    setAnalyseBusy(
      false,
    );
  }
}


$("ballDiameterInput").addEventListener(
  "input",
  updateSpecReadout,
);

$("ballDiameterInput").addEventListener(
  "change",
  invalidateCalibration,
);

for (
  const inputId
  of [
    "hueToleranceInput",
    "saturationToleranceInput",
    "valueToleranceInput",
  ]
) {
  $(inputId).addEventListener(
    "change",
    invalidateCalibration,
  );
}

$("openCalibrationCamera").addEventListener(
  "click",
  startCalibrationCamera,
);

$("closeCalibrationCamera").addEventListener(
  "click",
  () => {
    closeCalibrationCamera(
      true,
    );
  },
);

$("captureCalibrationButton").addEventListener(
  "click",
  captureCalibrationFrame,
);

$("uploadCalibrationButton").addEventListener(
  "click",
  () => {
    $("calibrationFileInput").click();
  },
);

$("calibrationFileInput").addEventListener(
  "change",
  (event) => {
    handleCalibrationUpload(
      event.target.files?.[0],
    );

    event.target.value =
      "";
  },
);

$("calibrationCanvas").addEventListener(
  "click",
  calibrateAtPoint,
);

$("openVideoCameraButton").addEventListener(
  "click",
  openVideoCamera,
);

$("startRecordingButton").addEventListener(
  "click",
  startRecording,
);

$("stopRecordingButton").addEventListener(
  "click",
  stopRecording,
);

$("closeVideoCameraButton").addEventListener(
  "click",
  () => {
    closeVideoCamera(
      true,
    );
  },
);

$("uploadVideoButton").addEventListener(
  "click",
  () => {
    $("videoFileInput").click();
  },
);

$("videoFileInput").addEventListener(
  "change",
  (event) => {
    handleVideoUpload(
      event.target.files?.[0],
    );

    event.target.value =
      "";
  },
);

$("analyseButton").addEventListener(
  "click",
  analyseDelivery,
);

attachChartInteraction(
  $("speedChart"),
  $("speedChartTooltip"),
);

attachChartInteraction(
  $("accelerationChart"),
  $("accelerationChartTooltip"),
);

attachChartInteraction(
  $("swingChart"),
  $("swingChartTooltip"),
);

window.addEventListener(
  "resize",
  () => {
    if (
      state.chartPayload
    ) {
      redrawChart(
        $("speedChart"),
      );

      redrawChart(
        $("accelerationChart"),
      );

      redrawChart(
        $("swingChart"),
      );
    }
  },
);

window.addEventListener(
  "beforeunload",
  () => {
    stopStream(
      state.calibrationStream,
    );

    stopStream(
      state.videoStream,
    );

    if (
      state.videoObjectUrl
    ) {
      URL.revokeObjectURL(
        state.videoObjectUrl,
      );
    }
  },
);

updateSpecReadout();
