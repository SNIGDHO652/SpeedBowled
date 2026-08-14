# Speedbowled

Speedbowled is a Flask/OpenCV bowling-analysis web app.

## Workflow

1. Enter the ball diameter. Default: **7.25 cm**.
2. Configure HSV detection tolerances:
   - H: **10%**
   - S: **40%**
   - V: **40%**
3. Place the ball exactly **1 m** from the camera.
4. Capture/upload a calibration image and click the ball.
5. Focal length is calculated with:

   `f = pixel_diameter * distance / real_diameter`

6. Record/upload a bowling video without changing camera lens/zoom.
7. Choose analysis start/end instants.
   - Start defaults to **0 s**.
   - End defaults to the **video duration**.
   - Only frames inside this interval are tracked and analysed.
8. Speedbowled tracks the selected coloured moving ball with HSV,
   motion filtering and SIFT-assisted association.
9. Pinhole reconstruction:

   `X = (u-cx)R/r`

   `Y = (v-cy)R/r`

   `Z = fR/r`

10. Velocity and acceleration use finite differences only.
11. Metrics:
    - Speed: km/h
    - Acceleration: m/s²
    - Swing: 3-D velocity direction-change rate in °/s
12. Bounce/abrupt-change detection uses span-1 finite-difference
    velocity changes with a robust median/MAD threshold.
13. Detected bounces are marked on all graphs.
14. Speed samples near detected bounces remain visible but are excluded
    from speed min/max/average.

## HSV tolerance meaning

The percentages are converted to channel ranges:

- H tolerance percentage is applied to OpenCV's 0–179 circular hue range.
- S tolerance percentage is applied to the 0–255 saturation range.
- V tolerance percentage is applied to the 0–255 value range.

For example, H=10% means approximately ±18 OpenCV hue units around the
clicked hue, with wrap-around.

## Interactive charts

- Hover or tap to inspect the nearest graph sample.
- Drag horizontally to select a time interval.
- The selected interval immediately shows:
  - minimum
  - maximum
  - average
  - number of samples
- On the speed chart, bounce-adjacent samples are automatically omitted
  from dragged-interval statistics too.
- Double-click a chart to clear its selected range.

## Camera controls

Both calibration and bowling camera flows include a **Close camera** option.
Closing while a bowling recording is active discards that active recording.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Render

The repository includes `render.yaml`.

Manual settings:

- Runtime: Python
- Build: `pip install -r requirements.txt`
- Start:
  `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180`
- Health check: `/api/health`

## Environment variables

- `DERIVATIVE_SPAN_FRAMES=3`
- `MIN_METRIC_RADIUS_PX=2.5`
- `BOUNCE_MIN_CHANGE_RATE_MPS2=30`
- `BOUNCE_MIN_ANGLE_DEG=8`
- `BOUNCE_ROBUST_SIGMA_MULTIPLIER=5`
- `BOUNCE_MERGE_WINDOW_S=0.12`
- `BOUNCE_SUMMARY_EXCLUSION_S=0.15`
- `MAX_UPLOAD_BYTES=262144000`
- `LOG_LEVEL=INFO`

## Measurement notes

Accuracy depends on exact 1 m placement, physical diameter, focal calibration,
motion blur, frame rate, ball visibility, and keeping the same camera/lens/zoom
between calibration and analysis.

Bounce detection means an abrupt measured velocity-vector change; it does not
fit a parabola or assume a particular ballistic trajectory.
