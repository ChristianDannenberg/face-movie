# Face-Movie – Face Select Extension

A privacy-friendly local fork of **Face-Movie** focused on real photo archives: group photos, small faces, difficult scans, and manual control over which photos actually enter the movie.

This fork keeps the original Face-Movie morphing pipeline and adds a review step before rendering.

## Added in this fork

- Detect up to **10 faces per photo** and select the correct face in the Web UI.
- Multi-pass face detection: normal, sensitive, 2× upscale, CLAHE contrast enhancement, and overlapping zoom tiles.
- Manual crop fallback for difficult or very small faces.
- **Include / exclude each photo after analysis.** Photos without a detected face start excluded; a successful manual crop enables them again.
- Configurable still-image pause between morphs, specified in seconds and converted using the selected output FPS.
- CPU-first Docker setup with `libegl1` and `libgles2`, tested for headless/NAS-style deployments.
- Processing stays local; uploaded photos are stored only in the temporary job directory.

## Web workflow

1. Select photos or a folder.
2. Click **Gesichter analysieren**.
3. Review each image:
   - click the correct face box;
   - disable **Im Video verwenden** when the photo should be skipped;
   - if no face is found, choose **Bereich wählen** and draw a generous box around the face/head.
4. Set transition frames, FPS and optional pause.
5. Render and download the MP4.

At least two photos must remain enabled.

## Docker / Synology Container Manager

```bash
docker compose up --build -d
```

The included `compose.yaml` exposes the Web UI on port **8098**:

```text
http://YOUR-NAS-IP:8098
```

No Python installation is required on the host when using Docker.

## Settings

| Setting | Meaning |
|---|---|
| Scale | Output resolution scale, `0.1`–`1.0` |
| Frames pro Übergang | Number of morph frames between two included photos |
| FPS | Output video frame rate |
| Pause zwischen Morphings | Still time after a transition, in seconds. Internally `seconds × FPS` frames are generated. |
| Dateiname einblenden | Burn the source filename into the video |
| Nur frontale Gesichter | Keep the upstream front-facing filter enabled |

## How face detection differs from upstream

The original pipeline requests one MediaPipe face. This fork uses a review-oriented detector with multiple passes. Candidate detections are merged using bounding-box IoU so the UI can present several distinct faces without showing obvious duplicates.

Detection enhancements are used only to locate landmarks. Morphing still uses the original source image. A manual crop is also reused during rendering so a face that was found only inside the crop does not disappear during the final pass.

## Project structure

```text
face_select_extension.py   Extended detection, selection and pause integration
main.py                    Upstream Face-Movie morphing pipeline
webapp/server.py           Analyze/crop/render API
webapp/static/index.html   Web UI
Dockerfile                 Container based on the upstream image
compose.yaml               NAS-friendly deployment example
```

## Upstream

This project is based on **leachiM2k/face-movie**. The original project provides the face alignment, Delaunay triangulation, morphing and video encoding pipeline.

Upstream repository: https://github.com/leachiM2k/face-movie

When publishing this fork on GitHub, keep the upstream license and attribution from the original repository. GitHub's **Fork** button is preferable to creating an unrelated repository because it preserves the relationship to the upstream project.

## Development

Docker is the recommended development/runtime path. For native Python development, use the requirements supplied by the upstream project.

Basic syntax check:

```bash
python -m py_compile main.py face_select_extension.py webapp/server.py
```

## Status

This fork is intended for personal/local photo processing. The Web UI has no authentication; do not expose it directly to the public Internet.
