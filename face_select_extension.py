from __future__ import annotations
import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import main as upstream

MAX_FACES = 10
NORMAL_CONFIDENCE = 0.50
SENSITIVE_CONFIDENCE = 0.25


def _new_landmarker(num_faces: int, confidence: float):
    model_path = str(upstream.ensure_model())
    Delegate = mp_tasks.BaseOptions.Delegate
    forced = os.environ.get('FACE_MOVIE_DELEGATE', 'cpu').lower()
    order = (Delegate.GPU,) if forced == 'gpu' else ((Delegate.CPU,) if forced == 'cpu' else (Delegate.GPU, Delegate.CPU))
    last = None
    for delegate in order:
        try:
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path, delegate=delegate),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=num_faces,
                min_face_detection_confidence=confidence,
                min_face_presence_confidence=0.35 if confidence < 0.5 else 0.5,
                min_tracking_confidence=0.35 if confidence < 0.5 else 0.5,
                output_facial_transformation_matrixes=True,
            )
            return mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception as exc:
            last = exc
    raise RuntimeError(f'could not initialize MediaPipe FaceLandmarker: {last}')


class MultiPassLandmarker:
    """Owns normal and sensitive MediaPipe detectors; close() closes both."""
    def __init__(self, num_faces: int = MAX_FACES):
        self.normal = _new_landmarker(num_faces, NORMAL_CONFIDENCE)
        try:
            self.sensitive = _new_landmarker(num_faces, SENSITIVE_CONFIDENCE)
        except Exception:
            self.normal.close()
            raise

    def close(self):
        self.normal.close()
        self.sensitive.close()


def make_multi_landmarker(num_faces: int = MAX_FACES):
    return MultiPassLandmarker(num_faces)


def _detect_one(image_bgr: np.ndarray, detector, original_w: int, original_h: int):
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba))
    matrices = result.facial_transformation_matrixes or []
    out = []
    for idx, face in enumerate(result.face_landmarks or []):
        # MediaPipe coordinates are normalized, so transformed/upscaled inputs map
        # directly back to the original image dimensions.
        coords = np.array([(p.x * original_w, p.y * original_h) for p in face[:468]], dtype=np.float32)
        matrix = np.array(matrices[idx], dtype=np.float32) if idx < len(matrices) else None
        out.append((coords, matrix))
    return out


def _clahe_bgr(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _bbox(lm: np.ndarray):
    x1, y1 = lm.min(axis=0); x2, y2 = lm.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _bbox(a); bx1, by1, bx2, by2 = _bbox(b)
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.0,ix2-ix1); ih=max(0.0,iy2-iy1); inter=iw*ih
    aa=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); ba=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    union=aa+ba-inter
    return inter/union if union > 0 else 0.0


def _append_unique(target, candidates):
    for cand in candidates:
        if all(_iou(cand[0], old[0]) < 0.55 for old in target):
            target.append(cand)
            if len(target) >= MAX_FACES:
                return



def _detect_crop(crop_bgr: np.ndarray, detector, full_w: int, full_h: int, x0: int, y0: int, crop_w: int, crop_h: int):
    """Detect in a crop and map normalized crop landmarks back to full-image pixels."""
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba))
    matrices = result.facial_transformation_matrixes or []
    out = []
    for idx, face in enumerate(result.face_landmarks or []):
        coords = np.array([
            (x0 + p.x * crop_w, y0 + p.y * crop_h) for p in face[:468]
        ], dtype=np.float32)
        coords[:, 0] = np.clip(coords[:, 0], 0, full_w - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, full_h - 1)
        matrix = np.array(matrices[idx], dtype=np.float32) if idx < len(matrices) else None
        out.append((coords, matrix))
    return out


def _zoom_tiles(image_bgr: np.ndarray):
    """Yield overlapping crops for small-face detection. No rotation is performed."""
    h, w = image_bgr.shape[:2]
    # 2x2-style windows with 50% overlap, then tighter 3x3-style windows.
    for frac in (0.68, 0.48):
        cw = max(96, min(w, int(round(w * frac))))
        ch = max(96, min(h, int(round(h * frac))))
        if cw >= w and ch >= h:
            continue
        step_x = max(1, cw // 2)
        step_y = max(1, ch // 2)
        xs = list(range(0, max(1, w - cw + 1), step_x))
        ys = list(range(0, max(1, h - ch + 1), step_y))
        if not xs or xs[-1] != w - cw: xs.append(max(0, w - cw))
        if not ys or ys[-1] != h - ch: ys.append(max(0, h - ch))
        seen=set()
        for y in ys:
            for x in xs:
                key=(x,y,cw,ch)
                if key in seen: continue
                seen.add(key)
                yield x, y, cw, ch, image_bgr[y:y+ch, x:x+cw]


def detect_landmarks_crop(image_bgr: np.ndarray, landmarker, crop: tuple[float,float,float,float]):
    """Sensitive manual-crop fallback. Crop values are normalized x,y,w,h."""
    if image_bgr is None or image_bgr.size == 0 or not isinstance(landmarker, MultiPassLandmarker):
        return []
    h,w=image_bgr.shape[:2]
    x,y,cw,ch=crop
    x0=max(0,min(w-1,int(round(x*w)))); y0=max(0,min(h-1,int(round(y*h))))
    x1=max(x0+2,min(w,int(round((x+cw)*w)))); y1=max(y0+2,min(h,int(round((y+ch)*h))))
    crop_img=image_bgr[y0:y1,x0:x1]
    if crop_img.size == 0: return []
    # Give the landmark model a large face while preserving crop geometry.
    target=1200.0
    scale=max(1.0,min(5.0,target/max(crop_img.shape[:2])))
    variants=[crop_img, _clahe_bgr(crop_img)]
    out=[]
    for base in variants:
        enlarged=cv2.resize(base,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale>1 else base
        _append_unique(out,_detect_crop(enlarged,landmarker.sensitive,w,h,x0,y0,x1-x0,y1-y0))
        if out: break
    return out

def detect_landmarks_all(image_bgr: np.ndarray, landmarker) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Multi-pass detection without rotation, including overlapping zoom tiles."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    if not isinstance(landmarker, MultiPassLandmarker):
        return _detect_one(image_bgr, landmarker, w, h)

    out = []
    enhanced = _clahe_bgr(image_bgr)
    up = cv2.resize(image_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up_enhanced = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    passes = (
        (image_bgr, landmarker.normal),
        (image_bgr, landmarker.sensitive),
        (up, landmarker.sensitive),
        (enhanced, landmarker.sensitive),
        (up_enhanced, landmarker.sensitive),
    )
    for variant, detector in passes:
        _append_unique(out, _detect_one(variant, detector, w, h))
        if len(out) >= MAX_FACES:
            return out

    # Small faces: crop overlapping windows, enlarge each window and map results
    # back to original coordinates. This is detection-only; rendering stays original.
    for x0,y0,cw,ch,crop in _zoom_tiles(image_bgr):
        scale=max(1.0,min(3.0,900.0/max(crop.shape[:2])))
        enlarged=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale>1 else crop
        _append_unique(out,_detect_crop(enlarged,landmarker.sensitive,w,h,x0,y0,cw,ch))
        if len(out) >= MAX_FACES:
            break
    return out


def run_pipeline_selected(input_dir: Path, output_path: Path, *, face_selections: dict[str, int] | None = None, manual_crops: dict[str, tuple[float,float,float,float]] | None = None,
                          excluded_files: set[str] | None = None, hold_seconds: float = 0.0, **kwargs):
    """Run upstream pipeline with deterministic per-file face choice and between-photo holds."""
    input_dir = Path(input_dir)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    selections = face_selections or {}
    crops = manual_crops or {}
    excluded = set(excluded_files or ())
    # The upstream pipeline works on a directory. Build a temporary view containing
    # only explicitly included files, so excluded photos cannot affect alignment,
    # morphing, pauses, progress counts, or output duration.
    source_input_dir = input_dir
    filtered_tmp = None
    if excluded:
        import tempfile
        import shutil
        filtered_tmp = Path(tempfile.mkdtemp(prefix='face-movie-selected-'))
        for src in files:
            if src.name not in excluded:
                shutil.copy2(src, filtered_tmp / src.name)
        input_dir = filtered_tmp
        files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(files) < 2:
        if filtered_tmp is not None:
            import shutil
            shutil.rmtree(filtered_tmp, ignore_errors=True)
        raise ValueError('At least two included images are required to render a movie.')
    fps = int(kwargs.get('fps', 30))
    hold_frames = max(0, int(round(float(hold_seconds) * fps)))
    call_index = 0

    orig_make = upstream.make_landmarker
    orig_detect = upstream.detect_landmarks
    orig_morph = upstream.morph_pair

    def make_landmarker_override():
        return make_multi_landmarker(MAX_FACES)

    def detect_override(image_bgr, landmarker):
        nonlocal call_index
        filename = files[call_index].name if call_index < len(files) else ''
        if filename in crops:
            faces = detect_landmarks_crop(image_bgr, landmarker, crops[filename])
        else:
            faces = detect_landmarks_all(image_bgr, landmarker)
        call_index += 1
        if not faces:
            return None
        selected = int(selections.get(filename, 0))
        if selected < 0 or selected >= len(faces):
            selected = 0
        return faces[selected]

    def morph_override(img1, img2, pts1, pts2, triangles, n_frames, include_endpoint):
        yield from orig_morph(img1, img2, pts1, pts2, triangles, n_frames, include_endpoint)
        if hold_frames and not include_endpoint:
            for _ in range(hold_frames):
                yield img2.copy()

    upstream.make_landmarker = make_landmarker_override
    upstream.detect_landmarks = detect_override
    upstream.morph_pair = morph_override
    try:
        result = upstream.run_pipeline(input_dir, output_path, **kwargs)
        pair_count = max(0, len(result.used_files) - 1)
        result.total_frames += max(0, pair_count - 1) * hold_frames
        return result
    finally:
        upstream.make_landmarker = orig_make
        upstream.detect_landmarks = orig_detect
        upstream.morph_pair = orig_morph
        if filtered_tmp is not None:
            import shutil
            shutil.rmtree(filtered_tmp, ignore_errors=True)
