from __future__ import annotations
import asyncio, base64, json, secrets, shutil, sys, tempfile, threading, traceback
from dataclasses import dataclass, field
from pathlib import Path
import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_select_extension import detect_landmarks_all, detect_landmarks_crop, make_multi_landmarker, run_pipeline_selected

STATIC_DIR = Path(__file__).resolve().parent / 'static'
MAX_FILES = 25_000
app = FastAPI(title='Face-Movie Face Select')
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

@dataclass
class Job:
    id: str; work_dir: Path; input_dir: Path; output_path: Path
    stage: str='uploaded'; current:int=0; total:int=0; error:str|None=None
    encoder:str|None=None; skipped:int=0; used:int=0; pose_filtered:int=0; duration_s:float=0.0
    lock: threading.Lock=field(default_factory=threading.Lock)
JOBS: dict[str, Job] = {}
RENDER_LOCK = threading.Lock()

@app.get('/', response_class=HTMLResponse)
async def index(): return HTMLResponse((STATIC_DIR/'index.html').read_text(encoding='utf-8'))

@app.post('/api/analyze')
async def analyze(request: Request):
    form=await request.form(max_files=MAX_FILES+1,max_fields=MAX_FILES+8)
    files=[f for f in form.getlist('files') if isinstance(f,UploadFile)]
    if not 2 <= len(files) <= MAX_FILES: raise HTTPException(400,'need 2 or more images')
    jid=secrets.token_hex(8); wd=Path(tempfile.mkdtemp(prefix=f'fm-{jid}-')); inp=wd/'input'; inp.mkdir(); saved=0
    for upload in files:
        raw=upload.filename or 'img.jpg'; ext=Path(raw).suffix.lower()
        if ext not in {'.jpg','.jpeg','.png'}: continue
        stem=Path(raw).stem or 'img'; target=inp/f'{stem}{ext}'; n=0
        while target.exists(): n+=1; target=inp/f'{stem}-{n}{ext}'
        target.write_bytes(await upload.read()); saved+=1
    if saved < 2: shutil.rmtree(wd,ignore_errors=True); raise HTTPException(400,'need at least 2 valid images')
    job=Job(jid,wd,inp,wd/'out.mp4'); JOBS[jid]=job
    landmarker=make_multi_landmarker()
    items=[]
    try:
        for p in sorted(inp.iterdir()):
            img=cv2.imread(str(p)); faces=[]; preview=''
            if img is not None:
                h,w=img.shape[:2]
                for idx,(lm,_matrix) in enumerate(detect_landmarks_all(img,landmarker)):
                    x1,y1=lm.min(axis=0); x2,y2=lm.max(axis=0)
                    pad_x=(x2-x1)*.12; pad_y=(y2-y1)*.18
                    x1=max(0,x1-pad_x); y1=max(0,y1-pad_y); x2=min(w,x2+pad_x); y2=min(h,y2+pad_y)
                    faces.append({'index':idx,'x':float(x1/w),'y':float(y1/h),'w':float((x2-x1)/w),'h':float((y2-y1)/h)})
                s=min(1.0,900/max(h,w)); prev=cv2.resize(img,None,fx=s,fy=s) if s<1 else img
                ok,buf=cv2.imencode('.jpg',prev,[cv2.IMWRITE_JPEG_QUALITY,82])
                if ok: preview='data:image/jpeg;base64,'+base64.b64encode(buf).decode()
            items.append({'filename':p.name,'faces':faces,'preview':preview})
    finally: landmarker.close()
    return {'job_id':jid,'items':items}

@app.post('/api/crop/{job_id}')
async def crop_detect(job_id:str, request:Request):
    job=JOBS.get(job_id)
    if not job: raise HTTPException(404,'no such job')
    data=await request.json()
    filename=str(data.get('filename',''))
    try:
        crop=tuple(float(data[k]) for k in ('x','y','w','h'))
    except (KeyError,TypeError,ValueError) as e:
        raise HTTPException(400,'invalid crop') from e
    x,y,cw,ch=crop
    if not (0<=x<1 and 0<=y<1 and 0<cw<=1 and 0<ch<=1 and x+cw<=1.001 and y+ch<=1.001):
        raise HTTPException(400,'crop outside image')
    path=job.input_dir/filename
    if not path.exists() or path.parent!=job.input_dir: raise HTTPException(404,'image not found')
    img=cv2.imread(str(path))
    if img is None: raise HTTPException(400,'image unreadable')
    lm=make_multi_landmarker(); faces=[]
    try:
        h,w=img.shape[:2]
        for idx,(pts,_matrix) in enumerate(detect_landmarks_crop(img,lm,crop)):
            x1,y1=pts.min(axis=0); x2,y2=pts.max(axis=0)
            pad_x=(x2-x1)*.12; pad_y=(y2-y1)*.18
            x1=max(0,x1-pad_x); y1=max(0,y1-pad_y); x2=min(w,x2+pad_x); y2=min(h,y2+pad_y)
            faces.append({'index':idx,'x':float(x1/w),'y':float(y1/h),'w':float((x2-x1)/w),'h':float((y2-y1)/h)})
    finally: lm.close()
    return {'filename':filename,'faces':faces}

@app.post('/api/render/{job_id}')
async def render(job_id:str, request:Request):
    job=JOBS.get(job_id)
    if not job: raise HTTPException(404,'no such job')
    data=await request.json()
    try:
        scale=float(data.get('scale',1)); fpp=int(data.get('frames_per_pair',30)); fps=int(data.get('fps',30)); hold=float(data.get('hold_seconds',0))
        overlay=bool(data.get('overlay',True)); front=bool(data.get('front_facing_only',False)); selections={str(k):int(v) for k,v in dict(data.get('selections',{})).items()}; crops={str(k):tuple(float(z) for z in v) for k,v in dict(data.get('manual_crops',{})).items()}; excluded={str(x) for x in data.get('excluded_files',[])}
    except (TypeError,ValueError) as e: raise HTTPException(400,f'invalid parameter: {e}') from e
    if not .1<=scale<=1: raise HTTPException(400,'scale must be 0.1..1.0')
    if not 1<=fpp<=60: raise HTTPException(400,'frames_per_pair must be 1..60')
    if not 10<=fps<=60: raise HTTPException(400,'fps must be 10..60')
    if not 0<=hold<=60: raise HTTPException(400,'pause must be 0..60 seconds')
    asyncio.get_running_loop().run_in_executor(None,_run,job,scale,fpp,fps,overlay,front,selections,crops,excluded,hold)
    return {'job_id':job.id}

def _run(job,scale,fpp,fps,overlay,front,selections,crops,excluded,hold):
    def progress(stage,current,total):
        with job.lock: job.stage=stage; job.current=current; job.total=total
    try:
        with RENDER_LOCK:
            with job.lock: job.stage='starting'; job.error=None
            r=run_pipeline_selected(job.input_dir,job.output_path,scale=scale,frames_per_pair=fpp,fps=fps,overlay=overlay,front_facing_only=front,on_progress=progress,face_selections=selections,manual_crops=crops,excluded_files=excluded,hold_seconds=hold)
            with job.lock:
                job.stage='done'; job.encoder=r.encoder; job.used=len(r.used_files); job.skipped=len(r.skipped_files); job.pose_filtered=len(r.pose_filtered_files); job.duration_s=r.duration_seconds
    except Exception as e:
        traceback.print_exc()
        with job.lock: job.stage='error'; job.error=str(e) or e.__class__.__name__

@app.get('/api/events/{job_id}')
async def events(job_id:str):
    if job_id not in JOBS: raise HTTPException(404)
    job=JOBS[job_id]
    async def stream():
        last=None
        while True:
            with job.lock: snap=(job.stage,job.current,job.total,job.error,job.encoder,job.used,job.skipped,job.pose_filtered,job.duration_s)
            if snap!=last:
                yield 'data: '+json.dumps(dict(stage=snap[0],current=snap[1],total=snap[2],error=snap[3],encoder=snap[4],used=snap[5],skipped=snap[6],pose_filtered=snap[7],duration_s=snap[8]))+'\n\n'; last=snap
            if snap[0] in ('done','error'): return
            await asyncio.sleep(.4)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.get('/api/download/{job_id}')
async def download(job_id:str):
    job=JOBS.get(job_id)
    if not job: raise HTTPException(404)
    if job.stage!='done': raise HTTPException(409,'not ready')
    return FileResponse(job.output_path,media_type='video/mp4',filename='face-movie.mp4')

@app.delete('/api/job/{job_id}')
async def delete(job_id:str):
    job=JOBS.pop(job_id,None)
    if not job: raise HTTPException(404)
    shutil.rmtree(job.work_dir,ignore_errors=True); return {'ok':True}
