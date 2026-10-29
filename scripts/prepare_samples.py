"""Rebuild 32 clips from downloaded MEVA r13 sources. Requires --research dir."""
import argparse,sys,json,subprocess,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core import ROOT,video_info,ffmpeg,atomic_json,extract_frames
import yaml
p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);args=p.parse_args()
sources=json.loads((args.research/'sources.json').read_text())
source_starts=[[10,40,75,82,90,130,190,250],[10,25,35,39,45,80,160,240],[10,40,62,68,74,130,142,220],[10,70,155,165,175,180,188,250]]
category={'Enter_Facility':'entry_exit','Exit_Facility':'entry_exit','Object_Transfer':'object_handling','Heavy_Carry':'object_handling','Load_Vehicle':'vehicle_loading','Unload_Vehicle':'vehicle_loading'}
clips=[]
for sn,(source,starts) in enumerate(zip(sources,source_starts)):
 raw=args.research/'raw'/(source['name']+'.avi');info=video_info(raw)
 annotation=args.research/'annotations'/(source['name']+'.activities.yml')
 truth=yaml.safe_load(annotation.read_text())
 dest=ROOT/'samples/annotations';dest.mkdir(exist_ok=True)
 shutil.copyfile(annotation,dest/annotation.name)
 for j,start in enumerate(starts):
  cid=f'meva-{sn+1:02}-{j+1:02}';out=ROOT/'samples'/(cid+'.mp4');duration=min(12,info['duration']-start)
  if not out.exists():
   subprocess.run([ffmpeg(),'-y','-v','error','-i',str(raw),'-ss',str(start),'-t',str(duration),'-an','-vf','scale=960:-2','-r','15','-c:v','libx264','-threads','2','-preset','fast','-crf','25','-movflags','+faststart',str(out)],check=True)
  gt=[]
  for item in truth:
   if 'act' not in item:continue
   a=item['act'];label=next(iter(a['act3']));st,en=a['timespan'][0]['tsr0'];st/=info['fps'];en/=info['fps']
   if en>start and st<start+duration:
    gt.append({'activity':label,'category':category.get(label,'other'),'start':round(max(0,st-start),3),'end':round(min(duration,en-start),3),'source':'MEVA example annotation; incomplete coverage'})
  extract_frames(out,0,duration,1)[0][1].save(ROOT/'samples'/(cid+'.jpg'),quality=85)
  clips.append({'id':cid,'title':f'카메라 {sn+1:02} · 구간 {j+1:02}','file':f'samples/{cid}.mp4','thumbnail':f'samples/{cid}.jpg',**video_info(out),'source':'MEVA KF1','source_url':source['url'],'source_name':source['name'],'source_start':start,'source_end':start+duration,'source_fps':info['fps'],'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'ground_truth':gt,'annotation_status':'공개 부분 주석 / 사람 검토 전','license':'CC BY 4.0','modifications':'12-second excerpt, resized to 960px, 15fps H.264, audio removed; no ground-truth overlays','split_group':source['name']})
  print(cid,len(gt),flush=True)
manifest={'title':'AICAFE Scene Lab — MEVA pilot set','version':1,'clips':clips,'notice':'32 clips from 4 source videos. Related windows are not independent. Unannotated does not mean negative. Not a production benchmark. Ground truth is never sent to the VLM.','attribution':'MEVA by Kitware Inc. and IARPA, CC BY 4.0; Corona et al., WACV 2021. https://mevadata.org/'}
atomic_json(ROOT/'samples/manifest.json',manifest)
