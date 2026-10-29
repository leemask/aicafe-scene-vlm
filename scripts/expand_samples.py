"""Add 32 nonoverlapping, 20-second clips from eight additional MEVA cameras.
Run: python scripts/expand_samples.py --work PATH
Original downloads (~1 GB) stay in PATH, outside the released application.
"""
import argparse,concurrent.futures,hashlib,json,shutil,sys,time,urllib.request
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core import ROOT,video_info,extract_frames,ffmpeg,atomic_json
import subprocess
BASE='https://mevadata-public-01.s3.amazonaws.com/'
SOURCES=[
('2018-03-07.16-50-00.16-55-00.admin.G329',[20,82,145,230],'출입·이동'),
('2018-03-07.16-50-01.16-55-01.school.G328',[38,62,145,258],'차량 주변'),
('2018-03-07.16-50-07.16-55-07.hospital.G301',[20,90,145,230],'출입·이동'),
('2018-03-07.17-05-00.17-10-00.school.G299',[20,140,190,250],'출입·이동'),
('2018-03-11.11-30-01.11-35-01.school.G421',[28,65,150,240],'물건·일상 행동'),
('2018-03-11.11-50-01.11-55-01.school.G424',[27,58,145,230],'차량 주변'),
('2018-03-12.10-00-00.10-05-00.hospital.G436',[20,133,190,250],'차량 주변'),
('2018-03-12.10-00-01.10-05-01.school.G423',[14,47,89,210],'물건·일상 행동'),
]
CAT={'Enter_Facility':'entry_exit','Exit_Facility':'entry_exit','Object_Transfer':'object_handling','Heavy_Carry':'object_handling','Set_Down_Object':'object_handling','Pick_Up_Object':'object_handling','Load_Vehicle':'vehicle_loading','Unload_Vehicle':'vehicle_loading'}
def download(url,path):
 if path.exists():return
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=90) as response,path.with_suffix('.part').open('wb') as out:shutil.copyfileobj(response,out,1024*1024)
   path.with_suffix('.part').replace(path);return
  except Exception:
   if attempt==2:raise
   time.sleep(1+attempt)
def process(item):
 number,(name,starts,focus)=item;date=name[:10];hour=name.split('.')[1][:2]
 raw_url=BASE+f'drops-123-r13/{date}/{hour}/{name}.r13.avi'
 annotation_url=BASE+f'examples/annotations/{date}/{hour}/{name}.activities.yml'
 raw=work/(name+'.avi');annotation=ROOT/'samples/annotations'/(name+'.activities.yml')
 download(annotation_url,annotation);download(raw_url,raw)
 info=video_info(raw);sha=hashlib.sha256(raw.read_bytes()).hexdigest();acts=yaml.safe_load(annotation.read_text())
 entries=[]
 for n,start in enumerate(starts,1):
  cid=f'meva-{number:02}-{n:02}';dest=ROOT/'samples'/(cid+'.mp4');length=20
  if start+length>info['duration']:raise ValueError('Source too short '+name)
  if not dest.exists():
   temp=dest.with_name(cid+'.partial.mp4')
   subprocess.run([ffmpeg(),'-y','-v','error','-i',str(raw),'-ss',str(start),'-t',str(length),'-an','-vf','scale=960:-2','-r','15','-c:v','libx264','-threads','2','-preset','fast','-crf','27','-movflags','+faststart',str(temp)],check=True)
   vi=video_info(temp)
   if abs(vi['duration']-length)>.1:raise ValueError('Clip duration mismatch')
   temp.replace(dest)
  extract_frames(dest,0,length,1)[0][1].save(ROOT/'samples'/(cid+'.jpg'),quality=85)
  truth=[]
  for item in acts:
   if 'act' not in item:continue
   act=item['act'];label=next(iter(act['act3']));begin,end=act['timespan'][0]['tsr0'];begin/=info['fps'];end/=info['fps']
   if end>start and begin<start+length:truth.append({'activity':label,'category':CAT.get(label,'other'),'start':round(max(0,begin-start),3),'end':round(min(length,end-start),3),'source':'MEVA example annotation; incomplete coverage'})
  entries.append({'id':cid,'title':f'{name.split(".")[-1]} · 구간 {n:02}','file':f'samples/{cid}.mp4','thumbnail':f'samples/{cid}.jpg',**video_info(dest),'source':'MEVA KF1','source_url':raw_url,'source_name':name,'source_start':start,'source_end':start+length,'source_fps':info['fps'],'source_sha256':sha,'annotation_url':annotation_url,'ground_truth':truth,'annotation_status':'공개 부분 주석 / 사람 검토 전','license':'CC BY 4.0','modifications':'20-second excerpt, resized to 960px, 15fps H.264, audio removed; no ground-truth overlays','split_group':name,'camera_id':name.split('.')[-1],'recording_date':date,'test_focus':focus,'sample_set':'추가 샘플','selection_note':'검증 주제는 카메라별 시험 목적이며 이 구간의 행동 정답이 아닙니다.'})
  print(cid,'ready',len(truth),'partial annotations',flush=True)
 return entries
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);args=p.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:added=[c for group in pool.map(process,enumerate(SOURCES,5)) for c in group]
 m=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'));ids={c['id'] for c in added}
 old=[c for c in m['clips'] if c['id'] not in ids]
 for c in old:
  c.update(camera_id=c['source_name'].split('.')[-1],recording_date=c['source_name'][:10],sample_set='기존 샘플',test_focus={'G326':'출입·이동','G331':'물건·일상 행동','G340':'차량 주변','G336':'차량 주변'}.get(c['source_name'].split('.')[-1],'기타'))
 m['clips']=old+added;m['version']=2;m['notice']='64 clips from 12 source videos and 12 cameras in one MEVA facility. New 20-second clips do not overlap within a source. Older clips may overlap. Partial annotations are not exhaustive; unannotated does not mean negative. Not a production benchmark.'
 atomic_json(ROOT/'samples/manifest.json',m)
 print('Expansion complete:',len(m['clips']),flush=True)
