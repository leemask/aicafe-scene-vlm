"""Reproducible full-set Qwen analysis; existing results are preserved.
QWEN_VLM_PATH=/path/to/model python scripts/analyze_all.py --batch-size 4 --threads 8
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core import ROOT,Provider,extract_frames,atomic_json,PROMPT,normalize
import torch
p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=4);p.add_argument('--threads',type=int,default=8);p.add_argument('--limit',type=int,default=64);args=p.parse_args()
manifest=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))
clips=manifest['clips'][:args.limit]
pending=[c for c in clips if not (ROOT/'results'/('qwen-full-'+c['id']+'.json')).exists()]
provider=Provider('qwen');torch.set_num_threads(args.threads)
provider.processor.tokenizer.padding_side='left'
model_path=Path(provider.model);revision=(model_path/'revision.txt').read_text().strip() if (model_path/'revision.txt').exists() else None
print('MODEL LOADED; pending',len(pending),'batch',args.batch_size,'threads',args.threads,flush=True)
for offset in range(0,len(pending),args.batch_size):
 batch=pending[offset:offset+args.batch_size];messages=[];sampled=[]
 for clip in batch:
  frames=extract_frames(ROOT/clip['file'],0,clip['duration'],3);sampled.append(frames);content=[]
  for t,im in frames:
   im=im.copy();im.thumbnail((576,324));content.extend([{'type':'text','text':f'Frame at {t:.2f}s'},{'type':'image','image':im}])
  content.append({'type':'text','text':PROMPT+' Keep each JSON text field under 40 Korean characters.'})
  messages.append([{'role':'user','content':content}])
 started=time.perf_counter()
 inputs=provider.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True)
 with torch.inference_mode():output=provider.engine.generate(**inputs,max_new_tokens=260,do_sample=False)
 raw_outputs=provider.processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 elapsed=round(time.perf_counter()-started,3);batch_id='full-'+batch[0]['id']
 for clip,frames,raw in zip(batch,sampled,raw_outputs):
  jid='qwen-full-'+clip['id'];folder=ROOT/'results'/jid;folder.mkdir(exist_ok=True);evidence=[]
  for n,(t,im) in enumerate(frames):
   im=im.copy();im.thumbnail((576,324));path=folder/f'frame-{n}.jpg';im.save(path,quality=82)
   evidence.append({'time':round(t,3),'file':f'/results/{jid}/frame-{n}.jpg'})
  segment=normalize(raw)|{'id':'0','start':0,'end':clip['duration'],'thumbnail':evidence[1]['file'],'evidence_frames':evidence,'frame_times':[round(t,3) for t,_ in frames],'review':'unreviewed','provider':'qwen','model':'Qwen/Qwen3-VL-2B-Instruct','inference_seconds':round(elapsed/len(batch),3),'timing_kind':'amortized_batch','batch_wall_seconds':elapsed,'batch_size':len(batch),'batch_id':batch_id}
  job={'id':jid,'clip_id':clip['id'],'status':'completed','duration':clip['duration'],'created_at':datetime.now(timezone.utc).isoformat(),'provider':'qwen','model':'Qwen/Qwen3-VL-2B-Instruct','segments':[segment],'progress':1,'total':1,'sampled_until':clip['duration'],'scope':'Three sampled frames describe a whole clip; not continuous motion detection. Ground truth, metadata and file names are not sent to the model.','inference_config':{'dtype':'bfloat16','attention':'sdpa','threads':args.threads,'max_new_tokens':260,'sampling':False,'max_image_size':[576,324],'frames_per_clip':3,'batch_size':len(batch),'padding_side':'left','revision':revision},'input_video_sha256':hashlib.sha256((ROOT/clip['file']).read_bytes()).hexdigest()}
  atomic_json(ROOT/'results'/(jid+'.json'),job)
  print(clip['id'],segment['category'],segment['summary_ko'],'parse',segment['parse_ok'],flush=True)
 print('BATCH',offset+len(batch),'/',len(pending),'wall',elapsed,flush=True)
print('DONE',len(clips),'clips',flush=True)
