import sys,json,os,argparse
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core import ROOT,Provider,extract_frames,atomic_json
parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=32);args=parser.parse_args()
m=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))
p=Provider('qwen');print('Qwen ready',flush=True)
order=[c for c in m['clips'] if c['id'] in ['meva-01-04','meva-02-04','meva-03-04','meva-04-06']]+[c for c in m['clips'] if c['id'] not in ['meva-01-04','meva-02-04','meva-03-04','meva-04-06']]
for c in order[:args.limit]:
 jid='qwen-'+c['id'];dest=ROOT/'results'/(jid+'.json')
 if dest.exists():continue
 frames=extract_frames(ROOT/c['file'],0,c['duration'],3)
 result=p.analyze(frames)|{'id':'0','start':0,'end':c['duration'],'thumbnail':'/'+c['thumbnail']}
 job={'id':jid,'clip_id':c['id'],'status':'completed','duration':c['duration'],'created_at':datetime.now(timezone.utc).isoformat(),'provider':'qwen','model':'Qwen/Qwen3-VL-2B-Instruct','segments':[result],'progress':1,'total':1,'sampled_until':c['duration'],'scope':'3 frames per clip; 576px maximum model input. Ground truth and filenames excluded. Predicted category is not confirmed truth.','inference_config':{'dtype':'bfloat16','attention':'sdpa','threads':4,'max_new_tokens':260,'sampling':False}}
 atomic_json(dest,job);print(c['id'],result['inference_seconds'],result['category'],result['summary_ko'],result['raw'][:500],flush=True)
