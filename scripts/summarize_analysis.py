"""Validate all 64 completed inference records and write a factual report."""
import json,hashlib,statistics
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'));clips=m['clips'];records=[];batch_times={}
for c in clips:
 p=ROOT/'results'/('qwen-full-'+c['id']+'.json')
 if not p.exists():raise RuntimeError('Analysis incomplete: '+c['id'])
 j=json.loads(p.read_text(encoding='utf-8'));s=j['segments'][0]
 assert j['status']=='completed' and j['clip_id']==c['id']
 assert j['input_video_sha256']==hashlib.sha256((ROOT/c['file']).read_bytes()).hexdigest()
 assert len(s['frame_times'])==3 and all(0<=t<c['duration'] for t in s['frame_times'])
 assert len(s['evidence_frames'])==3
 for f in s['evidence_frames']:assert (ROOT/f['file'].lstrip('/')).exists()
 batch_times[s['batch_id']]=s['batch_wall_seconds'];records.append((c,j,s))
labels={'entry_exit':'출입·이동','object_handling':'물건 취급','vehicle_loading':'차량 상하차','other':'기타 관찰','uncertain':'판단 보류'}
counts=Counter(s['category'] for _,_,s in records)
qc_path=ROOT/'reports/spotchecks.json';qc=json.loads(qc_path.read_text()) if qc_path.exists() else []
summary={'clips':len(records),'model':'Qwen/Qwen3-VL-2B-Instruct','model_revision':records[0][1]['inference_config']['revision'],'source_videos':len({c['source_name'] for c in clips}),'cameras':len({c['camera_id'] for c in clips}),'input_video_seconds':sum(c['duration'] for c in clips),'sampled_frames':3*len(records),'valid_json':sum(s['parse_ok'] for _,_,s in records),'predicted_categories':dict(counts),'batch_count':len(batch_times),'inference_wall_seconds':round(sum(batch_times.values()),3),'amortized_seconds_per_clip':round(sum(batch_times.values())/len(records),3),'timing_note':'Model inference and processor time; excludes download, load, frame extraction, review and packaging. Batch averages are not single-video latency.','evaluation_note':'No accuracy/recall/false-positive rate estimated. Partial public annotations are not exhaustive; model predictions are not ground truth.','spotchecks':qc}
coverage={'intervals':0,'with_sample':0,'without_sample':0,'missed':[]}
for c,j,seg in records:
 for g in c['ground_truth']:
  if g['category'] not in {'entry_exit','object_handling','vehicle_loading'}:continue
  covered=any(g['start']<=t<=g['end'] for t in seg['frame_times'])
  coverage['intervals']+=1;coverage['with_sample' if covered else 'without_sample']+=1
  if not covered:coverage['missed'].append({'clip_id':c['id'],'activity':g['activity'],'start':g['start'],'end':g['end'],'frame_times':seg['frame_times']})
coverage['note']='Clip-level partial reference intervals; overlapping clips may duplicate source events. Timestamp coverage is not recognition accuracy and depends on annotation alignment.'
summary['reference_interval_coverage']=coverage
(ROOT/'reports').mkdir(exist_ok=True);(ROOT/'reports/analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 64개 샘플 실제 VLM 분석 결과','', '**전체 64개에 실제 모델 추론을 완료했습니다. 동작 인식 정확도 검증을 통과했다는 뜻은 아닙니다.**','',
 f'- 모델: {summary["model"]}; revision `{summary["model_revision"]}`.',
 '- CPU, bfloat16, SDPA, 8스레드, 배치 4개, greedy decoding, 출력 한도 260토큰.',
 '- 영상당 시간순 프레임 3개, 최대 576×324 입력. 정답 주석·파일명·카메라 정보는 모델에 제공하지 않았습니다.',
 f'- 입력 영상 총 {summary["input_video_seconds"]:.0f}초, 추출 프레임 {summary["sampled_frames"]}장. 모든 중간 동작을 연속 분석한 것은 아닙니다.',
 f'- JSON 파싱 성공 {summary["valid_json"]}/{summary["clips"]}. 형식 성공은 내용 정확도와 다릅니다.',
 f'- 추론 배치 총 {summary["inference_wall_seconds"]:.1f}초, 영상당 배치 평균 {summary["amortized_seconds_per_clip"]:.2f}초. 모델 다운로드·로딩·프레임 추출 시간 제외.',
 '- 기존 분석 9개는 보존했습니다. 새 전체 분석 64개는 `results/qwen-full-*.json`이며 화면은 최신 결과를 우선 표시합니다.',
 '- 모델 설명은 확인 전 추정이며 자동 경보나 현장 의사결정에 바로 사용하기 어렵습니다.','',
 '## 모델이 부여한 분류 수','', '| 모델 추정 분류 | 영상 수 |','|---|---:|']
lines += [f'| {labels[k]} | {v} |' for k,v in counts.items()]
lines += ['', '이 수치는 실제 사건 발생 건수가 아닙니다. 공개 부분 주석과 미검토 구간이 섞여 있어 정확도·재현율을 계산하지 않았습니다.', '', '## 샘플 프레임 대조 검토','', '아래는 ChatGPT의 프레임 기반 보조 검토입니다. 사람의 전수 정답 라벨이나 전체 영상의 연속 검토가 아닙니다.','', '| 영상 | 프레임 관찰 및 모델 대조 |','|---|---|']
lines += [f'| {q["clip_id"]} | {q["note"]} |' for q in qc]
lines += ['', '## 샘플링이 짧은 행동을 놓치는 문제','', f'출입·물건 취급·상하차에 해당하는 클립별 공개 주석 구간 {coverage["intervals"]}개 중 {coverage["without_sample"]}개는 분석 프레임 시각이 구간 안에 한 번도 들어가지 않았습니다. 이는 참고 주석의 시간 기준 입력 커버리지이며 모델 정확도가 아닙니다. 겹치는 기존 클립 때문에 같은 원본 사건이 중복 집계될 수 있습니다.', '', '예: meva-10-01의 Load_Vehicle 주석은 11.33–14.20초지만 입력은 3.33·10·16.67초입니다. 모델 크기 비교와 함께 시간 샘플링 간격도 개선해야 합니다.']
lines += ['', '## 영상별 실제 출력','', '공개 주석은 참고용 부분 주석이며, 빈 칸이 정상이나 사건 부재를 뜻하지 않습니다.','', '| 영상 | 카메라 | 모델 분류 | 실제 생성 설명 | 공개 부분 주석 |','|---|---|---|---|---|']
def clean(s):return str(s).replace('|','/').replace('\n',' ')
for c,j,s in records:
 refs=', '.join(dict.fromkeys(g['activity'] for g in c['ground_truth'])) or '주석 없음(미확정)'
 lines.append(f'| {c["id"]} | {c["camera_id"]} | {labels[s["category"]]} | {clean(s["summary_ko"])} | {clean(refs)} |')
lines += ['', '## 파일과 재현','', '- 결과 JSON에는 원문, 추론 조건, 모델 revision, 입력 영상 SHA256, 실제 프레임 시각이 있습니다.', '- 각 `results/qwen-full-영상ID/` 폴더에 분석 프레임 3장을 저장했습니다. 화면에서 각 프레임을 클릭하면 확대할 수 있습니다.', '- `QWEN_VLM_PATH` 지정 후 `python scripts/analyze_all.py --batch-size 4 --threads 8`로 재현합니다. 이미 저장된 결과는 건너뜁니다. 새 결과 생성 전 기존 파일을 별도 보관하세요.', '- 소형 모델·입력 해상도·프레임 샘플링의 영향을 분리하는 비교 실험은 하지 않았습니다. 더 큰 모델 및 탐지·추적기와 같은 정답셋에서 비교하는 것이 다음 단계입니다.']
(ROOT/'ANALYSIS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
