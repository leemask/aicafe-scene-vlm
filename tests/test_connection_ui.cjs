const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

// Execute the actual UI controller with an in-memory DOM and HTTP responses.
// This tests state transitions, not visual layout or a live model connection.
async function app(protocol,respond,seed){
  const elements=new Map();
  function el(key){if(!elements.has(key))elements.set(key,{
    value:'',textContent:'',innerHTML:'',hidden:false,disabled:false,
    querySelectorAll:()=>[],classList:{toggle(){}},
    insertAdjacentHTML(){},scrollIntoView(){this.scrolled=true},
  });return elements.get(key)}
  const calls=[];
  const context=vm.createContext({
    document:{querySelector:el,querySelectorAll:()=>[]},
    window:{SCENE_BUNDLE:seed||{clips:[{id:'clip1',title:'Sample',duration:12,file:'samples/test.mp4'}],jobs:[]}},
    location:{protocol},localStorage:{getItem:()=>null,setItem(){}},
    AbortSignal,console,setTimeout:()=>1,clearTimeout(){},
    fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>respond(url,options)}},
  });
  vm.runInContext(source,context);await vm.runInContext('initialization',context);
  return {el,calls,context,run:code=>vm.runInContext(code,context)};
}
function server(url){
  if(url==='/api/status')return {busy:false};
  if(url==='/api/clips')return {clips:[{id:'clip1',title:'Sample',duration:12,file:'samples/test.mp4'}]};
  if(url==='/api/jobs')return {jobs:[]};
  throw Error('Unexpected request '+url);
}
test('file mode offers an enabled setup action without making API requests',async()=>{
  const a=await app('file:',()=>{throw Error('No API in file mode')});
  assert.equal(a.el('#analyze').disabled,false);
  assert.equal(a.el('#analyze').textContent,'분석 실행 안내 →');
  assert.equal(a.el('#setupGuide').hidden,false);
  assert.equal(a.el('#retryConnection').hidden,true);
  await a.run('run()');
  assert.equal(a.el('#setupGuide').scrolled,true);
  assert.equal(a.calls.length,0);
});
test('unreachable HTTP server shows setup and retry instead of a dead button',async()=>{
  const a=await app('http:',()=>{throw Error('Server offline')});
  assert.equal(a.el('#analyze').disabled,false);
  assert.equal(a.el('#setupGuide').hidden,false);
  assert.equal(a.el('#retryConnection').hidden,false);
  assert.match(a.el('#setupReason').textContent,/연결하지 못했습니다/);
});
test('reachable server enables actual analysis and hides setup',async()=>{
  const a=await app('http:',server);
  assert.equal(a.el('#analyze').disabled,false);
  assert.equal(a.el('#analyze').textContent,'이 영상 분석하기 →');
  assert.equal(a.el('#setupGuide').hidden,true);
  assert.equal(a.el('#connection').textContent,'로컬 서버 연결');
});
test('retry recovers after the server starts',async()=>{
  let ready=false;
  const a=await app('http:',url=>{if(!ready)throw Error('Offline');return server(url)});
  ready=true;await a.run('connect()');
  assert.equal(a.el('#setupGuide').hidden,true);
  assert.equal(a.el('#analyze').textContent,'이 영상 분석하기 →');
  assert.equal(a.el('#retryConnection').disabled,false);
});
test('active analysis prevents duplicate submit with a visible reason',async()=>{
  const a=await app('http:',url=>url==='/api/status'?{busy:true}:server(url));
  assert.equal(a.el('#analyze').disabled,true);
  assert.equal(a.el('#analyze').textContent,'분석 진행 중…');
  await a.run('run()');
  assert.equal(a.calls.some(c=>c.url==='/api/analyze'),false);
});
test('enabled action submits selected clip and becomes available after completion',async()=>{
  const job={id:'job1',clip_id:'clip1',status:'completed',segments:[],progress:1,total:1,sampled_until:12};
  const a=await app('http:',url=>url==='/api/analyze'||url==='/api/jobs/job1'?job:server(url));
  a.el('#provider').value='ollama';a.el('#model').value='qwen3-vl:2b';
  await a.run('run()');
  await new Promise(resolve=>setImmediate(resolve));
  const request=a.calls.find(c=>c.url==='/api/analyze');
  assert.equal(request.options.method,'POST');
  assert.equal(JSON.parse(request.options.body).clip_id,'clip1');
  assert.equal(a.el('#analyze').disabled,false);
  assert.equal(a.el('#analyze').textContent,'이 영상 분석하기 →');
});

const filterSeed={clips:[
 {id:'old',title:'OLD_CLIP',duration:12,file:'old.mp4',sample_set:'기존 샘플',test_focus:'출입·이동',camera_id:'G326',recording_date:'2018-03-07'},
 {id:'new',title:'NEW_CLIP',duration:20,file:'new.mp4',sample_set:'추가 샘플',test_focus:'차량 주변',camera_id:'G424',recording_date:'2018-03-11'}
],jobs:[]};
test('new sample filter hides existing clips',async()=>{
 const a=await app('file:',()=>{},filterSeed);a.el('#sampleSet').value='추가 샘플';a.run('renderList()');
 assert.match(a.el('#clipList').innerHTML,/NEW_CLIP/);assert.doesNotMatch(a.el('#clipList').innerHTML,/OLD_CLIP/);
 assert.equal(a.el('#matchedCount').textContent,'1개 표시 / 전체 2개');
});
test('test focus filter selects the requested camera topic',async()=>{
 const a=await app('file:',()=>{},filterSeed);a.el('#testFocus').value='출입·이동';a.run('renderList()');
 assert.match(a.el('#clipList').innerHTML,/OLD_CLIP/);assert.doesNotMatch(a.el('#clipList').innerHTML,/NEW_CLIP/);
});
test('camera and date metadata are searchable without model results',async()=>{
 const a=await app('file:',()=>{},filterSeed);a.el('#search').value='g424';a.run('renderList()');
 assert.match(a.el('#clipList').innerHTML,/NEW_CLIP/);assert.doesNotMatch(a.el('#clipList').innerHTML,/OLD_CLIP/);
 a.el('#search').value='2018-03-07';a.run('renderList()');assert.match(a.el('#clipList').innerHTML,/OLD_CLIP/);
});
