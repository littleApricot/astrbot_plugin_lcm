'use strict';
const $ = id => document.getElementById(id);
const state = {scope:'', mode:'context', cursor:'', searching:false, query:'', trail:[], generation:0, viewVersion:null, snapshot:null, sessions:[], saved:null, runOffset:0, timezone:'Asia/Shanghai',resetting:false};
const labels = {queued:'等待压缩',running:'压缩中',ready:'等待发布',complete:'已发布',retry:'等待重试',conflict:'来源已变化',failed:'失败',cancelled:'已取消',interrupted:'已中断',legacy_inferred:'旧历史 · 轮次为推断'};
const size = n => Number(n||0).toLocaleString('zh-CN');
function element(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
async function api(endpoint,params={}){
  const bridge=window.AstrBotPluginPage;
  if(!bridge)throw Error('请从 AstrBot 侧栏的「LCM 上下文与历史」打开此页面。');
  const result=await bridge.apiGet(endpoint,{scope:state.scope,...params});
  if(result?.status==='error')throw Error(result.message||'读取失败');
  return result?.status==='ok'||result?.status==='success'?result.data:result;
}
function notice(text,error=false){$('notice').textContent=text;$('notice').classList.toggle('error',error);}
function date(value,unknown=false){return unknown?'旧记录 · 时间未知':new Date(value*1000).toLocaleString('zh-CN',{hour12:false,timeZone:state.timezone,timeZoneName:'short'});}
function filterTime(value){
  const fields=value.split(/[-T:]/).map(Number),target=Date.UTC(fields[0],fields[1]-1,fields[2],fields[3],fields[4]);
  let guess=target;
  const format=new Intl.DateTimeFormat('en-GB',{timeZone:state.timezone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'});
  for(let i=0;i<3;i++){
    const p=Object.fromEntries(format.formatToParts(new Date(guess)).map(x=>[x.type,x.value]));
    const shown=Date.UTC(+p.year,+p.month-1,+p.day,+p.hour,+p.minute);
    if(shown===target)return new Date(guess).toISOString();
    guess+=target-shown;
  }
  throw Error('这个时间在当前时区不存在，请调整筛选时间。');
}
function highlight(el,text){
  const query=state.searching?state.query:'';
  if(!query){el.textContent=text;return;}
  let start=0,index;const lower=text.toLocaleLowerCase(),needle=query.toLocaleLowerCase();
  while((index=lower.indexOf(needle,start))>=0){el.append(document.createTextNode(text.slice(start,index)),element('mark',text.slice(index,index+needle.length)));start=index+needle.length;}
  el.append(document.createTextNode(text.slice(start)));
}
function detail(parent,title,data){const d=element('details');d.append(element('summary',title),element('pre',typeof data==='string'?data:JSON.stringify(data,null,2)));parent.append(d);}
function card(record,compact=false){
  const summary=record.level>0,message=record.payload?.message||record.message||record;
  const role=summary?'摘要':message.role||'记录';
  const root=element('article',undefined,'card'+(summary?' summary':''));root.dataset.id=record.id||'';
  const head=element('div',undefined,'card-head'),title=element('div');
  title.append(element('span',summary?'L'+record.level:role==='user'?'用户':role==='assistant'?'机器人':role==='tool'?'工具':role,'badge '+(summary?'':role==='tool'?'tool':role==='assistant'?'assistant':'raw')));
  title.append(document.createTextNode(summary?'历史摘要':record.kind==='revision'?'历史修订 · '+(record.sender||message.name||''):record.sender||message.name||''));
  head.append(title,element('span',record.timestamp?date(record.timestamp,record.payload?.extra?.time_unknown):'','meta'));
  root.append(head);
  const body=element('div',undefined,'card-body');
  if(summary)highlight(body,record.payload.text);
  else if(typeof message.content==='string')highlight(body,message.content);
  else if(Array.isArray(message.content))for(const part of message.content){
    if(part.type==='text'){const p=element('div');highlight(p,part.text||'');body.append(p);}
    else if(part.type==='image_url'){
      const url=part.image_url?.url||'';
      const btn=element('button','查看此记录图片','media-button');
      btn.onclick=()=>{const img=element('img');img.className='media-img';img.alt='历史图片';
        if(/^data:image\//.test(url)){img.src=url;body.append(img);btn.remove();}
        else{btn.textContent='原引用保留在详细记录中；媒体存档请展开原轮次查看';btn.disabled=true;}
      };body.append(btn);
    }else detail(body,part.type==='think'?'思考内容':part.type||'媒体',part);
  }
  root.append(body);
  if(message.tool_calls?.length)detail(root,'工具调用 · '+message.tool_calls.length+' 项',message.tool_calls);
  if(message.tool_call_id)detail(root,'工具结果 · '+message.tool_call_id,message.content);
  if(record.payload?.extra && Object.keys(record.payload.extra).length)detail(root,'原始事件 / 媒体引用',record.payload.extra);
  if(record.media?.length){for(const m of record.media){const btn=element('button','原媒体 · '+(m.status==='ready'?'查看':m.status),'media-button');btn.disabled=m.status!=='ready';btn.onclick=async()=>{try{const r=await api('media',{id:m.id});if(r.mime.startsWith('image/')){const img=element('img');img.className='media-img';img.src=r.data_url;img.alt='归档原图';body.append(img);}else{const link=element('a','下载原媒体');link.href=r.data_url;link.download=m.id;body.append(link);}}catch(e){notice(e.message,true);}};body.append(btn);}}
  const foot=element('div',undefined,'card-foot');
  foot.append(element('span',[record.id,record.tokens?size(record.tokens)+' tokens（估算）':'',record.round_id?'整轮 '+record.round_id.slice(0,10):''].filter(Boolean).join(' · ')));
  if(record.id){const btn=element('button',summary||record.kind==='revision'?'逐层查看来源':'查看完整轮次');btn.onclick=()=>inspect(record.id);foot.append(btn);}
  root.append(foot);return root;
}
function empty(title,text){const box=element('div',undefined,'empty');box.append(element('div','◇','icon'),element('h2',title),element('p',text));return box;}
function setMore(cursor,callback){state.cursor=cursor||'';$('more').hidden=!cursor;$('more').onclick=callback;}
async function sessions(){
  try{const data=await api('sessions');state.sessions=data.sessions;$('budget').textContent='分层基准 B = '+size(data.base_tokens)+' tokens（估算）';
    $('sessions').replaceChildren();
    for(const s of data.sessions){const b=element('button',undefined,'session'+(s.id===state.scope?' active':''));b.append(element('strong',s.title),element('small',s.cid.slice(0,12)+' · '+size(s.records)+' 条原文 · '+(s.current?'当前使用':'历史存档')));b.disabled=state.resetting;b.onclick=()=>selectSession(s.id);$('sessions').append(b);}
    updateResetButton();
    if(!data.sessions.length){$('stream').replaceChildren(empty('历史从下一条消息开始','插件启用后，收到的消息和完整回复会出现在这里。'));notice('尚无已归档会话。');}
    else if(!state.scope)await selectSession(data.sessions[0].id);
  }catch(e){notice(e.message,true);$('stream').replaceChildren(empty('暂时无法读取','请确认插件已加载，并从 AstrBot 的插件页面打开。'));}
}
function updateResetButton(){const current=state.sessions.find(s=>s.id===state.scope);$('reset-context').disabled=state.resetting||!current?.current;$('reset-context').title=current?.current?'新建空白聊天上下文，旧记录保留；长期记忆和记事本继续使用。':'请选择正在使用的会话。';}
async function selectSession(id,force=false){
  if(state.resetting&&!force)return;
  state.scope=id;state.viewVersion=null;updateResetButton();state.searching=false;state.snapshot=null;state.saved=null;state.query='';$('query').value='';$('sidebar').classList.remove('open');closeInspector();
  for(const [index,b] of [...$('sessions').children].entries())b.classList.toggle('active',state.sessions[index].id===id);
  const session=state.sessions.find(s=>s.id===id);state.timezone=session.timezone||'Asia/Shanghai';$('session-name').textContent=session.title+' · '+session.cid+' · '+state.timezone;
  $('start').title=$('end').title='按机器人时区 '+state.timezone+' 筛选';
  $('raw-count').textContent=size(session.records);await load();
}
async function load(append=false){
  if(!state.scope)return;
  const generation=++state.generation;notice('正在读取…');
  try{
    if(state.mode==='request'&&!state.searching){await loadRuns();return;}
    const endpoint=state.searching?'search':state.mode;
    const params={cursor:append?state.cursor:''};
    if(append&&state.mode==='context')params.version=String(state.viewVersion);
    if(state.searching){Object.assign(params,{q:state.query,target:$('target').value,sender:$('sender').value,in_context:state.mode==='context'?'1':'0',level:$('level').value});
      if(state.mode==='context')params.version=String(state.viewVersion);
      if($('start').value)params.start_time=filterTime($('start').value);if($('end').value)params.end_time=filterTime($('end').value);}
    const data=await api(endpoint,params);if(generation!==state.generation)return;
    if(!append)$('stream').replaceChildren();
    if(data.tokens){state.viewVersion=data.version;$('version').textContent='v'+data.version;const total=Object.values(data.tokens).reduce((a,b)=>a+b,0);$('token-count').textContent=size(total);$('layers').textContent=Math.max(0,...Object.keys(data.tokens).map(Number));$('distribution').replaceChildren();for(const [level,n] of Object.entries(data.tokens)){const i=element('i');i.style.flex=n||1;i.style.opacity=.4+Math.min(.6,Number(level)*.16);i.title='L'+level+' · '+size(n);$('distribution').append(i);}
      $('jobs').replaceChildren();for(const j of data.jobs||[]){const row=element('div',undefined,'job');row.append(element('span','L'+j.level+' → L'+(j.level+1)),element('span',labels[j.status]||j.status),element('span',j.error||''));$('jobs').append(row);}}
    for(const r of data.records||[])$('stream').append(card(r));
    if(!append&&!data.records?.length)$('stream').append(empty(state.searching?'没有匹配的内容':'这里还没有记录',state.searching?'可以调整关键词、发送者、时间或搜索范围。':'新的聊天会自动归档，点击刷新查看。'));
    setMore(data.next_cursor,()=>load(true));
    notice(state.searching?(state.mode==='context'?'仅搜索此上下文版本的可见文字；摘要下的原文可切换「完整历史」搜索。':'搜索本会话所有原文与摘要。'):(state.mode==='context'?'这是下一轮使用的已发布历史；当次人格、记忆和工具以「实际请求」为准。':'完整原始记录，按新到旧显示；工具内容仅在页面折叠。'));
  }catch(e){notice(e.message,true);}
}
async function loadRuns(append=false){
  const data=await api('runs',{cursor:append?state.runOffset:0});
  if(!append){$('run').replaceChildren();state.runOffset=0;}
  for(const run of data.runs){const option=element('option',date(run.started)+' · '+(labels[run.status]||run.status)+' · '+run.id.slice(0,10));option.value=run.id;$('run').append(option);}
  state.runOffset+=data.runs.length;
  if(!append){if(data.runs.length)await loadSteps();else{$('stream').replaceChildren(empty('还没有实际请求','正式回复开始后，这里会保存每一步发送给模型的逻辑输入。'));notice('旧历史无法补造当时的请求快照。');setMore(null);}}
  if(data.runs.length===40)setMore('runs',()=>loadRuns(true));
}
async function loadSteps(){
  const run=$('run').value;if(!run)return;
  const data=await api('requests',{run});$('step').replaceChildren();
  for(const req of data.requests){const o=element('option','第 '+req.step+' 次调用 · '+size(req.tokens)+' tokens（估算）');o.value=req.id;$('step').append(o);}
  if(data.requests.length){$('step').value=data.requests.at(-1).id;await loadRequest();}
  else{$('stream').replaceChildren(empty('这轮没有发出模型请求','可在完整历史中查看已保存的输入及中断状态。'));notice('没有可展示的请求快照。');}
}
function renderSnapshot(){
  const data=state.snapshot;if(!data)return;
  const payload=data.payload;let messages=payload.contexts||[];
  if(state.searching)messages=messages.filter(m=>JSON.stringify(m).toLocaleLowerCase().includes(state.query.toLocaleLowerCase()));
  $('stream').replaceChildren(element('div','轮次 '+data.run+' · 调用 '+data.step+' · 历史版本 v'+data.run_info.version,'call-label'));
  for(const m of messages)$('stream').append(card({message:m}));
  const meta=element('article',undefined,'card');detail(meta,'本次请求的工具定义、模型及额外字段',Object.fromEntries(Object.entries(payload).filter(([k])=>k!=='contexts')));$('stream').append(meta);
  notice('本次调用实际交给提供商适配器的逻辑输入 · '+payload.provider+' / '+payload.model+' · '+size(data.tokens)+' tokens（估算）。');
  setMore(null);
}
async function loadRequest(){try{state.searching=false;state.snapshot=await api('request',{id:$('step').value});renderSnapshot();}catch(e){notice(e.message,true);}}
async function inspect(id,reset=true){
  if(reset)state.trail=[];
  if(state.trail.at(-1)!==id)state.trail.push(id);
  $('inspector').hidden=false;$('layout').classList.add('inspecting');$('source').textContent='正在读取来源…';
  try{const data=await api('describe',{id});$('breadcrumbs').replaceChildren();state.trail.forEach((key,index)=>{const b=element('button',index===0?'来源':key.slice(0,10));b.onclick=()=>{state.trail=state.trail.slice(0,index);inspect(key,false);};$('breadcrumbs').append(b);});
    $('source').replaceChildren(card(data.node,true));
    if(data.node.level>0||data.node.kind==='revision'){$('source').append(element('div','直接来源 · 按原时间顺序','source-label'));for(const r of data.records){const c=card(r,true);c.querySelector('.card-foot button').onclick=()=>inspect(r.id,false);$('source').append(c);}if(data.next_cursor){const more=element('button','更多直接来源');more.onclick=async()=>{const next=await api('describe',{id,cursor:data.next_cursor});data.next_cursor=next.next_cursor;for(const r of next.records){const c=card(r,true);c.querySelector('.card-foot button').onclick=()=>inspect(r.id,false);more.before(c);}more.hidden=!next.next_cursor;};$('source').append(more);}}
    const original=element('button','查看原始完整轮次');original.onclick=()=>originals(id);$('source').append(original);
  }catch(e){$('source').textContent=e.message;}
}
async function originals(id,cursor='',append=false){
  try{const data=await api('expand',{id,cursor});if(!append)$('source').replaceChildren();
    for(const unit of data.units||[]){const header=element('div',unit.id+' · '+(labels[unit.round?.status]||unit.round?.status||'未回复原始消息'),'source-label');$('source').append(header);for(const n of unit.records)$('source').append(card(n,true));}
    if(data.next_cursor){const next=element('button','下一组完整轮次');next.onclick=()=>{next.remove();originals(id,data.next_cursor,true);};$('source').append(next);}
  }catch(e){$('source').textContent=e.message;}
}
function closeInspector(){$('inspector').hidden=true;$('layout').classList.remove('inspecting');state.trail=[];}
for(const b of document.querySelectorAll('[data-mode]'))b.onclick=async()=>{state.mode=b.dataset.mode;state.searching=false;state.snapshot=null;state.saved=null;$('query').value='';$('requestbar').hidden=state.mode!=='request';$('jobs-panel').hidden=state.mode!=='context';$('back').hidden=true;for(const other of document.querySelectorAll('[data-mode]'))other.classList.toggle('active',other===b);closeInspector();await load();};
$('search-form').onsubmit=async event=>{event.preventDefault();const query=$('query').value.trim();if(!query){state.searching=false;await load();return;}state.query=query;state.searching=true;if(state.mode==='request'&&state.snapshot)renderSnapshot();else await load();};
$('filter-toggle').onclick=()=>{$('filters').hidden=!$('filters').hidden;};
$('run').onchange=()=>loadSteps().catch(e=>notice(e.message,true));$('step').onchange=loadRequest;
$('reset-context').onclick=async()=>{
 const session=state.sessions.find(s=>s.id===state.scope);if(state.resetting||!session?.current)return;
 if(!confirm('重置这个会话的 Context？将新建空白聊天上下文，停止本会话正在处理的回复。旧聊天与摘要保留，长期记忆和记事本继续使用。'))return;
 state.resetting=true;state.generation++;updateResetButton();$('refresh').disabled=true;for(const b of $('sessions').children)b.disabled=true;notice('正在重置聊天上下文…');
 try{const bridge=window.AstrBotPluginPage;const response=await bridge.apiPost('reset',{scope:session.id});if(response?.status==='error')throw Error(response.message||'重置失败');const result=response?.status==='ok'||response?.status==='success'?response.data:response;
  state.scope='';state.mode='context';$('requestbar').hidden=true;$('jobs-panel').hidden=false;$('back').hidden=true;for(const b of document.querySelectorAll('[data-mode]'))b.classList.toggle('active',b.dataset.mode==='context');await sessions();await selectSession(result.scope,true);notice('Context 已重置。新对话从空白聊天上下文开始，长期记忆和记事本继续使用；旧会话保留在左侧。');
 }catch(e){state.resetting=false;await sessions();notice(e.message,true);}finally{state.resetting=false;$('refresh').disabled=false;for(const b of $('sessions').children)b.disabled=false;updateResetButton();}
};
$('refresh').onclick=async()=>{await sessions();await load();};$('close-inspector').onclick=closeInspector;
$('nav').onclick=()=>$('sidebar').classList.add('open');$('close-nav').onclick=()=>$('sidebar').classList.remove('open');
function theme(value){document.documentElement.dataset.theme=value==='dark'?'dark':'light';}
theme(new URLSearchParams(location.search).get('theme')||'light');
$('theme').onclick=()=>theme(document.documentElement.dataset.theme==='dark'?'light':'dark');
window.addEventListener('message',event=>{if(event.source!==window.parent)return;const d=event.data;if(d?.channel==='astrbot-plugin-page'&&d?.type==='theme-change')theme(d.theme||d.payload?.theme);});
window.addEventListener('keydown',e=>{if(e.key==='Escape'){closeInspector();$('sidebar').classList.remove('open');}});
sessions();
