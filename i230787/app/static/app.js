const $=id=>document.getElementById(id);
let session=crypto.randomUUID(),count=0;
function message(role,text){const p=document.createElement('article');p.className=role;p.textContent=(role==='user'?'You':'Agent')+': '+text;$('messages').append(p);p.scrollIntoView({block:'nearest'});}
function describe(e){
  switch(e.event){
    case 'run_start':return `start · model ${e.model} · max ${e.max_steps} steps`+(e.fault!=='none'?` · fault ${e.fault}`:'')+(e.resumed_clarification?' · continuing pending question':'');
    case 'boundary_detected':return `autonomy boundary: ${e.flag} (${e.source})`;
    case 'model_call':return `model call ${e.model||''} · ${e.latency_ms} ms · tokens ${e.input_tokens??'?'} in / ${e.output_tokens??'?'} out`;
    case 'model_error':return `model error: ${e.kind} (attempt ${e.attempt})`;
    case 'fault_injected':return `fault injected: ${e.fault}`+(e.tool?` on ${e.tool}`:'');
    case 'model_decision':return `decision: ${e.status}`+(e.action?` → ${e.action} ${JSON.stringify(e.arguments)}`:'')+(e.reason?` · “${e.reason}”`:'');
    case 'validation':return e.result==='accepted'?'validated ✓':`rejected (${e.kind}): ${e.detail}`;
    case 'repair_requested':return `repair requested (${e.attempt}/${e.limit})`;
    case 'tool_call':return `execute ${e.tool}`;
    case 'tool_result':return `${e.tool} attempt ${e.attempt}: ${e.outcome}`+(e.detail?` – ${e.detail}`:'');
    case 'observation':return `observation: ${e.summary}`;
    case 'output_guard':return 'output guard redacted privileged text';
    case 'agent_stop':return `stop: ${e.status} (${e.reason}) after ${e.steps} steps`;
    case 'warnings':return `warnings: ${e.items.join('; ')}`;
    default:return e.event;
  }
}
function render(data){
  $('status').textContent=`${data.status} · ${data.stop_reason} · ${data.steps} steps`;
  $('status').className='status-'+data.status;
  const m=data.metrics;$('metrics').textContent=`${Math.round(m.latency_ms)} ms · ${m.model_calls} model calls · tokens ${m.input_tokens??'n/a'} in / ${m.output_tokens??'n/a'} out · est. cost ${m.estimated_cost_usd==null?'n/a':'$'+m.estimated_cost_usd.toFixed(6)}`;
  $('timeline').replaceChildren(...data.events.map(e=>{const li=document.createElement('li');li.textContent=`[step ${e.step}] `+describe(e);li.className='ev-'+e.event;return li;}));
  $('trace').textContent=data.tool_calls.length?data.tool_calls.map(t=>`step ${t.step} · ${t.tool} · attempt ${t.attempt} · ${t.outcome} · ${Math.round(t.latency_ms)} ms`).join('\n'):'No tool calls in this run.';
  $('observations').textContent=JSON.stringify({errors:data.errors,events:data.events},null,2);
}
async function init(){try{const r=await fetch('/models');if(!r.ok)throw Error('Models unavailable');const data=await r.json();if(!data.models.length)throw Error('No model configured on the server');for(const m of data.models){const o=document.createElement('option');o.value=m;o.textContent=m+(m===data.default?' (default)':'');$('model').append(o);}}catch(e){$('status').textContent=e.message;$('send').disabled=true;}}
$('chat').onsubmit=async e=>{e.preventDefault();const task=$('task').value.trim();if(!task)return;const note=$('external').value.trim();message('user',task);$('send').disabled=true;$('reset').disabled=true;$('status').textContent='Running…';$('timeline').replaceChildren();$('trace').textContent='Waiting for result…';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session,task,model:$('model').value,external_context:note?[{source:'browser-note',content:note,trust:'untrusted'}]:[]})});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));message('agent',data.final_response);count+=2;$('memory').textContent=Math.min(count,12)+' messages retained in this chat (server keeps at most 12)';render(data);$('task').value='';}catch(e){$('status').textContent='Error: '+e.message;}finally{$('send').disabled=false;$('reset').disabled=false;}};
$('reset').onclick=async()=>{try{const r=await fetch('/chat/'+session,{method:'DELETE'});if(!r.ok)throw Error('Reset failed');session=crypto.randomUUID();count=0;$('messages').replaceChildren();$('memory').textContent='0 messages in this chat';$('external').value='';$('task').value='';$('status').textContent='Idle';$('status').className='';$('metrics').textContent='';$('timeline').replaceChildren();$('trace').textContent='No tool calls yet.';$('observations').textContent='No observations yet.';}catch(e){$('status').textContent=e.message;}};
init();
