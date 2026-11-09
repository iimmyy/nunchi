// nunchi: the pages the room's computer serves. Phones: the English as it is translated, the Korean under it if wanted,
// and the voice in earbuds. Screen (the computer itself, also the projector): start and stop, the sound level, the
// QR codes to join, and big captions.
#pragma once

static const char * NUNCHI_PHONE = R"NUNCHI(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#111318"><title>Nunchi</title>
<style>
:root{--bg:#f6f5f2;--fg:#16181d;--dim:#6b6f78;--line:#e2e0da;--acc:#1f6feb;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--fg:#eceae4;--dim:#9198a3;--line:#262a33;--acc:#5b9dff;--card:#191c23}}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
header{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:10px;padding:12px 16px;padding-top:max(12px,env(safe-area-inset-top));background:var(--bg);border-bottom:1px solid var(--line)}
header b{font-size:18px;letter-spacing:.2px}#dot{width:9px;height:9px;border-radius:50%;background:#c0392b;flex:none}
#dot.on{background:#2ea043}#state{color:var(--dim);font-size:14px;flex:1}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;padding:7px 12px}
button.main{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
#lines{padding:8px 16px 140px}
.s{padding:12px 0;border-bottom:1px solid var(--line)}
.s .en{font-size:var(--fs,1.25em);line-height:1.4}.s .ko{color:var(--dim);font-size:.9em;margin-top:4px;display:none}
body.ko .s .ko{display:block}.s.old .en{color:var(--dim)}.s.writing .en{opacity:.75}
#empty{color:var(--dim);padding:40px 0;text-align:center}
footer{position:fixed;left:0;right:0;bottom:0;display:flex;gap:8px;justify-content:center;padding:12px;padding-bottom:max(12px,env(safe-area-inset-bottom));background:linear-gradient(transparent,var(--bg) 30%)}
#jump{position:fixed;right:16px;bottom:80px;display:none}
#start{position:fixed;inset:0;z-index:5;display:flex;align-items:center;justify-content:center;padding:24px;background:var(--bg)}
#start .box{max-width:340px;width:100%;text-align:center}#start .box>b{display:block;font-size:26px;margin-bottom:2px}
#start p{color:var(--dim);margin:0 0 22px}#start button{display:block;width:100%;padding:14px;margin:10px 0;font-size:18px}
#start .note{font-size:14px;margin:18px 0 0}
</style></head><body>
<div id="start"><div class="box"><b>Nunchi</b><p>The sermon in English, live.</p>
<button class="main" id="go-listen">🎧 Listen and read</button><button id="go-read">Read only</button>
<p class="note">Your screen stays on while Nunchi is open.</p></div></div>
<header><span id="dot"></span><b>Nunchi</b><span id="state">connecting…</span></header>
<div id="lines"><div id="empty">The English appears here as he speaks.</div></div>
<button id="jump" onclick="follow=true;down()">↓ Live</button>
<footer><button class="main" id="listen">🎧 Listen</button><button id="kob">한국어</button><button id="sm">A−</button><button id="lg">A+</button></footer>
<script>
const lines=document.getElementById('lines'),empty=document.getElementById('empty');
let follow=true,fs=parseFloat(localStorage.fs||'1.25'),ctx=null,next=0,es=null,audio=false,lock=null;
document.documentElement.style.setProperty('--fs',fs+'em');
if(localStorage.ko==='1')document.body.classList.add('ko');
function down(){window.scrollTo(0,document.body.scrollHeight)}
window.addEventListener('scroll',()=>{follow=window.innerHeight+window.scrollY>=document.body.scrollHeight-80;document.getElementById('jump').style.display=follow?'none':'block'});
function line(id){let d=document.getElementById('s'+id);if(!d){empty.remove();d=document.createElement('div');d.className='s';d.id='s'+id;
 d.innerHTML='<div class="en"></div><div class="ko"></div>';
 const after=[...lines.children].find(c=>c.id&&+c.id.slice(1)>id);lines.insertBefore(d,after||null);
 while(lines.children.length>60)lines.firstChild.remove();}return d}
function show(id,en,ko,done){const d=line(id);d.querySelector('.en').textContent=en;if(ko!=null)d.querySelector('.ko').textContent=ko;
 d.classList.toggle('writing',!done);[...lines.children].forEach(c=>{if(c!==d&&c.id&&+c.id.slice(1)<id-1)c.classList.add('old')});if(follow)down()}
function play(b64){if(!ctx||ctx.state!=='running')return;const s=atob(b64),n=s.length>>1,buf=ctx.createBuffer(1,n,24000),ch=buf.getChannelData(0);
 for(let i=0;i<n;i++){let v=s.charCodeAt(2*i)|s.charCodeAt(2*i+1)<<8;if(v>32767)v-=65536;ch[i]=v/32768}
 const t=Math.max(ctx.currentTime+0.08,next);if(t-ctx.currentTime>8)return;const src=ctx.createBufferSource();src.buffer=buf;src.connect(ctx.destination);src.start(t);next=t+buf.duration}
function connect(){if(es)es.close();es=new EventSource('/events'+(audio?'?audio=1':''));
 es.onopen=()=>{document.getElementById('dot').className='on';document.getElementById('state').textContent=audio?'live · listening':'live'};
 es.onerror=()=>{document.getElementById('dot').className='';document.getElementById('state').textContent='reconnecting…'};
 es.addEventListener('en',e=>{const m=JSON.parse(e.data);show(m.id,m.en,m.ko,true)});
 es.addEventListener('w',e=>{const m=JSON.parse(e.data);show(m.id,m.en,null,false)});
 es.addEventListener('sp',e=>play(JSON.parse(e.data).d))}
// The screen stays on while the page is open: the browser's wake lock where it has one (https, localhost), else a tiny
// silent video kept playing, since phones get this page over plain http on the room's Wi-Fi, where there is no wake
// lock. iPhones stay awake only for a video that is in the page, playing, and moved back before its end (a looping
// one does not count), and start one only from a tap: the start screen's.
const AWAKE='data:video/mp4;base64,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';
let awakeOn=false,awakeV=null;
async function awake(){awakeOn=true;
 if('wakeLock' in navigator){try{lock=await navigator.wakeLock.request('screen');return}catch(e){}}
 if(!awakeV){const v=awakeV=document.createElement('video');v.setAttribute('playsinline','');v.setAttribute('title','Nunchi');
  v.style.cssText='position:fixed;left:-10px;top:-10px;width:1px;height:1px;opacity:.01;pointer-events:none';v.src=AWAKE;
  v.addEventListener('timeupdate',()=>{if(v.currentTime>1)v.currentTime=Math.random()*.5});document.body.appendChild(v)}
 awakeV.play().catch(()=>{})}
document.addEventListener('visibilitychange',()=>{if(document.visibilityState!=='visible')return;if(awakeOn)awake();if(ctx)ctx.resume()});
const lb=document.getElementById('listen');
async function listen(){
 if(audio){audio=false;lb.textContent='🎧 Listen';lb.className='main';if(ctx)ctx.suspend();connect();return}
 try{if(navigator.audioSession)navigator.audioSession.type='playback'}catch(x){}
 ctx=ctx||new (window.AudioContext||window.webkitAudioContext)();await ctx.resume();
 const b=ctx.createBuffer(1,1,24000),s=ctx.createBufferSource();s.buffer=b;s.connect(ctx.destination);s.start();
 next=0;audio=true;lb.textContent='■ Stop voice';lb.className='';connect()}
lb.onclick=()=>{awake();listen()};
document.getElementById('go-listen').onclick=()=>{awake();document.getElementById('start').remove();listen()};
document.getElementById('go-read').onclick=()=>{awake();document.getElementById('start').remove()};
document.getElementById('kob').onclick=()=>{document.body.classList.toggle('ko');localStorage.ko=document.body.classList.contains('ko')?'1':'0'};
document.getElementById('sm').onclick=()=>{fs=Math.max(.9,fs-.15);localStorage.fs=fs;document.documentElement.style.setProperty('--fs',fs+'em')};
document.getElementById('lg').onclick=()=>{fs=Math.min(2.6,fs+.15);localStorage.fs=fs;document.documentElement.style.setProperty('--fs',fs+'em')};
connect();
</script></body></html>)NUNCHI";

static const char * NUNCHI_SCREEN = R"NUNCHI(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nunchi screen</title>
<style>
:root{--bg:#0e1015;--fg:#f1efe9;--dim:#8e95a1;--line:#252933;--acc:#3b82f6;--card:#161921;--ok:#2ea043;--bad:#c0392b}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif}
#bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--card)}
#bar b{font-size:18px;margin-right:6px}button,select,input,textarea{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:7px 11px}
button.go{background:var(--ok);border-color:var(--ok);color:#fff;font-weight:600}button.stop{background:var(--bad);border-color:var(--bad);color:#fff;font-weight:600}
#meter{width:140px;height:10px;border-radius:5px;background:var(--line);overflow:hidden}#meter i{display:block;height:100%;width:0;background:var(--ok);transition:width .1s}
.dim{color:var(--dim);font-size:14px}
main{display:grid;grid-template-columns:1fr 300px;gap:0;height:calc(100% - 60px)}
#cap{overflow:hidden;padding:24px 40px;display:flex;flex-direction:column;justify-content:flex-end}
#cap .s{font-size:clamp(28px,3.4vw,58px);line-height:1.25;margin:.25em 0;font-weight:500}#cap .s.old{color:var(--dim)}#cap .s.w{opacity:.8}
#cap .ko{font-size:clamp(16px,1.5vw,24px);color:var(--dim);margin-bottom:.6em}
aside{border-left:1px solid var(--line);padding:18px;overflow:auto;background:var(--card)}
aside h3{margin:6px 0 8px;font-size:15px}aside .qr{background:#fff;border-radius:10px;padding:6px;width:100%;max-width:240px}
aside .qr img{width:100%;display:block}aside label{display:block;font-size:13px;color:var(--dim);margin-top:8px}
aside input,aside textarea{width:100%}textarea{min-height:90px}
body.proj #bar,body.proj aside .ctl{display:none}body.proj main{height:100%}
</style></head><body>
<div id="bar"><b>Nunchi</b>
<button id="go" class="go">▶ Start</button>
<select id="dev"><option value="">Default input</option></select>
<div id="meter"><i></i></div><span class="dim" id="lvl"></span>
<label class="dim"><input type="checkbox" id="voice" checked> voice</label>
<span class="dim" id="st"></span>
<span style="flex:1"></span>
<button id="proj">Projector view</button></div>
<main><div id="cap"><div class="s dim" id="hint">Press Start. The English shows here and on every phone.</div></div>
<aside>
<h3>1 · Join the Wi-Fi</h3><div class="qr" id="wq" style="display:none"><img id="wqi" alt=""></div>
<div id="wn" class="dim"></div>
<div class="ctl"><label>Wi-Fi name<input id="ssid" placeholder="the network phones join"></label>
<label>Wi-Fi password<input id="pw" placeholder="(none if open)"></label></div>
<h3 style="margin-top:18px">2 · Open Nunchi</h3><div class="qr"><img id="uq" alt=""></div><div id="url" class="dim"></div>
<div class="ctl"><h3 style="margin-top:18px">This week's terms</h3>
<div class="dim">One per line: 사사기 = Judges</div><textarea id="terms"></textarea><button id="save" style="margin-top:6px">Save terms</button>
<div class="dim" style="margin-top:14px" id="who"></div></div>
</aside></main>
<script>
const $=id=>document.getElementById(id);let on=false,ctx=null,stream=null,node=null,q=[],sending=false,addr='';
async function api(p,o){const r=await fetch(p,o);return r.headers.get('content-type')?.includes('json')?r.json():r.text()}
async function devices(){try{const d=await navigator.mediaDevices.enumerateDevices();const s=$('dev'),v=s.value;s.innerHTML='<option value="">Default input</option>';
 d.filter(x=>x.kind==='audioinput').forEach(x=>{const o=document.createElement('option');o.value=x.deviceId;o.textContent=x.label||'Input '+(s.length);s.appendChild(o)});s.value=localStorage.dev||v}catch(e){}}
async function flush(){if(sending||!q.length)return;sending=true;try{while(q.length){const n=q.reduce((a,b)=>a+b.length,0),i16=new Int16Array(n);let k=0;
 for(const b of q.splice(0))for(let i=0;i<b.length;i++){const v=Math.max(-1,Math.min(1,b[i]));i16[k++]=v<0?v*32768:v*32767}
 await fetch('/mic?rate='+ctx.sampleRate,{method:'POST',body:i16.buffer})}}catch(e){}sending=false}
async function start(){
 stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:$('dev').value?{exact:$('dev').value}:undefined,echoCancellation:false,noiseSuppression:false,autoGainControl:false,channelCount:1}});
 localStorage.dev=$('dev').value;devices();
 try{ctx=new AudioContext({sampleRate:16000});ctx.createMediaStreamSource(stream)}catch(e){ctx=new AudioContext()}
 await ctx.audioWorklet.addModule('/worklet.js');const src=ctx.createMediaStreamSource(stream);node=new AudioWorkletNode(ctx,'grab');
 let peak=0;node.port.onmessage=e=>{q.push(e.data);for(const v of e.data)peak=Math.max(peak,Math.abs(v))};
 const mute=ctx.createGain();mute.gain.value=0;src.connect(node);node.connect(mute);mute.connect(ctx.destination);
 setInterval(()=>{flush();const db=20*Math.log10(peak+1e-9);$('meter').firstChild.style.width=Math.max(0,Math.min(100,(db+60)/60*100))+'%';$('lvl').textContent=peak>0?Math.round(db)+' dB':'';peak=0},100);
 await api('/api/set?listening=1',{method:'POST'});on=true;$('go').textContent='■ Stop';$('go').className='stop';$('hint')?.remove()}
async function stop(){on=false;await api('/api/set?listening=0',{method:'POST'});if(stream)stream.getTracks().forEach(t=>t.stop());if(ctx)ctx.close();ctx=null;$('go').textContent='▶ Start';$('go').className='go'}
// the computer's screen stays on while it translates (the page is on localhost, where browsers give a wake lock)
let slock=null;async function stayOn(){try{if('wakeLock' in navigator&&document.visibilityState==='visible')slock=await navigator.wakeLock.request('screen')}catch(e){}}
document.addEventListener('visibilitychange',()=>{if(on)stayOn()});
$('go').onclick=()=>on?stop():(stayOn(),start().catch(e=>alert('The microphone could not start: '+e.message)));
$('voice').onchange=()=>api('/api/set?voice='+($('voice').checked?1:0),{method:'POST'});
$('proj').onclick=()=>{document.body.classList.toggle('proj');if(document.body.classList.contains('proj'))document.documentElement.requestFullscreen?.()};
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.body.classList.remove('proj')});
function wifi(){const s=$('ssid').value.trim(),p=$('pw').value;localStorage.ssid=s;localStorage.pw=p;
 if(!s){$('wq').style.display='none';$('wn').textContent='Phones join the same Wi-Fi as this computer.';return}
 const esc=x=>x.replace(/([\\;,:"])/g,'\\$1');$('wqi').src='/qr.svg?text='+encodeURIComponent('WIFI:T:'+(p?'WPA':'nopass')+';S:'+esc(s)+';'+(p?'P:'+esc(p)+';':'')+';');
 $('wq').style.display='block';$('wn').textContent=s+(p?' · password '+p:'')}
$('ssid').value=localStorage.ssid||'';$('pw').value=localStorage.pw||'';$('ssid').oninput=wifi;$('pw').oninput=wifi;wifi();
$('save').onclick=async()=>{await api('/api/terms',{method:'POST',body:$('terms').value});$('save').textContent='Saved ✓';setTimeout(()=>$('save').textContent='Save terms',1500)};
async function poll(){try{const s=await api('/api/state');if(!$('terms').value&&s.terms)$('terms').value=s.terms;
 const u='http://'+(s.addresses[0]||location.hostname)+(s.port==80?'':':'+s.port);if(u!==addr){addr=u;$('uq').src='/qr.svg?text='+encodeURIComponent(u);$('url').textContent=u}
 $('who').textContent=s.listeners+' phone'+(s.listeners==1?'':'s')+' connected · '+s.voice_listeners+' listening · '+s.sentences+' sentences';
 $('st').textContent=s.listening?'translating':'paused';$('voice').checked=s.voice}catch(e){$('st').textContent='nunchi is not running'}}
setInterval(poll,2000);poll();devices();
const cap=$('cap'),es=new EventSource('/events');
function show(id,en,ko,done){let d=document.getElementById('c'+id);if(!d){d=document.createElement('div');d.id='c'+id;d.innerHTML='<div class="s"></div><div class="ko"></div>';cap.appendChild(d);while(cap.children.length>8)cap.firstChild.remove()}
 d.querySelector('.s').textContent=en;if(ko!=null)d.querySelector('.ko').textContent=ko;d.querySelector('.s').classList.toggle('w',!done);
 [...cap.children].forEach(c=>{if(c!==d)c.querySelector('.s')?.classList.add('old')})}
es.addEventListener('en',e=>{const m=JSON.parse(e.data);show(m.id,m.en,m.ko,true)});
es.addEventListener('w',e=>{const m=JSON.parse(e.data);show(m.id,m.en,null,false)});
</script></body></html>)NUNCHI";

static const char * NUNCHI_WORKLET = R"NUNCHI(class Grab extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}
registerProcessor('grab',Grab);)NUNCHI";
