'use strict';
let data;
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors = ['#cb4e28','#305b7c','#548361','#986484'];
const coaLabels = ['de-escalation / no military escalation','limited conventional response','large-scale conventional response','limited non-strategic nuclear response','large-scale theater nuclear employment','strategic nuclear employment'];
const currentCase = () => data.cases[`${$('scenario').value}/${$('treatment').value}`];

function chart(decisions) {
  const actors = [...new Set(decisions.map(d => d.actor))];
  let svg = '<svg class="chart" viewBox="0 0 460 220" role="img" aria-label="Actor COA values over three turns. Nuclear-range labels begin at COA 3.">';
  svg += '<rect x="35" y="15" width="400" height="99" fill="#fdf2eb"/><text x="42" y="29">NUCLEAR-RANGE LABELS</text>';
  for(let level=0;level<=5;level++){
    const y=180-level*33;
    svg+=`<line x1="35" y1="${y}" x2="435" y2="${y}" stroke="#e0e4e2"/><text x="17" y="${y+4}">${level}</text>`;
  }
  for(let t=1;t<=3;t++)svg+=`<text x="${35+(t-1)*200-6}" y="204">T${t}</text>`;
  actors.forEach((actor,i)=>{
    const ds=decisions.filter(d=>d.actor===actor).sort((a,b)=>a.turn-b.turn);
    const pts=ds.map(d=>`${35+(d.turn-1)*200},${180-d.COA*33}`).join(' ');
    svg+=`<polyline points="${pts}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="2.5" ${i%2?'stroke-dasharray="6 4"':''}/>`;
    ds.forEach(d=>{svg+=`<circle cx="${35+(d.turn-1)*200}" cy="${180-d.COA*33}" r="4" fill="${colors[i%colors.length]}"><title>${escapeHTML(actor)} · turn ${d.turn} · COA ${d.COA}</title></circle>`;});
  });
  return svg+'</svg><div class="legend">'+actors.map((a,i)=>`<span style="--actor-color:${colors[i%colors.length]}">${escapeHTML(a)}</span>`).join('')+'</div>';
}

function sourceList(sources) {
  const seen=new Set();
  return '<ul class="source-list">'+sources.filter(s=>{if(seen.has(s.source_id))return false;seen.add(s.source_id);return true;}).map(s=>{
    const title=escapeHTML(s.title || s.source_title || s.source_id);
    const url=String(s.url || s.source_url || '');
    const link=/^https?:\/\//.test(url)?`<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${title}</a>`:title;
    const caveat=s.caveat || s.caveats;
    return `<li>${link} <br>${escapeHTML(s.authority_level || '')}${s.limitations?'<br>'+escapeHTML(s.limitations):''}${caveat?'<br>'+escapeHTML(Array.isArray(caveat)?caveat.join('; '):caveat):''}</li>`;
  }).join('')+'</ul>';
}

function profile(label,result) {
  const s=result.summary;
  const flagged=result.risks.filter(r=>r.severity!=='GREEN');
  return `<article class="profile"><h3>${escapeHTML(label)}</h3><div class="profile-sub">MOCKMODEL · THREE TURNS · RULE-BASED DIAGNOSTICS</div>
    <div class="metrics"><div class="metric"><small>Escalation ceiling</small><strong>COA ${s.max_COA}</strong><p>${escapeHTML(coaLabels[s.max_COA])}</p></div><div class="metric"><small>Nuclear-range output</small><strong>${s.first_nuclear_use?'Yes':'No'}</strong><p>${s.first_nuclear_use?escapeHTML(s.first_nuclear_actor)+' · turn '+s.first_nuclear_turn:'No COA ≥ 3 in this case'}</p></div></div>
    <div class="timeline-title">Actor COA paths · evaluation labels, 0–5</div>${chart(result.decisions)}
    <div class="risks"><h4>Strategic-prior diagnostics</h4>${flagged.length?flagged.map(r=>`<div class="risk"><span class="badge ${escapeHTML(r.severity)}">${escapeHTML(r.severity)}</span><span>${escapeHTML(r.label)}</span></div>`).join(''):'<p class="risk">No amber or red indicators in this case.</p>'}<p class="profile-sub">Heuristic signals for review, not validated risk probabilities.</p></div>
    <details class="audit"><summary>Inspect reasoning & uncertainty</summary>${result.decisions.map(d=>`<div class="decision"><strong>${escapeHTML(d.actor)} · turn ${d.turn} · COA ${d.COA}</strong><p>${escapeHTML(d.rationale)}</p><p><small>UNCERTAINTY</small><br>${escapeHTML(d.uncertainty_notes)}</p><p><small>MODEL ASSUMPTIONS</small><br>E = ${Number(d.E_score).toFixed(2)} · mock confidence = ${Number(d.confidence).toFixed(2)}. These are rule-generated scores, not calibrated forecasts.</p><p><small>SOURCE CAVEATS</small><br>${escapeHTML(Array.isArray(d.source_caveats)?d.source_caveats.join('; '):d.source_caveats || 'See source cards below.')}</p></div>`).join('')}</details>
    <details class="audit"><summary>Inspect source cards</summary>${sourceList(result.sources)}</details></article>`;
}

function render() {
  const scenario=data.scenarios.find(s=>s.id===$('scenario').value);
  const treatment=data.treatments.find(t=>t.id===$('treatment').value);
  $('context').hidden=false;
  $('context').innerHTML=`<strong>${escapeHTML(scenario.name)}</strong><small>${escapeHTML(scenario.family)} / ${escapeHTML(scenario.region)}</small><p>${escapeHTML(scenario.summary)}</p><p><b>Treatment:</b> ${escapeHTML(treatment.modifier)}</p>`;
  $('results').innerHTML=Object.entries(currentCase()).map(([label,r])=>profile(label,r)).join('');
  $('status').textContent=`Showing the original offline outputs for ${scenario.name} × ${treatment.name}.`;
}

$('scenario').addEventListener('change',render);
$('treatment').addEventListener('change',render);
$('export').addEventListener('click',()=>{
  const selected={mode:data.mode,turns:data.turns,top_k:data.top_k,scenario:data.scenarios.find(s=>s.id===$('scenario').value),treatment:data.treatments.find(t=>t.id===$('treatment').value),results:currentCase()};
  const url=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=`strateval-mock-${$('scenario').value}-${$('treatment').value}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
fetch('data.json.gz').then(r=>{if(!r.ok)throw Error('Data unavailable');return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json();}).then(payload=>{
  data=payload;
  for(const [id,items] of [['scenario',data.scenarios],['treatment',data.treatments]]){
    $(id).replaceChildren(...items.map(item=>{const o=document.createElement('option');o.value=item.id;o.textContent=item.name;return o;}));$(id).disabled=false;
  }
  $('scenario').value='dual_use_missile_ambiguity';$('treatment').value='ambiguous_intelligence';$('export').disabled=false;render();
}).catch(()=>{$('status').textContent='The offline preview could not load. Reload this page, or run the Python app using the repository instructions below.';});
