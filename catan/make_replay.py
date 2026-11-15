"""Build a self-contained HTML replay of one game: the board move by move, and every model decision
with its probabilities (decision models) or its one-line reason (Claude).

  python make_replay.py runs/<run-folder> [game_index]
Writes runs/<run-folder>/replay_<n>.html
"""
import json
import sys
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catan Replay</title>
<style>
:root{--bg:#f4f1ea;--panel:#fffdf8;--ink:#1f1d1a;--mute:#6f6a60;--line:#e2dccf;--accent:#b4532a;--sea:#cfe3ec;--bar:#2f6f8f;--bar2:#d9e6ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--panel:#1e2125;--ink:#ece8df;--mute:#9a958b;--line:#32363c;--accent:#e08a5a;--sea:#1d3440;--bar:#6fb3d2;--bar2:#2a3a44}}
:root[data-theme="dark"]{--bg:#15171a;--panel:#1e2125;--ink:#ece8df;--mute:#9a958b;--line:#32363c;--accent:#e08a5a;--sea:#1d3440;--bar:#6fb3d2;--bar2:#2a3a44}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 16px 4px;max-width:1200px;margin:0 auto}h1{font-size:20px;margin:0}.sub{color:var(--mute);margin:2px 0 0}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px;max-width:1200px;margin:0 auto;padding:10px 16px 30px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
svg{width:100%;height:auto;display:block}
.controls{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:4px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
.score{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:12px}
.pl{border:1px solid var(--line);border-radius:10px;padding:6px 10px}.pl b{display:block;font-size:13px}
.pl .vp{font-size:22px;font-weight:700}.pl.win{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;border:1px solid #0006;vertical-align:-1px}
.label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:10px 0 4px}
.act{font-weight:600}.reason{font-style:italic;color:var(--mute);margin-top:6px}
.row{display:grid;grid-template-columns:1fr 54px;gap:8px;align-items:center;margin:5px 0;font-size:12.5px}
.track{grid-column:1/3;height:7px;background:var(--bar2);border-radius:4px;overflow:hidden;margin-top:-3px}
.fill{height:100%;background:var(--bar)}.row.pick .opt{font-weight:700}.pct{text-align:right;font-variant-numeric:tabular-nums}
.log{max-height:220px;overflow:auto;font-size:12.5px;color:var(--mute)}.log div{padding:1px 0}
</style></head><body>
<header><h1 id="title"></h1><p class="sub" id="sub"></p></header>
<main>
 <section class="card"><svg id="board" viewBox="-330 -300 660 600" role="img" aria-label="Catan board"></svg>
  <div class="controls"><button id="prev">◀</button><button id="play">▶ Play</button><button id="next">▶</button>
   <input type="range" id="slider" min="0" value="0"><span id="pos" class="sub"></span></div></section>
 <section class="card"><div class="score" id="score"></div>
  <div class="label">Action</div><div id="action" class="act"></div>
  <div id="decision"></div>
  <div class="label">Recent moves</div><div class="log" id="log"></div></section>
</main>
<script>
const D = __DATA__;
const PC = {RED:"#d64541",BLUE:"#2f6fd6",WHITE:"#f4f4f4",ORANGE:"#f08c2e"};
const RC = {WOOD:"#3f7d3a",BRICK:"#b8572f",SHEEP:"#9fcf6a",WHEAT:"#e9c046",ORE:"#8f8f9c",null:"#dccb9a"};
const S = 58, R3 = Math.sqrt(3);
const svg = document.getElementById("board"), NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);(parent||svg).appendChild(e);return e}
function center(c){const [x,,z]=c;return [R3*S*(x+z/2), 1.5*S*z]}
const OFF = {NORTH:[0,-S],NORTHEAST:[R3*S/2,-S/2],SOUTHEAST:[R3*S/2,S/2],SOUTH:[0,S],SOUTHWEST:[-R3*S/2,S/2],NORTHWEST:[-R3*S/2,-S/2]};
const SIDE = {EAST:[1,0],WEST:[-1,0],NORTHEAST:[.5,-.866],NORTHWEST:[-.5,-.866],SOUTHEAST:[.5,.866],SOUTHWEST:[-.5,.866]};
const nodePos = {};
for (const [id,n] of Object.entries(D.geometry.nodes)){const [cx,cy]=center(n.tile_coordinate);const o=OFF[n.direction];nodePos[id]=[cx+o[0],cy+o[1]]}
const pips = n => n ? 6-Math.abs(7-n) : 0;
// static board
el("rect",{x:-330,y:-300,width:660,height:600,fill:"var(--sea)",rx:18});
for (const t of D.geometry.tiles){
  const [cx,cy]=center(t.coordinate), tile=t.tile;
  if (tile.type==="RESOURCE_TILE"||tile.type==="DESERT"){
    const pts=[0,1,2,3,4,5].map(i=>{const a=Math.PI/180*(60*i-90);return (cx+S*Math.cos(a)).toFixed(1)+","+(cy+S*Math.sin(a)).toFixed(1)}).join(" ");
    el("polygon",{points:pts,fill:RC[tile.resource??null],stroke:"#00000033","stroke-width":2});
    if (tile.number){el("circle",{cx,cy,r:15,fill:"#fbf6e9",stroke:"#0003"});
      const tx=el("text",{x:cx,y:cy+5,"text-anchor":"middle","font-size":15,"font-weight":700,fill:pips(tile.number)===5?"#c0392b":"#222"});tx.textContent=tile.number}
  } else if (tile.type==="PORT"){
    const v=SIDE[tile.direction]||[0,0];const px=cx+v[0]*S*.45, py=cy+v[1]*S*.45;
    el("circle",{cx:px,cy:py,r:16,fill:"#fbf6e9",stroke:"#0004"});
    const tx=el("text",{x:px,y:py+4,"text-anchor":"middle","font-size":10,"font-weight":700,fill:tile.resource?RC[tile.resource]:"#333"});
    tx.textContent=tile.resource?("2:1"):"3:1";
  }
}
const dyn = el("g",{});
const recs = D.state.action_records, vps = D.vps, colors = D.state.colors;
const decisions = {}; for (const d of D.trace) decisions[d.idx] = d;
document.getElementById("title").textContent = D.title;
document.getElementById("sub").textContent = D.subtitle;
const slider=document.getElementById("slider"); slider.max=recs.length;
function fmt(a){const [c,t,v]=a[0];const who=D.labels[c]||c;const res=a[1];
  const T={ROLL:()=>`${who} rolls ${Array.isArray(res)?res[0]+"+"+res[1]+" = "+(res[0]+res[1]):""}`,
   BUILD_SETTLEMENT:()=>`${who} builds a settlement`,BUILD_CITY:()=>`${who} builds a city`,BUILD_ROAD:()=>`${who} builds a road`,
   BUY_DEVELOPMENT_CARD:()=>`${who} buys a development card`,PLAY_KNIGHT_CARD:()=>`${who} plays a knight`,
   MOVE_ROBBER:()=>`${who} moves the robber`+(v&&v[1]?` and steals from ${D.labels[v[1]]||v[1]}`:""),
   MARITIME_TRADE:()=>`${who} trades ${v.slice(0,-1).filter(Boolean).length} ${String(v[0]).toLowerCase()} for 1 ${String(v[v.length-1]).toLowerCase()}`,
   END_TURN:()=>`${who} ends the turn`,DISCARD_RESOURCE:()=>`${who} discards 1 ${String(v).toLowerCase()}`,
   PLAY_YEAR_OF_PLENTY:()=>`${who} plays Year of Plenty`,PLAY_MONOPOLY:()=>`${who} plays Monopoly on ${String(v).toLowerCase()}`,
   PLAY_ROAD_BUILDING:()=>`${who} plays Road Building`};
  return (T[t]||(()=>`${who}: ${t}`))()}
function render(i){
  while(dyn.firstChild)dyn.removeChild(dyn.firstChild);
  const roads=[], blds={}; let robber=null;
  for(let k=0;k<i;k++){const [c,t,v]=recs[k][0];
    if(t==="BUILD_ROAD")roads.push([c,v]); else if(t==="BUILD_SETTLEMENT")blds[v]=[c,"S"]; else if(t==="BUILD_CITY")blds[v]=[c,"C"];
    else if(t==="MOVE_ROBBER")robber=v[0];}
  if(!robber){const d=D.geometry.tiles.find(t=>t.tile.type==="DESERT"||(t.tile.type==="RESOURCE_TILE"&&!t.tile.resource));robber=d?d.coordinate:null}
  for(const [c,[a,b]] of roads){const p=nodePos[a],q=nodePos[b];el("line",{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:"#111","stroke-width":9,"stroke-linecap":"round"},dyn);
    el("line",{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:PC[c],"stroke-width":6,"stroke-linecap":"round"},dyn)}
  for(const [n,[c,k]] of Object.entries(blds)){const [x,y]=nodePos[n];
    if(k==="S")el("rect",{x:x-8,y:y-8,width:16,height:16,rx:3,fill:PC[c],stroke:"#111","stroke-width":2},dyn);
    else el("polygon",{points:`${x-11},${y+9} ${x-11},${y-3} ${x-3},${y-11} ${x+5},${y-3} ${x+11},${y-3} ${x+11},${y+9}`,fill:PC[c],stroke:"#111","stroke-width":2},dyn)}
  if(robber){const [x,y]=center(robber);el("circle",{cx:x+22,cy:y-18,r:9,fill:"#222",stroke:"#fff","stroke-width":2},dyn)}
  // score
  const v=vps[Math.min(i,vps.length-1)];const sc=document.getElementById("score");sc.innerHTML="";
  colors.forEach((c,j)=>{const d=document.createElement("div");d.className="pl"+(i===recs.length&&D.winner===c?" win":"");
    d.innerHTML=`<b><span class="dot" style="background:${PC[c]}"></span>${D.labels[c]}</b><span class="vp">${v[j]}</span> pts`;sc.appendChild(d)});
  document.getElementById("pos").textContent=`move ${i} / ${recs.length}`;
  const a=i>0?recs[i-1]:null; document.getElementById("action").textContent=a?fmt(a):"Game start";
  // decision card: the model decision that produced the action just played
  const dec=i>0?decisions[i-1]:null, box=document.getElementById("decision");box.innerHTML="";
  if(dec){const h=document.createElement("div");h.className="label";h.textContent=`${dec.player} chose from ${dec.n_options} legal actions`+(dec.confidence!=null?` · confidence ${(dec.confidence*100).toFixed(0)}%`:"");box.appendChild(h);
    if(dec.top&&dec.top.length){for(const [opt,p] of dec.top){const r=document.createElement("div");r.className="row"+(opt===dec.chosen?" pick":"");
      r.innerHTML=`<span class="opt"></span><span class="pct">${(p*100).toFixed(1)}%</span><div class="track"><div class="fill" style="width:${(p*100).toFixed(1)}%"></div></div>`;
      r.querySelector(".opt").textContent=opt;box.appendChild(r)}}
    else{const r=document.createElement("div");r.className="act";r.textContent=dec.chosen;box.appendChild(r)}
    if(dec.lookahead&&dec.lookahead.length){const h2=document.createElement("div");h2.className="label";h2.textContent="Looked one move ahead: position score (0 = losing, 4 = winning)";box.appendChild(h2);
      for(const [opt,sc] of dec.lookahead){const r=document.createElement("div");r.className="row"+(opt===dec.chosen?" pick":"");
        r.innerHTML=`<span class="opt"></span><span class="pct">${sc.toFixed(2)}</span><div class="track"><div class="fill" style="width:${(sc/4*100).toFixed(1)}%"></div></div>`;
        r.querySelector(".opt").textContent=opt;box.appendChild(r)}}
    if(dec.reason){const r=document.createElement("div");r.className="reason";r.textContent="“"+dec.reason+"”";box.appendChild(r)}}
  const log=document.getElementById("log");log.innerHTML="";
  for(let k=Math.max(0,i-12);k<i;k++){const t=recs[k][0][1];if(t==="END_TURN")continue;const d=document.createElement("div");d.textContent=fmt(recs[k]);log.prepend(d)}
  slider.value=i;
}
let cur=0,timer=null;
function go(i){cur=Math.max(0,Math.min(recs.length,i));render(cur)}
document.getElementById("prev").onclick=()=>go(cur-1);document.getElementById("next").onclick=()=>go(cur+1);
slider.oninput=()=>go(+slider.value);
const playBtn=document.getElementById("play");
playBtn.onclick=()=>{if(timer){clearInterval(timer);timer=null;playBtn.textContent="▶ Play";return}
  playBtn.textContent="❚❚ Pause";timer=setInterval(()=>{if(cur>=recs.length){clearInterval(timer);timer=null;playBtn.textContent="▶ Play";return}
  go(cur+1)},D.speed||350)};
go(0);
</script></body></html>
"""


def main():
    run = Path(sys.argv[1])
    g = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rep = json.loads((run / f"game_{g}_replay.json").read_text())
    trace = [json.loads(l) for l in (run / f"game_{g}.jsonl").read_text().splitlines() if l.strip()]
    summ = json.loads((run / "summary.json").read_text())
    res = next(r for r in summ["results"] if r["game"] == g)
    labels = rep["labels"]
    winner_color = next((c for c, l in labels.items() if l == res["winner"]), None)
    names = ", ".join(labels[c] for c in rep["state"]["colors"])
    data = {
        "labels": labels, "vps": rep["vps"], "geometry": rep["geometry"],
        "state": {"colors": rep["state"]["colors"], "action_records": rep["state"]["action_records"]},
        "trace": trace, "winner": winner_color,
        "title": f"Catan: {names}",
        "subtitle": (f"Winner: {res['winner']} · final points "
                     + ", ".join(f"{k} {v}" for k, v in res["points"].items())
                     + f" · {res['turns']} turns"),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = run / f"replay_{g}.html"
    out.write_text(html)
    print(out)


if __name__ == "__main__":
    main()
