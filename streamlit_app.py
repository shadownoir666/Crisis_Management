"""
streamlit_app.py  —  AEGIS Crisis Management AI  (LangGraph Edition)
======================================================================
Run:  streamlit run streamlit_app.py

Architecture — three-phase LangGraph execution with TWO admin approval gates
-------------------------------------------------------------------------
Phase 1 : vision → store_zone → drone_analysis → drone_decision
          → drone_dispatch → drone_vision → update_people
          → rescue_decision  →  INTERRUPT (before admin_resource)

Phase 2 : admin_resource decision injected  →  route_planner
          →  INTERRUPT (before admin_route)

Phase 3 : admin_route decision injected  →  communication  →  END

All stages read from the LangGraph MemorySaver checkpoint — real agent output.
"""

import streamlit as st

st.set_page_config(
    page_title="AEGIS — Crisis Management AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io, os, sys, json, sqlite3, contextlib, tempfile, traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import folium
from PIL import Image
from streamlit_folium import st_folium

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DB_PATH = os.path.join(_ROOT, "crisis.db")

# ============================================================================
#  LANGGRAPH — build + compile ONCE per server process
# ============================================================================

@st.cache_resource
def _get_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, END
    from master_agent.master_state import MasterState
    from master_agent.master_nodes import (
        vision_node, store_zone_node, drone_analysis_node,
        drone_decision_node, drone_dispatch_node, drone_vision_node,
        update_people_node, rescue_decision_node,
        admin_resource_node, resource_approval_router,
        route_planner_node, admin_route_node, route_approval_router,
        communication_node,
    )

    b = StateGraph(MasterState)
    for name, fn in [
        ("vision",          vision_node),
        ("store_zone",      store_zone_node),
        ("drone_analysis",  drone_analysis_node),
        ("drone_decision",  drone_decision_node),
        ("drone_dispatch",  drone_dispatch_node),
        ("drone_vision",    drone_vision_node),
        ("update_people",   update_people_node),
        ("rescue_decision", rescue_decision_node),
        ("admin_resource",  admin_resource_node),
        ("route_planner",   route_planner_node),
        ("admin_route",     admin_route_node),
        ("communication",   communication_node),
    ]:
        b.add_node(name, fn)

    b.set_entry_point("vision")
    for src, dst in [
        ("vision",          "store_zone"),
        ("store_zone",      "drone_analysis"),
        ("drone_analysis",  "drone_decision"),
        ("drone_decision",  "drone_dispatch"),
        ("drone_dispatch",  "drone_vision"),
        ("drone_vision",    "update_people"),
        ("update_people",   "rescue_decision"),
        ("rescue_decision", "admin_resource"),
        ("route_planner",   "admin_route"),
        ("communication",   END),
    ]:
        b.add_edge(src, dst)

    b.add_conditional_edges("admin_resource", resource_approval_router,
                            {"approved": "route_planner", "rejected": "rescue_decision"})
    b.add_conditional_edges("admin_route", route_approval_router,
                            {"approved": "communication", "rejected": "route_planner"})

    return b.compile(
        checkpointer     = MemorySaver(),
        interrupt_before = ["admin_resource", "admin_route"],
    )


def _cfg():
    return {"configurable": {"thread_id": st.session_state.get("thread_id", "aegis_main")}}


def _graph_state() -> dict:
    try:
        snap = _get_graph().get_state(_cfg())
        if snap and snap.values:
            return dict(snap.values)
    except Exception:
        pass
    return {}


def _next_nodes() -> list:
    try:
        snap = _get_graph().get_state(_cfg())
        return list(snap.next) if snap and snap.next else []
    except Exception:
        return []


# ── Phase runners ─────────────────────────────────────────────────────────────

def _invoke(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    if buf.getvalue().strip():
        _log(buf.getvalue().strip())
    return result


def _run_phase1(img_path: str, image_meta: dict):
    _log("LangGraph Phase 1 starting …")
    from dotenv import load_dotenv; load_dotenv()
    _invoke(
        _get_graph().invoke,
        {
            "satellite_image": img_path,
            "image_meta":      image_meta,
            "base_locations": {
                "ambulance":   {"name": "Hospital",      "lat": 19.06546856543151,  "lon": 72.86100899070198},
                "rescue_team": {"name": "Rescue Center", "lat": 19.06847079812735,  "lon": 72.85793995490616},
                "boat":        {"name": "Boat Depot",    "lat": 19.063380373548366, "lon": 72.85538649195271},
            },
            "field_reports":   [],
            "dispatch_config": {
                "send_sms":       bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("YOUR_PHONE_NUMBER")),
                "generate_audio": True,
                "language":       st.session_state.get("comm_language", "English"),
                "to_number":      os.getenv("YOUR_PHONE_NUMBER"),
            },
        },
        config=_cfg(),
    )
    st.session_state["pipeline_phase"] = "awaiting_resource"
    _log("Phase 1 complete — interrupted before admin_resource")


def _run_phase2(approved: bool):
    graph = _get_graph(); config = _cfg()
    graph.update_state(config, {"resource_approved": approved}, as_node="admin_resource")
    _invoke(graph.invoke, None, config=config)
    st.session_state["pipeline_phase"] = "awaiting_route" if approved else "awaiting_resource"
    if approved:
        _log("Phase 2 complete — interrupted before admin_route")


def _run_phase3(approved: bool):
    graph = _get_graph(); config = _cfg()
    graph.update_state(config, {"route_approved": approved}, as_node="admin_route")
    _invoke(graph.invoke, None, config=config)
    st.session_state["pipeline_phase"] = "complete" if approved else "awaiting_route"
    if approved:
        _log("Phase 3 complete — pipeline DONE")


# ============================================================================
#  THEME
# ============================================================================

THEME = {
    "bg":     "#080d14", "bg2":    "#0d1520",
    "cyan":   "#00d4ff", "red":    "#ff2d55",
    "orange": "#ff9500", "green":  "#30d158",
    "yellow": "#ffd60a", "text":   "#e5e5e7",
    "mono":   "#00ff88",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap');
  body,.stApp{{background:{THEME['bg']};color:{THEME['text']};}}
  h1,h2,h3{{font-family:'Rajdhani',sans-serif;}}
  .stButton>button{{background:{THEME['bg2']};color:{THEME['cyan']};border:2px solid {THEME['cyan']};
    border-radius:6px;padding:8px 18px;font-family:'Share Tech Mono',monospace;transition:.25s;}}
  .stButton>button:hover{{background:{THEME['cyan']};color:{THEME['bg']};}}
  .stMetric{{background:{THEME['bg2']};padding:16px;border-radius:8px;border-left:4px solid {THEME['cyan']};}}
  .terminal-log{{background:#000;color:{THEME['mono']};font-family:'Share Tech Mono',monospace;
    font-size:11px;line-height:1.6;padding:12px;border-radius:4px;border:1px solid {THEME['green']};
    max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;}}
  .card{{background:{THEME['bg2']};border-left:4px solid {THEME['cyan']};border-radius:6px;
    padding:12px;margin:6px 0;}}
  .card-warn{{background:#1a0f00;border-left:4px solid {THEME['orange']};border-radius:6px;
    padding:12px;margin:6px 0;}}
  div[data-testid="stDataFrame"]{{background:{THEME['bg2']};}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
#  CONSTANTS
# ============================================================================

_DEFAULT_META = {
    "center_lat": 19.062061, "center_lon": 72.863542,
    "coverage_km": 1.6, "width_px": 1024, "height_px": 522,
}
ROUTE_COLORS   = {"ambulance":"#e74c3c","rescue_team":"#2980b9","boat":"#16a085",
                  "helicopter":"#8e44ad","fire_truck":"#e67e22","truck":"#7f8c8d"}
RESOURCE_EMOJI = {"ambulance":"🚑","rescue_team":"🚒","boat":"🚤",
                  "helicopter":"🚁","fire_truck":"🚒","truck":"🚛"}
BASE_ICON      = {"ambulance":("red","plus-sign"),"rescue_team":("blue","home"),
                  "boat":("darkblue","tint"),"helicopter":("purple","plane")}
_DEFAULT_BASES = {
    "ambulance":   {"name":"Hospital",      "lat":19.06546856543151,  "lon":72.86100899070198},
    "rescue_team": {"name":"Rescue Center", "lat":19.06847079812735,  "lon":72.85793995490616},
    "boat":        {"name":"Boat Depot",    "lat":19.063380373548366, "lon":72.85538649195271},
}

STAGES = [
    "1️⃣ Upload","2️⃣ Zone Map","3️⃣ Drones","4️⃣ Gallery","5️⃣ Analysis",
    "6️⃣ Resources","7️⃣ Approve I","8️⃣ Routes","9️⃣ Approve II","🔟 Comms",
]

_PHASE_INFO = {
    "idle":              ("#555",          "⚫ Idle"),
    "running_phase1":    (THEME["yellow"], "🟡 Phase 1 — Running Agents"),
    "awaiting_resource": (THEME["orange"], "🟠 Awaiting Resource Approval"),
    "running_phase2":    (THEME["yellow"], "🟡 Phase 2 — Planning Routes"),
    "awaiting_route":    (THEME["orange"], "🟠 Awaiting Route Approval"),
    "running_phase3":    (THEME["yellow"], "🟡 Phase 3 — Dispatching"),
    "complete":          (THEME["green"],  "🟢 Pipeline Complete"),
}

# ============================================================================
#  HELPERS
# ============================================================================

def _ts(): return datetime.now().strftime("%H:%M:%S")

def _log(text):
    st.session_state.setdefault("log", "")
    for line in (text or "").strip().splitlines():
        if line.strip():
            st.session_state["log"] += f"[{_ts()}] {line}\n"

def _terminal():
    log = st.session_state.get("log", "(no output yet)")
    st.markdown(
        f'<div class="terminal-log" id="tlog">{log}</div>'
        '<script>var t=document.getElementById("tlog");if(t)t.scrollTop=t.scrollHeight;</script>',
        unsafe_allow_html=True)

def _save_upload(f):
    suffix = Path(f.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue()); tmp.close()
    return tmp.name

def _sev_label(s):
    if s >= 0.8: return "🔴 CRITICAL"
    if s >= 0.6: return "🟠 HIGH"
    if s >= 0.4: return "🟡 MODERATE"
    return "🟢 LOW"

def _rcolor(rt):
    k = rt.lower().rstrip("s")
    return ROUTE_COLORS.get(k, ROUTE_COLORS.get(rt.lower(), "#888"))

def _remoji(rt):
    k = rt.lower().rstrip("s")
    return RESOURCE_EMOJI.get(k, RESOURCE_EMOJI.get(rt.lower(), "🚗"))

# ── _nav: unique key via counter reset each render cycle ─────────────────────
_nav_calls = {}

def _reset_nav_counter():
    """Call once at the top of main() each render cycle."""
    _nav_calls.clear()

def _nav(back=None, fwd=None, fwd_label="▶ PROCEED"):
    stage = st.session_state.get("stage", 0)
    _nav_calls[stage] = _nav_calls.get(stage, 0) + 1
    uid = f"{stage}_{_nav_calls[stage]}"
    c1, c2 = st.columns(2)
    with c1:
        if back is not None and st.button("◀ BACK", key=f"nb_{uid}"):
            st.session_state["stage"] = back; st.rerun()
    with c2:
        if fwd is not None and st.button(fwd_label, key=f"nf_{uid}"):
            st.session_state["stage"] = fwd; st.rerun()

def _phase_badge():
    phase = st.session_state.get("pipeline_phase", "idle")
    color, label = _PHASE_INFO.get(phase, ("#555", phase))
    st.markdown(
        f'<span style="background:{color};color:#000;padding:4px 14px;border-radius:12px;'
        f'font-family:\'Share Tech Mono\',monospace;font-size:11px;font-weight:bold;">'
        f'{label}</span><br><br>', unsafe_allow_html=True)

# ============================================================================
#  SIDEBAR
# ============================================================================

def _sidebar():
    with st.sidebar:
        st.markdown("### 🛰️ AEGIS · LangGraph Pipeline")
        st.divider()

        phase = st.session_state.get("pipeline_phase", "idle")
        color, label = _PHASE_INFO.get(phase, ("#555", phase))
        st.markdown(f'<div class="card">🔗 <b>Master Graph</b><br>'
                    f'<span style="color:{color};font-size:12px;">{label}</span></div>',
                    unsafe_allow_html=True)

        nxt = _next_nodes()
        if nxt:
            st.markdown(f'<div class="card">⏸️ <b>Interrupted Before</b><br>'
                        f'<span style="color:{THEME["cyan"]};font-size:12px;">'
                        f'{", ".join(nxt)}</span></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown(
            f'<div class="card" style="font-family:\'Share Tech Mono\',monospace;font-size:10px;">'
            f'<b style="color:{THEME["cyan"]};">LangGraph Node Order</b><br><br>'
            f'vision<br>&nbsp;└─ store_zone<br>&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_analysis<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_decision<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_dispatch<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_vision<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ update_people<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ rescue_decision<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ <span style="color:{THEME["orange"]};">[⏸ admin_resource]</span><br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ route_planner<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ <span style="color:{THEME["orange"]};">[⏸ admin_route]</span><br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ communication<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ END'
            f'</div>', unsafe_allow_html=True)

        st.divider()
        gs = _graph_state()
        for name, icon, key in [
            ("Vision Agent",   "👁️",  "zone_map"),
            ("Drone Agent",    "🚁",  "drone_allocation"),
            ("Resource Agent", "📦",  "rescue_plan"),
            ("Route Agent",    "🗺️", "route_plan"),
            ("Comm Agent",     "📡",  "dispatch_result"),
        ]:
            val = gs.get(key)
            if val:
                if key == "rescue_plan" and gs.get("resource_approved"):
                    s = f'<span style="color:{THEME["green"]};">🟢 Approved</span>'
                elif key == "route_plan" and gs.get("route_approved"):
                    s = f'<span style="color:{THEME["green"]};">🟢 Approved</span>'
                else:
                    s = f'<span style="color:{THEME["green"]};">🟢 Done</span>'
            else:
                s = f'<span style="color:#888;">⚪ Idle</span>'
            st.markdown(f'<div class="card">{icon} <b>{name}</b><br>{s}</div>',
                        unsafe_allow_html=True)

        st.divider()
        st.metric("Stage", f"{st.session_state.get('stage',0)+1} / {len(STAGES)}")
        gs_filled = sum(1 for v in gs.values() if v is not None) if gs else 0
        if gs_filled:
            st.metric("State Fields", f"{gs_filled} populated")
        st.divider()
        if st.button("🔄 Full Reset", use_container_width=True):
            import uuid
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            st.rerun()

# ============================================================================
#  STEPPER
# ============================================================================

def _stepper():
    s = st.session_state.get("stage", 0)
    cols = st.columns(len(STAGES))
    for i, label in enumerate(STAGES):
        with cols[i]:
            if i < s:    bg, fg = THEME["green"], "black"
            elif i == s: bg, fg = THEME["cyan"],  THEME["bg"]
            else:         bg, fg = THEME["bg2"],   THEME["text"]
            st.markdown(
                f'<div style="background:{bg};color:{fg};padding:6px 2px;text-align:center;'
                f'border-radius:4px;font-size:9px;font-weight:bold;'
                f'border:1px solid {THEME["cyan"]};">{"✅ " if i<s else ""}{label}</div>',
                unsafe_allow_html=True)

# ============================================================================
#  FOLIUM MAP
# ============================================================================

def _folium_map():
    gs = _graph_state()
    routes = gs.get("route_plan", [])
    meta   = gs.get("image_meta") or _DEFAULT_META
    fmap   = folium.Map(location=[meta.get("center_lat",19.06), meta.get("center_lon",72.86)],
                        zoom_start=15, tiles="CartoDB positron")

    seen_b, seen_z = set(), set()
    for r in routes:
        rk   = r.get("resource_type","").lower().rstrip("s")
        base = _DEFAULT_BASES.get(rk)
        if base and base["name"] not in seen_b:
            ic_c, ic_i = BASE_ICON.get(rk, ("gray","info-sign"))
            folium.Marker([base["lat"],base["lon"]], tooltip=f"📍 {base['name']}",
                          icon=folium.Icon(color=ic_c,icon=ic_i)).add_to(fmap)
            seen_b.add(base["name"])
        dest = r.get("destination_latlon")
        if dest and r.get("zone") not in seen_z:
            folium.Marker(list(dest), tooltip=f"🚨 Zone {r['zone']}",
                          icon=folium.Icon(color="orange",icon="exclamation-sign")).add_to(fmap)
            seen_z.add(r["zone"])

    for r in routes:
        if not r.get("success"): continue
        color = _rcolor(r["resource_type"]); emoji = _remoji(r["resource_type"])
        wpts  = r.get("waypoints", [])
        if len(wpts) < 2:
            dest = r.get("destination_latlon"); rk = r["resource_type"].lower().rstrip("s")
            base = _DEFAULT_BASES.get(rk)
            if base and dest: wpts = [(base["lat"],base["lon"]),dest]
            else: continue
        folium.PolyLine(wpts, color=color, weight=5, opacity=0.9,
                        tooltip=(f"{emoji} {r.get('unit_count',1)}× {r['resource_type']}\n"
                                 f"Zone {r['zone']} · {r.get('distance_km',0)} km "
                                 f"ETA {r.get('eta_minutes',0)} min")).add_to(fmap)
        for lat, lon in wpts[1:-1]:
            folium.CircleMarker([lat,lon],radius=3,color=color,fill=True,
                                fill_color=color,fill_opacity=0.8).add_to(fmap)
        folium.Marker(list(wpts[-1]), tooltip=f"{emoji} Zone {r['zone']}",
                      icon=folium.DivIcon(html=f'<div style="font-size:16px;color:{color};">▼</div>',
                                          icon_size=(16,16),icon_anchor=(8,8))).add_to(fmap)

    seen_types = sorted({r["resource_type"] for r in routes if r.get("success")})
    lines = "".join(f'<span style="color:{_rcolor(t)};font-size:16px;">━━</span> '
                    f'{_remoji(t)} {t.replace("_"," ").title()}<br>' for t in seen_types)
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:12px;right:12px;z-index:9999;background:white;'
        f'padding:12px 16px;border-radius:8px;border:2px solid #333;font-family:Arial;'
        f'font-size:12px;box-shadow:3px 3px 8px rgba(0,0,0,.3);">'
        f'<b>🗺 Route Legend</b><br>{lines}'
        f'<span style="background:#f39c12;padding:0 5px;border-radius:3px;">■</span> Crisis Zone<br>'
        f'<hr style="margin:6px 0;"><span style="font-size:10px;color:#666;">'
        f'LangGraph · Real OSM waypoints</span></div>'))
    return fmap

# ============================================================================
#  STAGE 1 — UPLOAD
# ============================================================================

def stage_1():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🖼️ Stage 1: Satellite Image Upload</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    c1, c2 = st.columns([1,1])
    with c1:
        uploaded = st.file_uploader("Upload satellite / aerial image",
                                    type=["jpg","jpeg","png"])
        if uploaded:
            st.session_state["upload_obj"] = uploaded
            pil = Image.open(uploaded)
            st.session_state["upload_pil"] = pil
            st.image(pil, caption=f"{uploaded.name}  ({pil.width}×{pil.height} px)",
                     use_container_width=True)
        st.markdown("### 📍 Geo Parameters")
        lat = st.number_input("Center Latitude",  value=19.062061, format="%.6f")
        lon = st.number_input("Center Longitude", value=72.863542, format="%.6f")
        cov = st.number_input("Coverage (km)",    value=1.60, min_value=0.1)
        pil_ref = st.session_state.get("upload_pil", Image.new("RGB",(1024,522)))
        st.session_state["image_meta"] = {
            "center_lat":lat,"center_lon":lon,"coverage_km":cov,
            "width_px":pil_ref.width,"height_px":pil_ref.height}
    with c2:
        st.markdown("**System Logs**"); _terminal()
    st.divider()
    if st.session_state.get("upload_obj"):
        if st.button("▶ PROCEED — START LANGGRAPH PIPELINE", key="btn1"):
            _log("Image accepted — LangGraph Phase 1 starting …")
            st.session_state["pipeline_phase"] = "running_phase1"
            st.session_state["stage"] = 1
            st.rerun()
    else:
        st.info("Upload a satellite image to begin.")

# ============================================================================
#  STAGE 2 — ZONE MAP  (Phase 1 runs here on first load)
# ============================================================================

def stage_2():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🗺️ Stage 2: Zone Map Analysis</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    # Trigger Phase 1 if we just came from Stage 1
    if st.session_state.get("pipeline_phase") == "running_phase1":
        upload_obj = st.session_state.get("upload_obj")
        img_path   = _save_upload(upload_obj) if upload_obj else "Images_for_testing/image.png"
        st.session_state["img_path"] = img_path
        meta = st.session_state.get("image_meta", _DEFAULT_META.copy())
        with st.spinner("🚀 LangGraph Phase 1 running — vision → drones → resource LLM …  (~60-120 s)"):
            try:
                _run_phase1(img_path, meta)
                st.success("✅ Phase 1 complete — graph interrupted before admin_resource")
            except Exception as e:
                _log(f"[ERROR] Phase 1:\n{traceback.format_exc()}")
                st.error(f"Phase 1 failed: {e}"); st.code(traceback.format_exc())
                _nav(back=0); return
        st.rerun()

    gs = _graph_state(); zone_map = gs.get("zone_map", {})
    if not zone_map:
        st.warning("Zone map not in graph state yet. Check terminal for errors.")
        _terminal(); st.divider(); _nav(back=0); return

    c1, c2 = st.columns([3,2])
    with c1:
        grid = os.path.join(_ROOT,"zone_results","grid_output.jpg")
        if os.path.exists(grid):
            st.image(Image.open(grid), caption="Zone Severity Grid (10×10)",
                     use_container_width=True)
        elif st.session_state.get("upload_pil"):
            st.image(st.session_state["upload_pil"], caption="Uploaded Image",
                     use_container_width=True)
    with c2:
        st.markdown("**Top Affected Zones**")
        top = sorted(zone_map.items(), key=lambda x:x[1].get("severity",0), reverse=True)[:15]
        st.dataframe(pd.DataFrame([{
            "Zone":zid, "Sev":f'{d.get("severity",0):.3f}',
            "Flood":f'{d.get("flood_score",0):.3f}',
            "Damage":f'{d.get("damage_score",0):.3f}',
            "Level":_sev_label(d.get("severity",0))}
            for zid,d in top]),
            use_container_width=True, hide_index=True)
        st.success(f"✅ {len(zone_map)} zones analysed via LangGraph Vision Node")

    _terminal(); st.divider()
    _nav(back=0, fwd=2, fwd_label="▶ PROCEED TO DRONE ALLOCATION")

# ============================================================================
#  STAGE 3 — DRONE ALLOCATION
# ============================================================================

def stage_3():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🚁 Stage 3: Drone Allocation</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs = _graph_state()
    alloc = gs.get("drone_allocation", {})
    most  = gs.get("most_affected_zones", [])

    if not alloc:
        st.warning("Drone allocation not in graph state yet.")
        _terminal(); st.divider(); _nav(back=1); return

    if most:
        st.info(f"Top affected zones (from crisis.db): **{', '.join(most)}**")

    n = min(len(alloc), 6)
    if n:
        cols = st.columns(n)
        for i,(d_id,z_id) in enumerate(list(alloc.items())[:n]):
            with cols[i]:
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<b style="color:{THEME["cyan"]};">{d_id.upper()}</b><br>'
                    f'<span style="font-size:22px;">🚁</span><br>→ <b>{z_id}</b><br>'
                    f'<span style="color:{THEME["green"]};font-size:12px;">✅ DISPATCHED</span>'
                    f'</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame([{"Drone":k,"Zone":v,"Status":"✅ Dispatched"}
                                for k,v in alloc.items()]),
                 use_container_width=True, hide_index=True)
    _terminal(); st.divider()
    _nav(back=1, fwd=3, fwd_label="▶ PROCEED TO GALLERY")

# ============================================================================
#  STAGE 4 — GALLERY
#  Shows: original drone images FIRST, then annotated detection results,
#         then people counts table — all from LangGraph checkpoint
# ============================================================================

def stage_4():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📸 Stage 4: Drone Imagery Gallery</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    gs     = _graph_state()
    alloc  = gs.get("drone_allocation", {})
    counts = gs.get("people_counts", {})
    zimmap = gs.get("zone_image_map",  {})
    akv    = list(alloc.items())

    # ── Section A: Original drone images ─────────────────────────────────
    st.markdown(f'<h4 style="color:{THEME["cyan"]};">📷 Raw Drone Footage</h4>',
                unsafe_allow_html=True)

    imgs = {}
    p = Path(os.path.join(_ROOT, "zone_images"))
    if p.exists():
        for f in sorted(list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png"))):
            try: imgs[f.stem] = Image.open(f)
            except: pass

    if imgs:
        cols = st.columns(3)
        for idx, (name, img) in enumerate(list(imgs.items())[:9]):
            with cols[idx % 3]:
                st.image(img, use_container_width=True)
                d_id   = akv[idx % len(akv)][0] if akv else f"drone_{idx+1}"
                z_id   = alloc.get(d_id, "—")
                pcount = counts.get(z_id)
                if pcount is not None:
                    st.caption(f"**{name}** · {d_id} → {z_id}  |  👤 **{pcount} people**")
                else:
                    st.caption(f"**{name}** · {d_id} → {z_id}")
    else:
        st.info("No zone images found in `zone_images/`. Place aerial photos there and rerun.")

    # ── Section B: Annotated results (bounding boxes) ─────────────────────
    rp = Path(os.path.join(_ROOT, "zone_results"))
    annotated = {}
    if rp.exists():
        for f in sorted(rp.glob("*_analysis.jpg")):
            zone_key = f.stem.replace("_analysis", "")
            try: annotated[zone_key] = Image.open(f)
            except: pass

    if annotated:
        st.divider()
        st.markdown(f'<h4 style="color:{THEME["cyan"]};">🔍 YOLO Detection Results (with bounding boxes)</h4>',
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (zone_id, img) in enumerate(annotated.items()):
            with cols[idx % 3]:
                st.image(img, use_container_width=True)
                n_ppl = counts.get(zone_id, 0)
                st.caption(f"Zone **{zone_id}** — 👤 {n_ppl} people detected")

    # ── Section C: People counts table ────────────────────────────────────
    if counts:
        st.divider()
        st.markdown("**👥 People Count Summary by Zone**")
        df = pd.DataFrame([{"Zone": k, "👤 People": v,
                             "Status": "✅ Detected" if v > 0 else "⚠️ 0 detected"}
                           for k, v in counts.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        total = sum(counts.values())
        st.success(f"✅ **{total} people** detected across **{len(counts)} zones** "
                   f"— results saved to crisis.db")
    elif alloc:
        st.info("People detection ran as part of Phase 1 — counts will appear here after run.")

    _terminal(); st.divider()
    _nav(back=2, fwd=4, fwd_label="▶ PROCEED TO ANALYSIS")

# ============================================================================
#  STAGE 5 — ANALYSIS
# ============================================================================

def stage_5():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📊 Stage 5: Zone Analysis Results</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs = _graph_state()
    people = gs.get("people_counts", {})
    zm     = gs.get("zone_map", {})
    top    = gs.get("most_affected_zones", [])

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Zone Severity, Flood & People**")
        zones = top or list(zm.keys())[:15]
        rows  = [{"Zone":z, "👤 People":people.get(z,0),
                  "Severity":f'{zm.get(z,{}).get("severity",0):.3f}',
                  "Flood":f'{zm.get(z,{}).get("flood_score",0):.3f}',
                  "Damage":f'{zm.get(z,{}).get("damage_score",0):.3f}',
                  "Level":_sev_label(zm.get(z,{}).get("severity",0))}
                 for z in zones]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if os.path.exists(DB_PATH):
            try:
                conn=sqlite3.connect(DB_PATH)
                df_db=pd.read_sql_query(
                    "SELECT zone_id,severity,flood_score,damage_score,people_count,last_updated "
                    "FROM zones ORDER BY severity DESC LIMIT 10", conn)
                conn.close()
                st.markdown("**📦 crisis.db — Live Snapshot**")
                st.dataframe(df_db, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"DB read error: {e}")
    with c2:
        st.markdown("**Zone Grid + Detection Images**")
        shown = 0
        rp = Path(os.path.join(_ROOT,"zone_results"))
        if rp.exists():
            for f in sorted(list(rp.glob("*.jpg"))+list(rp.glob("*.png"))):
                if f.name.startswith("route_map"): continue
                try:
                    st.image(Image.open(f), caption=f.stem, use_container_width=True)
                    shown += 1
                except: pass
                if shown >= 5: break
        if not shown:
            st.info("Result images appear in `zone_results/` after Phase 1 runs.")

    _terminal(); st.divider()
    _nav(back=3, fwd=5, fwd_label="▶ PROCEED TO RESOURCE ALLOCATION")

# ============================================================================
#  STAGE 6 — RESOURCE ALLOCATION  (display from LangGraph checkpoint)
# ============================================================================

def stage_6():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📦 Stage 6: Resource Allocation</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs   = _graph_state()
    plan = gs.get("rescue_plan", {})

    if not plan:
        st.warning("Rescue plan not in graph state yet — check terminal for errors.")
        _terminal(); st.divider(); _nav(back=4); return

    st.success("✅ Rescue plan generated by Gemini LLM via rescue_decision_node (Phase 1)")
    st.caption("ℹ️  Results are read from the LangGraph checkpoint — LLM ran in Phase 1.")

    rows = []; totals = {}
    for z, alloc in plan.items():
        row = {"Zone": z}
        for rt, cnt in alloc.items():
            row[rt] = cnt; totals[rt] = totals.get(rt, 0) + cnt
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).fillna(0), use_container_width=True, hide_index=True)

    if totals:
        st.divider()
        mc = st.columns(len(totals))
        for i,(k,v) in enumerate(totals.items()):
            with mc[i]: st.metric(k.replace("_"," ").title(), int(v))

    _terminal(); st.divider()
    _nav(back=4, fwd=6, fwd_label="▶ PROCEED TO APPROVAL GATE")

# ============================================================================
#  STAGE 7 — ADMIN APPROVAL GATE 1  ← FIRST of TWO approval gates
# ============================================================================

def stage_7():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">✅ Stage 7: Admin Approval Gate #1 — Resources</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs    = _graph_state()
    plan  = gs.get("rescue_plan", {})
    phase = st.session_state.get("pipeline_phase", "idle")

    if not plan:
        st.warning("No rescue plan in graph state — go back to Stage 6.")
        _nav(back=5); return

    # Already approved — show status and proceed button
    if gs.get("resource_approved") or phase in ("awaiting_route","running_phase2","running_phase3","complete"):
        st.success("✅ Resource allocation APPROVED — route planning has been triggered.")
        _nav(back=5, fwd=7, fwd_label="▶ VIEW ROUTE PLANNING")
        return

    st.markdown("**Proposed Rescue Resource Allocation (Gemini LLM)**")
    for z, alloc in plan.items():
        desc = " · ".join(f"{v}× {k}" for k,v in alloc.items() if v)
        st.markdown(f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b>  →  {desc}</div>',
                    unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE — TRIGGER ROUTE PLANNING", key="app1", use_container_width=True):
            _log("ADMIN ✓ Resource allocation APPROVED — resuming LangGraph (Phase 2) …")
            st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🗺️ Route Agent planning OSM routes … (~30-60 s)"):
                try:
                    _run_phase2(approved=True)
                    st.success("✅ Routes planned!"); st.balloons()
                except Exception as e:
                    _log(f"[ERROR] Phase 2:\n{traceback.format_exc()}")
                    st.error(f"Phase 2 error: {e}"); st.code(traceback.format_exc())
            st.rerun()
    with c2:
        if st.button("🔴 REJECT — RE-RUN LLM", key="hold1", use_container_width=True):
            import uuid
            _log("ADMIN ✗ Rejected — restarting Phase 1 with new thread …")
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            img_path = st.session_state.get("img_path","Images_for_testing/image.png")
            meta     = st.session_state.get("image_meta", _DEFAULT_META.copy())
            st.session_state["pipeline_phase"] = "running_phase1"
            with st.spinner("🔄 Re-running Phase 1 …"):
                try: _run_phase1(img_path, meta)
                except Exception as e: _log(f"[ERROR] Re-run: {e}")
            st.rerun()
    _terminal()

# ============================================================================
#  STAGE 8 — ROUTE PLANNING
# ============================================================================

def stage_8():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🗺️ Stage 8: Route Planning</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    if st.session_state.get("pipeline_phase") == "running_phase2":
        st.info("🗺️ Route planning is running … check terminal for progress.")
        _terminal(); return

    gs     = _graph_state()
    routes = gs.get("route_plan", [])

    if not routes:
        st.warning("Route plan is not in graph state yet. Check terminal for errors.")
        latest_map = os.path.join(_ROOT,"zone_results","route_map_latest.html")
        if os.path.exists(latest_map):
            st.info(f"💡 Routes were computed and HTML map saved at `{latest_map}` "
                    f"but failed to persist to LangGraph checkpoint. "
                    f"Ensure geo_reference.py and master_nodes.py are updated.")
        _terminal(); st.divider(); _nav(back=6); return

    st.success(f"✅ {sum(1 for r in routes if r.get('success'))} / {len(routes)} routes planned")
    st.dataframe(pd.DataFrame([{
        "Zone":r.get("zone"),
        "Resource":f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
        "Units":r.get("unit_count",1),
        "From":r.get("origin_name"),
        "Dist km":r.get("distance_km",0),
        "ETA min":r.get("eta_minutes",0),
        "Note":r.get("eta_note",""),
        "Waypoints":len(r.get("waypoints",[])),
        "Status":"✓ OK" if r.get("success") else f'✗ {r.get("error","?")}',
    } for r in routes]), use_container_width=True, hide_index=True)

    st.markdown("**Interactive Route Map — Real OSM Waypoints**")
    st_folium(_folium_map(), width=None, height=480, key="fmap8", returned_objects=[])

    mp = gs.get("route_map_path")
    if mp and os.path.exists(mp):
        st.success(f"📄 Full HTML map saved: `{mp}`")

    _terminal(); st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("◀ BACK", key="b8"): st.session_state["stage"]=6; st.rerun()
    with c2:
        if st.button("🔄 Re-plan Routes", key="rp8"):
            _log("Re-planning routes …")
            st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🔄 Re-running route planner …"):
                try: _run_phase2(approved=True)
                except Exception as e: _log(f"[ERROR] {e}")
            st.rerun()
    with c3:
        if routes and st.button("▶ PROCEED TO APPROVAL #2", key="f8"):
            st.session_state["stage"] = 8; st.rerun()

# ============================================================================
#  STAGE 9 — ADMIN APPROVAL GATE 2  ← SECOND of TWO approval gates
# ============================================================================

def stage_9():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">✅ Stage 9: Admin Approval Gate #2 — Routes</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs     = _graph_state()
    routes = gs.get("route_plan", [])
    phase  = st.session_state.get("pipeline_phase", "idle")

    if not routes:
        st.warning("No route plan — go back to Stage 8.")
        _nav(back=7); return

    # Already approved
    if gs.get("route_approved") or phase in ("running_phase3","complete"):
        st.success("✅ Routes APPROVED — Communication Agent running / complete.")
        _nav(back=7, fwd=9, fwd_label="▶ VIEW DISPATCH COMMUNICATIONS")
        return

    st.markdown("**Review all planned routes before dispatching:**")
    st.dataframe(pd.DataFrame([{
        "Resource":f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
        "Units":r.get("unit_count",1),
        "Zone":r.get("zone"),
        "From":r.get("origin_name"),
        "Dist km":r.get("distance_km",0),
        "ETA min":r.get("eta_minutes",0),
        "Status":"✓ OK" if r.get("success") else "✗ FAILED",
    } for r in routes]), use_container_width=True, hide_index=True)

    mp = gs.get("route_map_path")
    if mp and os.path.exists(mp):
        st.info(f"📄 Route map: `{mp}`")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE ROUTES & DISPATCH", key="app2", use_container_width=True):
            _log("ADMIN ✓ Route plan APPROVED — resuming LangGraph (Phase 3) …")
            st.session_state["pipeline_phase"] = "running_phase3"
            with st.spinner("📡 Communication Agent generating Gemini dispatch instructions …"):
                try:
                    _run_phase3(approved=True)
                    st.success("✅ Dispatch instructions ready!"); st.balloons()
                except Exception as e:
                    _log(f"[ERROR] Phase 3:\n{traceback.format_exc()}")
                    st.error(f"Phase 3 error: {e}"); st.code(traceback.format_exc())
            st.rerun()
    with c2:
        if st.button("🔴 REJECT — RE-PLAN ROUTES", key="hold2", use_container_width=True):
            _log("ADMIN ✗ Routes REJECTED — re-running route planner …")
            st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🔄 Re-running route planner …"):
                try: _run_phase2(approved=True)
                except Exception as e: _log(f"[ERROR] {e}")
            st.rerun()
    _terminal()

# ============================================================================
#  STAGE 10 — COMMUNICATIONS  (honest SMS status)
# ============================================================================

def stage_10():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📡 Stage 10: Communication Agent</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    if st.session_state.get("pipeline_phase") == "running_phase3":
        st.info("📡 Communication Agent running … check terminal."); _terminal(); return

    gs       = _graph_state()
    dispatch = gs.get("dispatch_result", {})
    routes   = gs.get("route_plan", [])

    if not dispatch and not routes:
        st.warning("Dispatch data not available yet.")
        _terminal(); st.divider(); _nav(back=8); return

    # Pipeline complete banner
    if st.session_state.get("pipeline_phase") == "complete":
        st.markdown(
            f'<div style="background:{THEME["bg2"]};border:2px solid {THEME["green"]};'
            f'border-radius:8px;padding:16px;text-align:center;margin-bottom:16px;">'
            f'<span style="color:{THEME["green"]};font-family:\'Rajdhani\';font-size:24px;font-weight:bold;">'
            f'🎯 AEGIS PIPELINE COMPLETE</span><br>'
            f'<span style="color:{THEME["text"]};font-size:13px;">'
            f'All {len(STAGES)} stages executed via LangGraph master_graph</span></div>',
            unsafe_allow_html=True)

    # Dispatch instructions
    instructions = (dispatch or {}).get("instructions", {})
    st.markdown("**📋 Dispatch Instructions (Gemini LLM)**")
    if instructions:
        for z, instr in instructions.items():
            text = instr if isinstance(instr, str) else json.dumps(instr, indent=2)
            st.markdown(
                f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b><br>'
                f'<pre style="margin:8px 0 0;font-size:11px;color:{THEME["text"]};'
                f'white-space:pre-wrap;">{text}</pre></div>', unsafe_allow_html=True)
    else:
        for r in routes:
            em    = _remoji(r.get("resource_type",""))
            rtype = r.get("resource_type","").replace("_"," ").title()
            st.markdown(
                f'<div class="card"><b style="color:{THEME["cyan"]};">'
                f'{em} {r.get("unit_count",1)}× {rtype} → Zone {r.get("zone")}</b><br>'
                f'<span style="font-family:\'Share Tech Mono\';font-size:11px;">'
                f'From: {r.get("origin_name","?")} · {r.get("distance_km","?")} km · '
                f'ETA {r.get("eta_minutes","?")} min · {len(r.get("waypoints",[]))} waypoints'
                f'</span></div>', unsafe_allow_html=True)

    summary = (dispatch or {}).get("summary","")
    if summary:
        st.info(f"**Commander Summary:** {summary}")

    # ── Honest SMS status ─────────────────────────────────────────────────
    st.markdown("**📱 SMS Dispatch Status**")
    from dotenv import load_dotenv; load_dotenv()
    sms_results       = (dispatch or {}).get("sms_results", [])
    twilio_sid        = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token      = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from       = os.getenv("TWILIO_PHONE_NUMBER")
    twilio_to         = os.getenv("YOUR_PHONE_NUMBER")
    twilio_configured = all([twilio_sid, twilio_token, twilio_from, twilio_to])

    if sms_results:
        for res in sms_results:
            if res.get("success"):
                st.markdown(
                    f'<div class="card" style="border-color:{THEME["green"]};">'
                    f'✅ SMS sent → Zone <b>{res.get("zone","")}</b>  ·  '
                    f'SID: <code>{res.get("sid","")}</code></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="card" style="border-color:{THEME["red"]};">'
                    f'❌ SMS FAILED → Zone <b>{res.get("zone","")}</b>  ·  '
                    f'{res.get("error","unknown error")}</div>',
                    unsafe_allow_html=True)
    elif not twilio_configured:
        st.markdown(
            f'<div class="card-warn">'
            f'⚠️ <b>SMS was NOT sent</b> — Twilio credentials not configured.<br><br>'
            f'To enable real SMS dispatch, add these to your <code>.env</code> file:<br>'
            f'<code>TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx</code><br>'
            f'<code>TWILIO_AUTH_TOKEN=your_auth_token</code><br>'
            f'<code>TWILIO_PHONE_NUMBER=+1xxxxxxxxxx</code><br>'
            f'<code>YOUR_PHONE_NUMBER=+91xxxxxxxxxx</code><br><br>'
            f'Dispatch instructions above were generated — only SMS delivery is skipped.'
            f'</div>', unsafe_allow_html=True)

    # Audio files
    audio = (dispatch or {}).get("audio_files", [])
    if audio:
        st.markdown("**🔊 Audio Dispatch Files (gTTS)**")
        for fpath in audio:
            if os.path.exists(fpath):
                st.audio(fpath); st.caption(fpath)

    st.markdown(
        f'<div style="color:{THEME["green"]};font-family:\'Share Tech Mono\';font-size:13px;'
        f'font-weight:bold;">📡 {len(routes)} instruction(s) ready · LangGraph run COMPLETE</div>',
        unsafe_allow_html=True)

    _terminal(); st.divider()
    c1, c2, c3 = st.columns(3)
    with c2:
        if st.button("📡 CONFIRM ALL DISPATCHES SENT", use_container_width=True, key="send"):
            _log("All dispatches confirmed sent.")
            st.success("✅ All dispatches confirmed!"); st.balloons()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ BACK", key="b10"): st.session_state["stage"]=8; st.rerun()
    with c2:
        if st.button("🏁 COMPLETE & RESET", key="done10"):
            import uuid
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            st.balloons(); st.rerun()

# ============================================================================
#  MAIN
# ============================================================================

STAGE_FNS = [stage_1,stage_2,stage_3,stage_4,stage_5,
             stage_6,stage_7,stage_8,stage_9,stage_10]

def main():
    _reset_nav_counter()   # must be first — clears nav key counter each render
    st.session_state.setdefault("stage",0)
    st.session_state.setdefault("log","")
    st.session_state.setdefault("pipeline_phase","idle")
    st.session_state.setdefault("thread_id","aegis_main")
    _sidebar()
    st.markdown(
        f'<h1 style="color:{THEME["cyan"]};font-family:\'Rajdhani\';text-align:center;">'
        f'🛰️ AEGIS · Crisis Management AI</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{THEME["mono"]};text-align:center;font-family:\'Share Tech Mono\';">'
        f'Agentic Emergency Response &amp; Intelligence System  ·  '
        f'<b>LangGraph Master Agent</b></p>', unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p style="color:{THEME["text"]};font-family:\'Share Tech Mono\';font-size:12px;">'
                f'PIPELINE PROGRESS</p>', unsafe_allow_html=True)
    _stepper(); st.divider()
    STAGE_FNS[st.session_state.get("stage",0)]()

if __name__ == "__main__":
    main()