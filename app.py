"""Step 8: farmer UI.   streamlit run app.py

Today's advice (Hindi / English), how much water and how long to pump, field maps compared
with the neighbouring fields, the irrigation log, the satellite trend and the validation report.
All decisions come from src/engine.py; the LLM only words them (src/explain.py).
"""
from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely.geometry import mapping

from src import engine, explain, irrigation_log, maps, pipeline

VALIDATION_JSON = Path("data/validation.json")
VALIDATION_PNG = Path("data/validation.png")

T = {
    "en": {
        "title": "Field Water Advisor", "today": "Today", "maps": "Field maps", "log": "Irrigation log",
        "trend": "Satellite trend", "validation": "Validation", "details": "Details",
        "water_left": "Water in root zone", "irrigate_in": "Irrigate in", "days": "days", "apply": "Water to apply",
        "pump": "Pump time", "use": "Crop water use", "set_pump": "set pump flow in the sidebar",
        "settings": "Field settings", "save": "Save settings", "refresh": "Refresh data",
        "use_llm": "Explain with local AI (Ollama)", "language": "Language",
    },
    "hi": {
        "title": "खेत जल सलाहकार", "today": "आज", "maps": "खेत के नक्शे", "log": "सिंचाई रिकॉर्ड",
        "trend": "उपग्रह रुझान", "validation": "जाँच", "details": "विवरण",
        "water_left": "जड़ क्षेत्र में पानी", "irrigate_in": "सिंचाई", "days": "दिन में", "apply": "देने वाला पानी",
        "pump": "पंप समय", "use": "फसल का पानी", "set_pump": "साइडबार में पंप का बहाव भरें",
        "settings": "खेत की जानकारी", "save": "सेव करें", "refresh": "डेटा ताज़ा करें",
        "use_llm": "लोकल AI (Ollama) से समझाएँ", "language": "भाषा",
    },
}
MAP_NAMES = {
    "ndvi": {"en": "Greenness (NDVI)", "hi": "हरियाली (NDVI)"},
    "ndmi": {"en": "Leaf water (NDMI)", "hi": "पत्तियों में पानी (NDMI)"},
    "water_use": {"en": "Crop water use", "hi": "फसल का पानी उपयोग"},
    "ndmi_change": {"en": "Leaf water change", "hi": "पत्तियों के पानी में बदलाव"},
    "thermal": {"en": "Surface temperature vs neighbours", "hi": "पड़ोसी खेतों से तापमान की तुलना"},
}
STATUS = {"irrigate_now": st.error, "irrigate_soon": st.warning, "watch": st.warning, "wait": st.success}
ICONS = {"irrigate_now": "🚿", "irrigate_soon": "⏳", "watch": "👀", "wait": "✅"}


# ---------- cached data ----------

@st.cache_data(ttl=3 * 3600, show_spinner="Fetching weather, soil and satellite data...")
def get_advice(cfg_json: str, log_version: float, day: str) -> dict:
    cfg = json.loads(cfg_json)
    result = pipeline.run_advice(cfg, today=day)
    result["geom_geojson"] = mapping(result.pop("geom")) if result.get("geom") is not None else None
    return result


@st.cache_data(ttl=6 * 3600, show_spinner="Building field maps from Sentinel-2 and Landsat...")
def get_maps(series: pd.DataFrame | None, et0_mm: float | None, kc_max: float) -> dict:
    from src.satellite import load_field_geometry

    if series is None:
        return {"maps": {}, "grids": {}, "errors": ["no satellite time series"]}
    return maps.build_maps(load_field_geometry(), series, et0_mm, kc_max)


@st.cache_data(ttl=6 * 3600, show_spinner="Writing the explanation...")
def get_explanation(decision_json: str, lang: str, use_llm: bool, model: str) -> dict:
    return explain.explain(json.loads(decision_json), lang, use_llm=use_llm, model=model)


def log_version() -> float:
    p = irrigation_log.LOG_PATH
    return p.stat().st_mtime if p.exists() else 0.0


# ---------- sidebar ----------

cfg = pipeline.load_config()
st.set_page_config(page_title="Field Water Advisor", page_icon="💧", layout="wide")

with st.sidebar:
    lang = "hi" if st.radio("भाषा / Language", ["हिन्दी", "English"],
                            index=0 if cfg.get("language", "").lower() == "hindi" else 1) == "हिन्दी" else "en"
    t = T[lang]
    use_llm = st.toggle(t["use_llm"], value=True)
    with st.expander(t["settings"]):
        crops = list(json.loads(engine.CROPS_PATH.read_text()))
        crop = st.selectbox("Crop / फसल", crops, index=crops.index(cfg["crop"]))
        sowing = st.date_input("Sowing date / बुवाई की तारीख", value=date.fromisoformat(cfg["sowing_date"]))
        flow = st.number_input("Pump flow, litres per second / पंप बहाव (लीटर/सेकंड)", min_value=0.0,
                               value=float(cfg.get("pump_flow_lps") or 0.0), step=0.5,
                               help="A 5 HP diesel pump typically gives 6-10 L/s. Time it filling a drum to measure.")
        if st.button(t["save"]):
            cfg.update(crop=crop, sowing_date=str(sowing), pump_flow_lps=flow or None)
            pipeline.save_config(cfg)
            st.cache_data.clear()
            st.rerun()
    if st.button(t["refresh"]):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"{cfg.get('farmer_name', '')} · {cfg['crop']} · {cfg['latitude']:.4f}, {cfg['longitude']:.4f}")

t = T[lang]
st.title(f"💧 {t['title']}")

try:
    result = get_advice(json.dumps(cfg, sort_keys=True), log_version(), str(date.today()))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not compute the advice: {e}")
    st.stop()
d = result["decision"]
depth = d.get("irrigation_depth_mm") or {}

tabs = st.tabs([t["today"], t["maps"], t["log"], t["trend"], t["validation"], t["details"]])

# ---------- Today ----------
with tabs[0]:
    text = get_explanation(json.dumps(d, default=str), lang, use_llm, cfg.get("llm_model", explain.DEFAULT_MODEL))
    STATUS.get(d["action"], st.info)(f"### {ICONS.get(d['action'], 'ℹ️')} {text['headline']}")

    if "water_available_pct" in d:
        c = st.columns(5)
        c[0].metric(t["water_left"], f"{d['water_available_pct']}%",
                    help=f"{d['soil_moisture_m3m3']} m3/m3; field capacity {d['field_capacity_m3m3']}, "
                         f"wilting point {d['wilting_point_m3m3']}")
        c[1].metric(t["irrigate_in"], "—" if d["days_to_irrigation"] is None
                    else f"{d['days_to_irrigation']} {t['days']}")
        irrigating = d["action"] in ("irrigate_now", "irrigate_soon") and depth.get("needed")
        c[2].metric(t["apply"], f"{depth.get('gross_mm', 0):.0f} mm" if irrigating else "—",
                    help=("" if irrigating else f"Not needed now. A refill today would take {depth.get('gross_mm', 0):.0f} mm; ")
                         + f"net {depth.get('net_mm', 0)} mm reaches the roots; "
                         f"{depth.get('efficiency', 0):.0%} application efficiency"
                         + (f"; {depth['water_m3']} m3 for the field" if depth.get("water_m3") else ""))
        c[3].metric(t["pump"], f"{depth['pump_hours']} h" if irrigating and depth.get("pump_hours") else "—",
                    help=None if depth.get("pump_hours") else t["set_pump"])
        c[4].metric(t["use"], f"{d['crop_use_mm_per_day']} mm/day" if d.get("crop_use_mm_per_day") else "—")
        st.progress(max(0, min(100, d["water_available_pct"])) / 100)

    st.write(text["body"])
    for line in text["actions"]:
        st.markdown(f"- {line}")
    src = (f"Explanation written by {text['model']} and checked against the engine's numbers."
           if text["source"] == "llm" else f"Explanation from fixed templates ({text['fallback_reason']}).")
    st.caption(f"{src} Confidence: {d.get('confidence')}.")
    if d.get("confidence_notes") or result["errors"]:
        with st.expander("Why this confidence / data problems"):
            for n in d.get("confidence_notes", []) + result["errors"]:
                st.markdown(f"- {n}")

# ---------- Maps ----------
with tabs[1]:
    et0 = pipeline.today_et0_mm(result["weather"])
    kc_max = engine.load_crop(cfg["crop"])["kc"]["mid"]
    built = get_maps(result["series"], et0, kc_max)
    if not built["maps"]:
        st.info("No maps yet: " + "; ".join(built["errors"]))
    else:
        names = [k for k in MAP_NAMES if k in built["maps"]]
        pick = st.radio("Map", names, format_func=lambda k: MAP_NAMES[k][lang], horizontal=True)
        m = built["maps"][pick]
        left, right = st.columns([1, 1])
        left.image(m["path"], width="stretch")
        left.caption(m["caption"])

        grid = built["grids"]["landsat" if pick == "thermal" else "s2"]
        ov = m["overlay"]
        png = maps.overlay_png(ov["values"], ov["valid"], ov["ramp"], ov["vmin"], ov["vmax"])
        w, s, e, n = maps.bounds_lonlat(grid, ov["values"].shape)
        layers = [pdk.Layer("BitmapLayer", image="data:image/png;base64," + base64.b64encode(png).decode(),
                            bounds=[w, s, e, n], opacity=0.85)]
        if result.get("geom_geojson"):
            layers.append(pdk.Layer("GeoJsonLayer", {"type": "Feature", "geometry": result["geom_geojson"]},
                                    stroked=True, filled=False, get_line_color=[11, 11, 11], line_width_min_pixels=2))
        right.pydeck_chart(pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                                    initial_view_state=pdk.ViewState(latitude=cfg["latitude"],
                                                                     longitude=cfg["longitude"],
                                                                     zoom=15 if pick != "thermal" else 13.5)))
        right.caption("Same map on a street map: zoom and pan to see which neighbouring fields differ.")

        st.subheader("Your field vs nearby crops" if lang == "en" else "आपका खेत बनाम पड़ोसी फसलें")
        rows = [{"map": MAP_NAMES[k][lang], "date": built["maps"][k]["date"],
                 "your field": built["maps"][k]["stats"]["field"], "nearby crops": built["maps"][k]["stats"]["around"],
                 "difference": built["maps"][k]["stats"]["diff"],
                 "clear view of field": f"{built['maps'][k]['stats']['clear_field_fraction']:.0%}"}
                for k in names]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for err in built["errors"]:
        st.caption(f"⚠️ {err}")

# ---------- Irrigation log ----------
with tabs[2]:
    st.write("Record every irrigation. This is the one thing satellites cannot see, and it keeps the advice right."
             if lang == "en" else "हर सिंचाई दर्ज करें। यह उपग्रह नहीं देख सकता, और इसी से सलाह सही रहती है।")
    with st.form("log", clear_on_submit=True):
        c = st.columns(3)
        day = c[0].date_input("Date / तारीख", value=date.today(), max_value=date.today())
        how = c[1].radio("Amount / मात्रा", ["Don't know (full flooding)", "Hours of pumping", "Depth in mm"])
        amount = c[2].number_input("Hours or mm", min_value=0.0, step=0.5)
        note = st.text_input("Note / टिप्पणी")
        if st.form_submit_button("Add / जोड़ें"):
            irrigation_log.add_entry(day, depth_mm=amount if how.startswith("Depth") and amount else None,
                                     hours=amount if how.startswith("Hours") and amount else None, note=note)
            st.cache_data.clear()
            st.rerun()
    log = irrigation_log.load_log()
    if log.empty:
        st.info("No irrigations recorded yet." if lang == "en" else "अभी कोई सिंचाई दर्ज नहीं है।")
    else:
        st.dataframe(log.assign(date=log["date"].dt.date), hide_index=True, width="stretch")
        gone = st.selectbox("Delete an entry / हटाएँ", [None] + list(log["date"].dt.date))
        if gone and st.button("Delete / हटाएँ"):
            irrigation_log.remove_entry(gone)
            st.cache_data.clear()
            st.rerun()

# ---------- Satellite trend ----------
with tabs[3]:
    series = result["series"]
    if series is None or series.empty:
        st.info("No satellite series available.")
    else:
        ok = series[series["usable"].astype(bool)].set_index("date")[["ndvi", "ndmi"]]
        st.markdown("**NDVI (greenness) and NDMI (leaf water), clear Sentinel-2 views only**")
        st.line_chart(ok.rename(columns={"ndvi": "NDVI", "ndmi": "NDMI"}))
        st.caption(f"{len(ok)} clear views out of {len(series)} passes; the rest were cloudy over the field.")
    wx = result["weather"]
    past = wx[~wx["is_forecast"].astype(bool)].set_index("date")
    st.markdown("**Rain, mm per day**")
    st.bar_chart(past["rain_mm"])

# ---------- Validation ----------
with tabs[4]:
    st.write("Does the water balance move like an independent soil-moisture dataset (ERA5-Land)? "
             "Model vs model, not ground truth.")
    if st.button("Run validation now (about a minute)"):
        from src.validate import validate_field

        with st.spinner("Replaying the last 90 days..."):
            validate_field(cfg)
    if VALIDATION_JSON.exists():
        rep = json.loads(VALIDATION_JSON.read_text())
        st.caption(f"{rep['from']} to {rep['to']} · soil {rep['soil'].get('texture_class')}")
        rows = [{"layer": k, "passed": "✅" if v["passed"] else "❌", "anomaly r": v.get("anomaly_r"),
                 "needs r": v.get("needs_anomaly_r"), "wet hit": v.get("wet_event_hit_rate"),
                 "dry hit": v.get("dry_event_hit_rate"), "needs hit": v.get("needs_event_hit_rate")}
                for k, v in rep["verdict"]["checks"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if VALIDATION_PNG.exists():
            st.image(str(VALIDATION_PNG), width="stretch")
        st.caption(rep["caveat"])
    else:
        st.info("No validation report yet.")

# ---------- Details ----------
with tabs[5]:
    st.json(d)
    if result["soil"]:
        st.json(result["soil"])
