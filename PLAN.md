# Field Water Advisor - build plan

How we work is in `.clinerules`. Do ONE step at a time; stop when its "done when" check passes.

Field: `config/field.json` + `config/field.geojson` (Muzaffarnagar area, UP) - crop: sugarcane, sown 2026-04-15.

## Step 1 - project setup [DONE]
Streamlit skeleton, `config/field.json`, pytest, requirements.
Done when: `pytest` passes and the config loads.

## Step 2 - weather: rain + ET0 [DONE]
`src/weather.py`. Open-Meteo daily `rain_mm`, `et0_mm`, `tmax_c`, `tmin_c`, `is_forecast`;
3-hour disk cache; offline fallback to the stale cache.
Done when: one row per day, and a failed download falls back to cache.

## Step 3 - satellite crop / leaf-water signal [DONE]
- 3A `src/satellite.py` - find Sentinel-2 L2A scenes over the field (STAC).
- 3B `src/bands.py` - read B04/B08/B11/SCL pixels inside the field polygon.
- 3C `src/indices.py` - reflectance -> NDVI (greenness), NDMI (leaf water).
- 3D `src/timeseries.py` - cached series, plot, latest-reading summary.
Done when: `data/satellite_series.csv` holds a usable NDVI/NDMI row per clear scene.

## Step 4 - irrigation decision engine [DONE]
`src/engine.py`. FAO-56 single-Kc daily water balance; Kc and root-depth curves from
`config/crops.json`; `Dr` kept inside [0, TAW]; irrigate when `Dr` reaches `RAW`.
Actions: `irrigate_now` / `irrigate_soon` / `watch` / `wait` / `not_sown` / `season_over`.
Done when: tests cover dry-down, rain reset, forecast delay and the satellite watch flag.

## Step 4B - soil moisture + HOW MUCH to irrigate  [DONE]
`decide()` must also report the root-zone soil moisture in m3/m3 and as % of available
water, and the irrigation depth in mm (net = refill root zone to field capacity,
gross = net / application efficiency).
Done when: `decide()` returns `soil_moisture_m3m3` and `irrigation_depth_mm`, with tests.

## Step 5 - field soil from SoilGrids (replaces the "assumed loam" guess)  [DONE]
`src/soil.py`: clay/sand/silt/OC/bulk density at the field point (ISRIC SoilGrids, no key)
-> USDA texture class -> Saxton & Rawls (2006) -> theta_fc, theta_wp, TAW mm/m for THIS field.
Done when: the field's texture class and TAW come from data, and `decide()` accepts them.

## Step 6 - validation against an independent dataset  [RUNS; verdict FAILS on 7-28 cm]
`src/validate.py`: replay the water balance over the last ~90 days on ERA5-Land rain/ET0
(`weather.fetch_weather_archive`) and compare the modelled soil-water *changes* with
ERA5-Land soil moisture (`weather.fetch_soil_moisture`, 0-7 cm and 7-28 cm).
Metrics: anomaly correlation, anomaly RMSE, level bias, wet/dry event hit rate.
Writes `data/validation.json` + `data/validation.png`.
Done when: the report runs end to end and gives a pass/fail against the thresholds.

## Step 7 - LLM explainer (never the decider)  [DONE]
`src/explain.py`: headline (action + mm + pump hours) and action list are fixed templates in
Hindi + English; a local Ollama model (`llm_model` in field.json, default gemma3:4b) writes only
the "why" paragraph. It is rejected if it uses a number the engine did not give, contradicts the
action, or is in the wrong script -> template paragraph instead (also the offline path).
Engine side: `engine.farmer_actions()` codes (apply depth, cool hours, hand check, mulch,
alternate furrow, log irrigation...) and `engine.pump_hours()` from field area + pump flow.
Done when: the text states the same action and mm depth as `decide()`, and works offline.

## Step 8 - farmer UI  [DONE]
`app.py`: Hindi/English; tabs Today (advice, mm, pump hours, actions), Field maps, Irrigation log,
Satellite trend, Validation, Details. `src/pipeline.py` runs the whole flow; each data source can
fail without stopping the advice. `src/irrigation_log.py` keeps data/irrigation_log.csv.
Done when: `streamlit run app.py` shows the advice without errors.

## Step 9 - field maps, field vs neighbours  [DONE]
`src/maps.py` (IrriWatch-style idea, NASA Spinoff "Farmers Get Tools from Space", free data only):
Sentinel-2 NDVI, NDMI, NDMI change between two clear dates, crop water use = Kc(NDVI) x ET0;
Landsat 8/9 surface temperature minus the median of nearby crops. Each map: field outline,
~300 m (thermal ~1 km) of neighbours, field-vs-nearby numbers, PNG + web-map overlay.
Limits: Landsat thermal is ~100 m native, larger than this field -> neighbourhood comparison only.
Maps inform; they do not change `decide()`.

## Next
- Dry-season validation run (Nov-Mar) and a hand-feel / probe check in the field.
- Confirm sugarcane Kc, stages and root depth with UPCSR / KVK, then set `verified: true`.
- Sentinel-1 radar (`sentinel-1-rtc`, no key) for surface soil moisture through monsoon cloud.
- Use Kc from NDVI in the engine (today only in the water-use map) once checked against the log.
- Delivery to phones: WhatsApp / SMS / voice in Hindi.

## Validation notes - first live run (2026-07-06 to 2026-10-04, Muzaffarnagar field)
- 0-7 cm: PASS - anomaly r 0.72, wet hit 0.68, dry hit 1.00.
- 7-28 cm: FAIL - anomaly r 0.40, wet hit 0.53, dry hit 0.92. The bucket is capped at field
  capacity (0.325 m3/m3, soil averaged over the 0-120 cm root zone) and ERA5-Land sits above that on 63 of the 91 monsoon days, so the
  model cannot follow the wet side.
- Drying rate (model 5.2 mm/day vs observed 1.3-2.2) is reported but NOT scored: a root-zone
  bucket and a thin surface layer are not the same thing.
- Model-vs-model only. Real ground truth needs a probe in the field, or ISMN / SMAP (account).
- Next: add percolation / drainage above field capacity, then re-run. Do not tune the
  thresholds to make this pass; re-check in the dry season (Nov-Mar) when the field is
  actually irrigation-limited.
