"""Step 7: turn the engine's decision into plain Hindi or English. The LLM never decides.

The text has three parts:
  headline  what to do and how much water  - always from a fixed template
  body      why (soil water, crop use, rain) - written by a local Ollama model, if it passes checks
  actions   what the farmer can do           - always from fixed templates (engine.farmer_actions codes)

So the action and the mm depth can never be changed by the model. The model's paragraph is
rejected (and the template paragraph used) if it uses a number the engine did not give, says
the opposite of the decision, is in the wrong script, or Ollama is not running.
"""
from __future__ import annotations

import json
import re

DEFAULT_MODEL = "gemma3:4b"
TIMEOUT_S = 60
MAX_BODY_CHARS = 900
ALWAYS_ALLOWED_NUMBERS = {3, 5, 7, 15}  # "next 5 days", "last 7 days", "3 weeks", "15 cm"
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
LANG_NAMES = {"en": "English", "hi": "Hindi"}

HEADLINES = {
    "en": {
        "irrigate_now": "Irrigate today: apply about {gross:.0f} mm of water{hours}.",
        "irrigate_soon": "Irrigate in about {days} day(s): plan for about {gross:.0f} mm of water{hours}.",
        "watch": ("Irrigation is not due by the water balance, but the satellite shows the crop losing "
                  "leaf water: check the soil by hand."),
        "wait": "No irrigation needed now. The next irrigation is due in about {days} days.",
        "wait_far": "No irrigation needed in the next 3 weeks.",
        "not_sown": "The crop is not sown yet: nothing to do.",
        "season_over": "The crop season is over: no irrigation advice.",
    },
    "hi": {
        "irrigate_now": "आज सिंचाई करें: लगभग {gross:.0f} मिमी पानी दें{hours}।",
        "irrigate_soon": "लगभग {days} दिन में सिंचाई करें: लगभग {gross:.0f} मिमी पानी की तैयारी रखें{hours}।",
        "watch": ("पानी के हिसाब से अभी सिंचाई का समय नहीं है, पर उपग्रह फसल की पत्तियों में पानी घटता "
                  "दिखा रहा है: हाथ से मिट्टी जाँचें।"),
        "wait": "अभी सिंचाई की ज़रूरत नहीं है। अगली सिंचाई लगभग {days} दिन बाद होगी।",
        "wait_far": "अगले 3 हफ़्तों तक सिंचाई की ज़रूरत नहीं है।",
        "not_sown": "फसल अभी बोई नहीं गई है: अभी कुछ करने की ज़रूरत नहीं।",
        "season_over": "फसल का मौसम पूरा हो गया है: सिंचाई की सलाह नहीं।",
    },
}
PUMP_HOURS = {"en": " ({h:g} hours of pumping)", "hi": " (पंप लगभग {h:g} घंटे चलाएँ)"}

FACTS = {
    "en": ("Soil water: {pct}% of what the soil can hold (the crop starts to suffer below {stress_pct}%). "
           "The {crop} is using about {use} mm of water per day. "
           "Rain: {r7:.0f} mm in the last 7 days, {r5:.0f} mm forecast for the next 5 days."),
    "hi": ("मिट्टी में पानी: क्षमता का {pct}% (इसके {stress_pct}% से नीचे जाने पर फसल पर दबाव शुरू होता है)। "
           "फसल रोज़ लगभग {use} मिमी पानी ले रही है। "
           "बारिश: पिछले 7 दिन में {r7:.0f} मिमी, अगले 5 दिन में {r5:.0f} मिमी का अनुमान।"),
}

ACTION_TEXT = {
    "apply_depth": {"en": "Apply about {gross_mm:.0f} mm{hours}.",
                    "hi": "लगभग {gross_mm:.0f} मिमी पानी दें{hours}।"},
    "irrigate_cool_hours": {"en": "Irrigate in the early morning or evening, when less water is lost to evaporation.",
                            "hi": "सुबह जल्दी या शाम को सिंचाई करें, तब कम पानी भाप बनकर उड़ता है।"},
    "rain_forecast_counted": {"en": "About {rain_mm} mm of rain is forecast and already counted. "
                                    "If much more falls, delay the irrigation.",
                              "hi": "लगभग {rain_mm} मिमी बारिश का अनुमान है, यह हिसाब में शामिल है। "
                                    "ज़्यादा बारिश हो जाए तो सिंचाई टाल दें।"},
    "hand_check": {"en": "Take soil from about 15 cm deep: if it crumbles and will not form a ball in your hand, irrigate.",
                   "hi": "लगभग 15 सेमी गहराई से मिट्टी लें: अगर वह मुट्ठी में गोला न बने और बिखर जाए, तो सिंचाई करें।"},
    "soil_nearly_full": {"en": "The soil is {water_pct}% full: more water now would be wasted and can harm the roots.",
                         "hi": "मिट्टी में {water_pct}% पानी भरा है: अभी और पानी देना बेकार होगा और जड़ों को नुकसान पहुँचा सकता है।"},
    "recheck_in": {"en": "Check the advice again in {days} days.",
                   "hi": "{days} दिन बाद सलाह फिर देखें।"},
    "trash_mulch": {"en": "Spread dry sugarcane trash between the rows as mulch: the soil stays moist for longer.",
                    "hi": "कतारों के बीच गन्ने की सूखी पत्तियाँ (पताई) बिछाएँ: इससे मिट्टी में नमी ज़्यादा दिन रहती है।"},
    "alternate_furrow": {"en": "Water every other furrow: it can save up to about a third of the water.",
                         "hi": "एक कूँड़ छोड़कर सिंचाई करें: इससे लगभग एक तिहाई तक पानी बच सकता है।"},
    "avoid_waterlogging": {"en": "Do not let water stand in the field for more than a day.",
                           "hi": "खेत में एक दिन से ज़्यादा पानी खड़ा न रहने दें।"},
    "log_irrigation": {"en": "Record every irrigation in the app, so the advice knows when the soil was last filled.",
                       "hi": "हर सिंचाई ऐप में दर्ज करें, ताकि सलाह को पता रहे कि मिट्टी आखिरी बार कब भरी गई।"},
}

# Phrases that would contradict the decision if the model wrote them.
SAYS_IRRIGATE = {"en": ["irrigate today", "irrigate now", "irrigate immediately"],
                 "hi": ["आज सिंचाई करें", "तुरंत सिंचाई", "अभी सिंचाई करें"]}
SAYS_DONT = {"en": ["no need to irrigate", "do not irrigate", "don't irrigate", "no irrigation needed",
                    "not need irrigation"],
             "hi": ["सिंचाई की ज़रूरत नहीं", "सिंचाई की जरूरत नहीं", "सिंचाई न करें", "सिंचाई ना करें"]}


def _hours(h, lang: str) -> str:
    return PUMP_HOURS[lang].format(h=h) if h else ""


def headline(decision: dict, lang: str = "hi") -> str:
    """The decision in one sentence: action, and mm / pump hours when irrigating."""
    action = decision["action"]
    depth = decision.get("irrigation_depth_mm") or {}
    days = decision.get("days_to_irrigation")
    key = "wait_far" if action == "wait" and days is None else action
    return HEADLINES[lang][key].format(gross=depth.get("gross_mm", 0.0), days=days,
                                       hours=_hours(depth.get("pump_hours"), lang))


def facts_line(decision: dict, lang: str = "hi") -> str:
    """Why: soil water, crop water use, rain. Empty before sowing / after the season."""
    if "water_available_pct" not in decision:
        return ""
    taw, raw = decision["root_zone_capacity_mm"], decision["allowed_depletion_mm"]
    use = decision.get("crop_use_mm_per_day")
    return FACTS[lang].format(pct=decision["water_available_pct"], stress_pct=round(100 * (1 - raw / taw)),
                              crop=decision["crop"], use="?" if use is None else use,
                              r7=decision["rain_last_7d_mm"], r5=decision["rain_next_5d_mm"])


def action_lines(decision: dict, lang: str = "hi") -> list[str]:
    """Each engine action code in the farmer's language."""
    lines = []
    for a in decision.get("actions", []):
        text = ACTION_TEXT.get(a["code"], {}).get(lang)
        if text:
            params = {k: v for k, v in a.items() if k != "code"}
            lines.append(text.format(**params, hours=_hours(a.get("pump_hours"), lang)))
    return lines


def allowed_numbers(decision: dict) -> set[float]:
    """Every number the model may mention: the engine's own, rounded the ways people round."""
    vals: list[float] = list(ALWAYS_ALLOWED_NUMBERS)
    depth = decision.get("irrigation_depth_mm") or {}
    for k in ("depletion_mm", "allowed_depletion_mm", "root_zone_capacity_mm", "days_to_irrigation",
              "crop_use_mm_per_day", "rain_last_7d_mm", "rain_next_5d_mm", "water_available_pct",
              "days_after_sowing", "root_depth_m", "soil_moisture_m3m3", "depletion_pct_of_allowed"):
        if isinstance(decision.get(k), (int, float)):
            vals.append(float(decision[k]))
    for k in ("gross_mm", "net_mm", "pump_hours", "water_m3"):
        if isinstance(depth.get(k), (int, float)):
            vals.append(float(depth[k]))
    if "water_available_pct" in decision:
        vals.append(100 - decision["water_available_pct"])
        vals.append(round(100 * (1 - decision["allowed_depletion_mm"] / decision["root_zone_capacity_mm"])))
    for a in decision.get("actions", []):
        vals.extend(float(v) for k, v in a.items() if k != "code" and isinstance(v, (int, float)))
    out = set()
    for v in vals:
        out.update({round(v), round(v, 1), round(v, 2), round(v, 3)})
    return out


def numbers_in(text: str) -> list[float]:
    """All numbers written in the text (Western or Devanagari digits)."""
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.translate(DEVANAGARI_DIGITS))]


def check_body(body: str, decision: dict, lang: str) -> str | None:
    """None if the model's paragraph is safe to show, else the reason it is not."""
    if not body.strip():
        return "empty answer"
    if len(body) > MAX_BODY_CHARS:
        return "answer too long"
    allowed = allowed_numbers(decision)
    for n in numbers_in(body):
        if not any(abs(n - a) < 1e-6 for a in allowed):
            return f"number {n:g} is not in the engine's result"
    low = body.lower()
    irrigating = decision["action"] in ("irrigate_now", "irrigate_soon")
    for phrase in (SAYS_DONT if irrigating else SAYS_IRRIGATE)[lang]:
        if phrase in low:
            return f"contradicts the decision ('{phrase}')"
    letters = [c for c in body if c.isalpha()]
    devanagari = sum("ऀ" <= c <= "ॿ" for c in letters)
    share = devanagari / len(letters) if letters else 0
    if (lang == "hi" and share < 0.6) or (lang == "en" and share > 0.1):
        return f"not written in {LANG_NAMES[lang]}"
    return None


def prompt_messages(decision: dict, lang: str) -> list[dict]:
    """Chat messages for the model: the decision is given, only the explanation is asked for."""
    name = LANG_NAMES[lang]
    system = (
        f"You explain irrigation advice to an Indian farmer in simple {name}. The decision is already made "
        "by a water-balance model and you must not change it or add a different one. "
        f"Write 3 to 4 short sentences in {name} explaining WHY, using only the facts given: soil water, "
        "crop water use, rain. Use only numbers that appear in the facts, written with digits 0-9. "
        "No dates, no headings, no lists, no markdown. Do not repeat the decision sentence."
    )
    facts = {
        "decision_sentence": headline(decision, lang),
        "facts": facts_line(decision, lang),
        "crop": decision.get("crop"), "growth_stage": decision.get("stage"),
        "days_to_irrigation": decision.get("days_to_irrigation"),
        "satellite_note": next((r for r in decision.get("reasons", []) if "Satellite" in r), None),
    }
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]


def ollama_chat(model: str, messages: list[dict]) -> str:
    """One call to the local Ollama server (raises if it is not running)."""
    import ollama

    client = ollama.Client(timeout=TIMEOUT_S)
    resp = client.chat(model=model, messages=messages, options={"temperature": 0.2})
    msg = resp["message"] if isinstance(resp, dict) else resp.message
    return (msg["content"] if isinstance(msg, dict) else msg.content).strip()


def explain(decision: dict, lang: str = "hi", use_llm: bool = True, model: str = DEFAULT_MODEL,
            chat=ollama_chat) -> dict:
    """Farmer-facing text for one decision. Works offline: without Ollama it is all templates."""
    if lang not in LANG_NAMES:
        raise ValueError(f"language must be one of {sorted(LANG_NAMES)}")
    head, template_body, actions = headline(decision, lang), facts_line(decision, lang), action_lines(decision, lang)
    body, source, why = template_body, "template", "LLM switched off" if not use_llm else None
    if use_llm and template_body:
        try:
            answer = chat(model, prompt_messages(decision, lang))
            why = check_body(answer, decision, lang)
            if why is None:
                body, source = answer, "llm"
        except Exception as e:  # noqa: BLE001  (Ollama not installed / not running / model missing)
            why = f"LLM unavailable: {e}"
    text = "\n\n".join(p for p in (head, body, "\n".join(f"- {a}" for a in actions)) if p)
    return {"lang": lang, "headline": head, "body": body, "actions": actions, "text": text,
            "source": source, "fallback_reason": why, "model": model if source == "llm" else None}


if __name__ == "__main__":
    import sys

    from src.pipeline import load_config, run_advice

    cfg = load_config()
    result = run_advice(cfg)
    for lang in sys.argv[1:] or ["hi", "en"]:
        out = explain(result["decision"], lang, model=cfg.get("llm_model", DEFAULT_MODEL))
        print(f"--- {LANG_NAMES[lang]} ({out['source']}{', ' + out['fallback_reason'] if out['fallback_reason'] else ''})")
        print(out["text"], "\n")
