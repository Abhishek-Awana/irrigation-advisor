import pytest

from src import explain

DECISION = {
    "crop": "sugarcane", "date": "2026-10-04", "action": "irrigate_now", "stage": "mid-season",
    "days_after_sowing": 172, "depletion_mm": 110.0, "allowed_depletion_mm": 104.1,
    "root_zone_capacity_mm": 160.1, "depletion_pct_of_allowed": 106, "days_to_irrigation": 0,
    "crop_use_mm_per_day": 4.8, "rain_last_7d_mm": 2.0, "rain_next_5d_mm": 0.0,
    "water_available_pct": 31, "root_depth_m": 1.2, "soil_moisture_m3m3": 0.256,
    "irrigation_depth_mm": {"needed": True, "net_mm": 110.0, "gross_mm": 157.1, "efficiency": 0.7,
                            "pump_hours": 6.5, "water_m3": 161.3},
    "reasons": [], "actions": [{"code": "apply_depth", "gross_mm": 157.1, "pump_hours": 6.5},
                               {"code": "irrigate_cool_hours"}, {"code": "trash_mulch"}],
}
WAIT = {**DECISION, "action": "wait", "days_to_irrigation": 12,
        "actions": [{"code": "recheck_in", "days": 7}, {"code": "log_irrigation"}]}


def fake_chat(answer):
    return lambda model, messages: answer


def test_headline_states_action_mm_and_pump_hours_in_both_languages():
    assert explain.headline(DECISION, "en") == "Irrigate today: apply about 157 mm of water (6.5 hours of pumping)."
    hi = explain.headline(DECISION, "hi")
    assert "आज सिंचाई करें" in hi and "157 मिमी" in hi and "6.5 घंटे" in hi


def test_wait_headlines():
    assert "12 days" in explain.headline(WAIT, "en")
    assert "3 weeks" in explain.headline({**WAIT, "days_to_irrigation": None}, "en")


def test_every_action_code_has_both_languages():
    for code, texts in explain.ACTION_TEXT.items():
        assert set(texts) == {"en", "hi"}, code


def test_engine_action_codes_all_have_text():
    import json
    from pathlib import Path

    codes = {"apply_depth", "irrigate_cool_hours", "rain_forecast_counted", "hand_check", "soil_nearly_full",
             "recheck_in", "log_irrigation"}
    for crop in json.loads(Path("config/crops.json").read_text()).values():
        codes.update(crop.get("water_saving_tips", []))
    assert codes <= set(explain.ACTION_TEXT)


def test_numbers_in_reads_devanagari_digits():
    assert explain.numbers_in("लगभग १५७ मिमी और 4.8") == [157.0, 4.8]


def test_check_accepts_a_faithful_answer():
    body = "The soil has only 31% of its water left and the cane uses 4.8 mm a day. No rain is expected."
    assert explain.check_body(body, DECISION, "en") is None


@pytest.mark.parametrize("body, reason", [
    ("Apply 200 mm of water.", "number 200"),
    ("There is no need to irrigate this week.", "contradicts"),
    ("मिट्टी में पानी कम है।", "not written in English"),
    ("", "empty"),
])
def test_check_rejects_unsafe_answers(body, reason):
    assert reason in explain.check_body(body, DECISION, "en")


def test_check_rejects_irrigate_today_when_engine_says_wait():
    assert "contradicts" in explain.check_body("आज सिंचाई करें, मिट्टी सूखी है।", WAIT, "hi")


def test_explain_uses_llm_body_when_it_passes():
    body = "मिट्टी में केवल 31% पानी बचा है और गन्ना रोज़ 4.8 मिमी पानी ले रहा है।"
    out = explain.explain(DECISION, "hi", chat=fake_chat(body))
    assert out["source"] == "llm" and out["body"] == body
    assert out["text"].startswith(explain.headline(DECISION, "hi"))
    assert "पताई" in out["text"]  # the actions are always the fixed templates


def test_explain_falls_back_on_invented_numbers():
    out = explain.explain(DECISION, "en", chat=fake_chat("You should apply 300 mm today."))
    assert out["source"] == "template" and "300" in out["fallback_reason"]
    assert "157 mm" in out["text"] and "300" not in out["text"]


def test_explain_works_offline():
    def broken(model, messages):
        raise ConnectionError("ollama not running")

    out = explain.explain(DECISION, "hi", chat=broken)
    assert out["source"] == "template" and "unavailable" in out["fallback_reason"]
    assert "157 मिमी" in out["text"]


def test_not_sown_has_no_facts_and_skips_the_llm():
    called = []
    d = {"crop": "wheat", "date": "2026-10-04", "action": "not_sown", "reasons": []}
    out = explain.explain(d, "en", chat=lambda m, msgs: called.append(1) or "x")
    assert out["text"] == "The crop is not sown yet: nothing to do." and not called


def test_prompt_contains_the_decision_but_asks_only_for_why():
    msgs = explain.prompt_messages(DECISION, "hi")
    assert "must not change" in msgs[0]["content"] and "Hindi" in msgs[0]["content"]
    assert "157" in msgs[1]["content"]
