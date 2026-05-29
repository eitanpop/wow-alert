from wow_alert.config import REPO_ROOT
from wow_alert.tag_rules import TagRules, load_tag_rules


def _table() -> TagRules:
    return load_tag_rules(REPO_ROOT / "config")


def test_loads_real_table():
    t = _table()
    assert "interrupt" in t.tags
    assert "big_damage_party" in t.tags
    assert t.precedence[0] == "interrupt"


def test_interrupt_chain():
    pr = _table().priorities_for(["interrupt"])
    assert len(pr) == 1
    assert pr[0].category == "interrupt"


def test_big_damage_party_order_puts_wings_under_the_wall():
    # Party-wide DR first, then self-throughput (wings), then party heal.
    pr = _table().priorities_for(["big_damage_party"])
    assert [(p.category, p.scope) for p in pr] == [
        ("defensive", "party_wide"),
        ("heal", "self"),
        ("heal", "party_wide"),
        ("defensive", "self"),   # personal DR, last resort
    ]


def test_multi_tag_uses_precedence_order_not_input_order():
    t = _table()
    # big_damage_single (earlier in precedence) sorts ahead of big_damage_party
    # regardless of the order the tags are listed on the spell.
    a = t.priorities_for(["big_damage_party", "big_damage_single"])
    b = t.priorities_for(["big_damage_single", "big_damage_party"])
    assert [(p.category, p.scope) for p in a] == [(p.category, p.scope) for p in b]
    # big_damage_single leads, and its first step is the self defensive.
    assert a[0].target_is_self is True
    assert (a[0].category, a[0].scope) == ("defensive", "self")


def test_dodge_is_no_action():
    assert _table().priorities_for(["dodge"]) == []


def test_unknown_tag_contributes_nothing():
    assert _table().priorities_for(["not_a_real_tag"]) == []
