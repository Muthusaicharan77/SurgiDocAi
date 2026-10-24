from surgidoc.agent.claims import Claim
from surgidoc.agent.facts import FrameFacts, StructureFacts, validated_classes
from surgidoc.agent.perception import contiguous_runs
from surgidoc.agent.timeline import build_timeline
from surgidoc.agent.verify import mentioned_structures, verify


def frame(i, phase="calot_triangle_dissection", t=None, **areas):
    return FrameFacts(
        frame_id=i, frame_number=i * 5, time_s=i * 0.2 if t is None else t, source=f"f{i}.png",
        phase=phase, phase_confidence=0.9,
        structures={k: StructureFacts(area=v, components=1) for k, v in areas.items()},
    )


def scene():
    """Frames 0-9: grasper, hook and gallbladder visible; blood in frames 6-8."""
    facts = []
    for i in range(10):
        areas = {"grasper": 0.03, "l_hook_electrocautery": 0.02, "gallbladder": 0.2}
        if 6 <= i <= 8:
            areas["blood"] = 0.02
        facts.append(frame(i, **areas))
    return facts, build_timeline(facts)


def check(claim, passages=frozenset({"TG18-12"})):
    facts, timeline = scene()
    return verify([claim], facts, timeline, set(passages)).results[0]


def test_validated_classes_come_from_results():
    v = validated_classes()
    assert "gallbladder" in v and "grasper" in v
    assert not v & {"cystic_duct", "blood", "background"}


def test_supported_observation_with_validated_classes():
    r = check(Claim(id="1", claim_type="structures_visible",
                    text="Calot triangle dissection with the L-hook; gallbladder held by a grasper",
                    frame_ids=list(range(10)),
                    structures=["l_hook_electrocautery", "gallbladder", "grasper"]))
    assert (r.status, r.basis) == ("supported", "validated_model")


def test_bleeding_event_is_supported_but_unvalidated():
    _, timeline = scene()
    bleed = [e for e in timeline.events if e.kind == "possible_bleeding"]
    assert len(bleed) == 1 and not bleed[0].validated
    r = check(Claim(id="2", claim_type="event", text="Possible bleeding during dissection",
                    event_ids=[bleed[0].id], structures=["blood"]))
    assert (r.status, r.basis) == ("supported", "unvalidated_model")


def test_guideline_claim_needs_a_retrieved_passage():
    ok = check(Claim(id="3", claim_type="guideline",
                     text="The critical view of safety should be confirmed before clipping; "
                          "two structures, the cystic duct and artery, enter the gallbladder",
                     passage_ids=["TG18-12"]))
    assert (ok.status, ok.basis) == ("supported", "retrieved_source")
    made_up = check(Claim(id="3b", claim_type="guideline", text="...", passage_ids=["TG18-99"]))
    assert made_up.status == "invalid"


def test_guideline_claim_cannot_carry_observations():
    r = check(Claim(id="3c", claim_type="guideline", text="CVS achieved", passage_ids=["TG18-12"],
                    frame_ids=[5]))
    assert r.status == "invalid"


def test_claim_about_undetected_structure_is_unsupported():
    r = check(Claim(id="4", claim_type="structures_visible", text="Cystic duct clearly exposed",
                    frame_ids=[5], structures=["cystic_duct"]))
    assert r.status == "unsupported"
    assert "cystic_duct visible in 0/1" in r.reasons[0]


def test_text_naming_unlisted_structure_is_invalid():
    r = check(Claim(id="5", claim_type="structures_visible", text="Grasper retracts the cystic duct",
                    frame_ids=[5], structures=["grasper"]))
    assert r.status == "invalid"
    assert "cystic_duct" in r.reasons[0]


def test_hallucinated_frame_ids_are_invalid():
    r = check(Claim(id="6", claim_type="structures_visible", text="Grasper visible",
                    frame_ids=[99], structures=["grasper"]))
    assert r.status == "invalid"


def test_phase_claim_needs_majority_of_frames_and_is_unvalidated():
    ok = check(Claim(id="7", claim_type="phase", text="Calot triangle dissection",
                     frame_ids=list(range(10)), phase="calot_triangle_dissection"))
    assert (ok.status, ok.basis) == ("supported", "unvalidated_model")
    wrong = check(Claim(id="8", claim_type="phase", text="Clipping and cutting",
                        frame_ids=list(range(10)), phase="clipping_cutting"))
    assert wrong.status == "unsupported"


def test_limitation_passes_without_evidence():
    r = check(Claim(id="9", claim_type="limitation",
                    text="The system cannot verify whether the cystic duct was exposed"))
    assert (r.status, r.basis) == ("supported", "none")


def test_mentioned_structures_ignores_phase_names_and_prefers_longest_match():
    assert mentioned_structures("During gallbladder dissection the liver ligament") == {
        "liver_ligament"}
    assert mentioned_structures("Hook near the liver bed") == {"l_hook_electrocautery", "liver"}


def test_timeline_splits_segments_at_phase_changes_and_gaps():
    facts = [frame(0, "preparation", t=0.0), frame(1, "preparation", t=0.2),
             frame(2, "calot_triangle_dissection", t=0.4),
             frame(3, "calot_triangle_dissection", t=60.0)]  # 1 minute later
    tl = build_timeline(facts)
    assert [(s.phase, s.start_frame, s.end_frame) for s in tl.segments] == [
        ("preparation", 0, 1), ("calot_triangle_dissection", 2, 2),
        ("calot_triangle_dissection", 3, 3)]
    assert tl.gaps == [(2, 3)]


def test_contiguous_runs():
    assert contiguous_runs([0, 1, 2, 10, 11], max_gap_s=3) == [range(3), range(3, 5)]
