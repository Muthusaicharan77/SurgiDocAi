import json
from pathlib import Path

import numpy as np
from PIL import Image

from surgidoc.agent.claims import Claim
from surgidoc.agent.facts import FrameFacts, StructureFacts
from surgidoc.agent.timeline import build_timeline
from surgidoc.evals.grounding import grade, ground_truth_facts, summarise
from surgidoc.evals.retrieval import DEFAULT_QUESTIONS, score

GRASPER, LIVER, UNLABELED = 31, 21, 255  # watershed grey values


def write_frame(tmp_path: Path, i: int, mask: np.ndarray) -> str:
    image = tmp_path / f"frame_{i}_endo.png"
    Image.fromarray(np.zeros((*mask.shape, 3), np.uint8)).save(image)
    Image.fromarray(np.stack([mask] * 3, -1)).save(tmp_path / f"frame_{i}_endo_watershed_mask.png")
    return str(image)


def test_ground_truth_facts_read_the_mask_and_skip_unlabeled(tmp_path):
    mask = np.full((10, 10), LIVER, np.uint8)
    mask[:2] = GRASPER
    mask[9, 9] = UNLABELED
    f = FrameFacts(frame_id=0, frame_number=0, time_s=0.0, source=write_frame(tmp_path, 0, mask),
                   phase="preparation", phase_confidence=0.9,
                   structures={"gallbladder": StructureFacts(area=0.5, components=1)})
    gt = ground_truth_facts([f])[0]
    assert set(gt.structures) == {"grasper", "liver"}
    assert gt.structures["grasper"].area == 0.2 and gt.structures["liver"].area == 0.79
    assert gt.phase == "preparation"


def test_claim_the_model_supports_but_the_masks_contradict(tmp_path):
    # The model sees a grasper in both frames; the masks have only liver.
    facts, gt = [], []
    for i in range(2):
        mask = np.full((10, 10), LIVER, np.uint8)
        f = FrameFacts(frame_id=i, frame_number=i * 5, time_s=i * 0.2,
                       source=write_frame(tmp_path, i, mask), phase="preparation",
                       phase_confidence=0.9,
                       structures={"grasper": StructureFacts(area=0.1, components=1),
                                   "liver": StructureFacts(area=0.9, components=1)})
        facts.append(f)
    gt = ground_truth_facts(facts)
    claims = [
        Claim(id="c1", claim_type="structures_visible", text="The liver is visible",
              frame_ids=[0, 1], structures=["liver"]),
        Claim(id="c2", claim_type="structures_visible", text="A grasper is visible",
              frame_ids=[0, 1], structures=["grasper"]),
        Claim(id="c3", claim_type="phase", text="Preparation", frame_ids=[0, 1],
              phase="preparation"),
    ]
    rows = grade(claims, facts, gt, build_timeline(facts), {})
    assert [r["gradable"] for r in rows] == [True, True, False]
    s = summarise(rows)
    assert (s["graded"], s["grounded"], s["contradicted"]) == (2, 1, 1)
    assert s["passed_verifier_but_contradicted"] == 1
    assert s["contradicted_structures"] == {"grasper": 1}


def test_named_but_unlisted_structure_is_still_graded(tmp_path):
    mask = np.full((10, 10), LIVER, np.uint8)
    f = FrameFacts(frame_id=0, frame_number=0, time_s=0.0, source=write_frame(tmp_path, 0, mask),
                   phase="preparation", phase_confidence=0.9,
                   structures={"grasper": StructureFacts(area=0.1, components=1)})
    timeline = build_timeline([f])
    claim = Claim(id="c1", claim_type="event", text="A grasper is in view",
                  event_ids=[timeline.events[0].id])
    [row] = grade([claim], [f], ground_truth_facts([f]), timeline, {})
    assert row["model_status"] == "invalid"  # verifier: grasper named but not listed
    assert (row["gradable"], row["gt_status"]) == (True, "unsupported")  # masks: no grasper


def test_retrieval_scores():
    s = score(["a", "b", "c", "d"], {"c", "x"})
    assert s["mrr"] == 1 / 3
    assert (s["hit@1"], s["hit@3"], s["recall@3"], s["recall@5"]) == (0.0, 1.0, 0.5, 0.5)
    assert score(["a"], {"x"})["mrr"] == 0.0


def test_retrieval_labels_exist_in_the_corpus():
    from surgidoc.retrieval.chunking import load_corpus

    ids = {p.id for p in load_corpus()}
    questions = json.loads(DEFAULT_QUESTIONS.read_text())["questions"]
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert q["relevant"] and set(q["relevant"]) <= ids, q["id"]


def test_injected_errors_are_rejected_and_controls_pass(tmp_path):
    from surgidoc.evals.injected import run_eval

    # 20 frames: grasper and liver throughout, hook in 10-19. The masks agree except
    # in frames 3-4, where the model also sees fat.
    facts = []
    for i in range(20):
        mask = np.full((10, 10), LIVER, np.uint8)
        mask[:2] = GRASPER
        areas = {"grasper": 0.2, "liver": 0.8}
        if i >= 10:
            areas["l_hook_electrocautery"] = 0.05
        if i in (3, 4):
            areas["fat"] = 0.1
        facts.append(FrameFacts(
            frame_id=i, frame_number=i * 5, time_s=i * 0.2, source=write_frame(tmp_path, i, mask),
            phase="preparation" if i < 10 else "calot_triangle_dissection", phase_confidence=0.9,
            structures={k: StructureFacts(area=v, components=1) for k, v in areas.items()}))
    passages = {f"P{k}": " ".join(f"word{k}_{j}" for j in range(40)) for k in range(4)}
    corpus = {**passages, "OTHER": " ".join(f"other{j}" for j in range(40))}
    rows = run_eval(facts, ground_truth_facts(facts), build_timeline(facts), passages, corpus,
                    seed=0)
    categories = {r["category"] for r in rows}
    assert {"absent_structure", "wrong_phase", "altered_quote", "vision_error",
            "true_structure"} <= categories
    assert all(r["correct"] for r in rows), [r for r in rows if not r["correct"]][:1]


def test_selection_and_tool_metrics():
    from surgidoc.evals.tool_use import selection_metrics, tool_metrics

    facts = [FrameFacts(frame_id=i, frame_number=i * 5, time_s=i * 0.2, source=f"f{i}.png",
                        phase="preparation" if i < 5 else "calot_triangle_dissection",
                        phase_confidence=0.9,
                        structures={"grasper": StructureFacts(area=0.1, components=1),
                                    **({"l_hook_electrocautery": StructureFacts(
                                        area=0.1, components=1)} if i >= 6 else {})})
             for i in range(10)]
    timeline = build_timeline(facts)  # 2 segments; grasper from 0, hook from 6
    m = selection_metrics([0, 1, 7], timeline, facts)
    assert m["segment_coverage"] == 1.0 and m["event_start_coverage"] == 1.0
    assert m["near_duplicate_pairs"] == 1  # frames 0 and 1
    assert m["structures_per_frame"] == 4 / 3  # grasper; grasper; grasper and hook
    assert selection_metrics([2], timeline, facts)["segment_coverage"] == 0.5

    calls = [{"tool": "frame_stats", "frame_ids": [0, 1, 99], "unknown_ids": [99],
              "returned": [0, 1]},
             {"tool": "view_frames", "frame_ids": [1], "unknown_ids": [], "returned": [1]},
             {"tool": "view_frames", "frame_ids": [1], "unknown_ids": [], "returned": [1]}]
    t = tool_metrics(calls, [0, 7])
    assert (t["calls_with_unknown_ids"], t["images_viewed"], t["frames_viewed_twice"]) == (1, 2, 1)
    assert t["key_frames_inspected"] == 0.5


def test_checklist_checks_and_labels():
    from surgidoc.evals.checklist import DEFAULT_CHECKLIST, present

    claims = [
        Claim(id="c1", claim_type="phase", text="Calot", frame_ids=[0],
              phase="calot_triangle_dissection"),
        Claim(id="c2", claim_type="event", text="The L-hook is in view", event_ids=["E1"]),
        Claim(id="c3", claim_type="limitation", text="The surgeon must confirm the critical view"),
    ]
    assert present({"phase": "calot_triangle_dissection"}, claims, [])
    assert present({"event_structure": "l_hook_electrocautery"}, claims, [])
    assert present({"limitation_mentions": ["critical view"]}, claims, [])
    assert not present({"structure": "gallbladder"}, claims, [])

    checklist = json.loads(DEFAULT_CHECKLIST.read_text())
    for item in checklist["items"]:
        assert item["category"] in checklist["categories"], item["id"]
        assert set(item["sources"]) <= set(checklist["sources"]), item["id"]
    count = {s: sum(s in i["sources"] for i in checklist["items"]) for s in checklist["sources"]}
    assert count["RCSI"] == 14 and count["DSS"] == 6  # the standards' own item counts
