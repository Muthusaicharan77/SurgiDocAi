"""Node verify: check every claim against the perception facts. No LLM involved.

The verifier checks that claims agree with what the vision models output. It does
not know the ground truth; how often the models themselves are wrong is measured
separately (RESULTS.md, and the grounding eval against ground-truth masks).
"""

from __future__ import annotations

import re

from surgidoc.vision.classes import CLASS_NAMES
from surgidoc.vision.phase import PHASES

from .claims import Basis, Claim, ClaimResult, VerificationReport
from .facts import FrameFacts, Timeline, validated_classes

# A claim about a set of frames holds if it is true in at least this share of them.
MIN_SUPPORT = 0.8

# A guideline claim must quote its passage. The quote is checked as a verbatim span,
# so the reviewer can compare the claim with the source's own words. Code checks that
# the quote exists, not that the claim paraphrases it faithfully; that stays with the
# reviewer. Bounds keep the quote meaningful and stop it from pasting the whole passage.
QUOTE_MIN_WORDS = 6
QUOTE_MAX_WORDS = 60

# Words in claim text that name a structure. Any structure the text names must be
# listed in claim.structures, so it gets checked: otherwise a claim could cite
# "grasper" while its sentence says "cystic duct". This is a keyword heuristic,
# not language understanding; it catches the plain case.
STRUCTURE_WORDS: dict[str, list[str]] = {
    "abdominal_wall": ["abdominal wall"],
    "liver": ["liver"],
    "gastrointestinal_tract": ["gastrointestinal", "gi tract", "bowel", "duodenum", "intestine"],
    "fat": ["fat", "fatty tissue"],
    "grasper": ["grasper", "graspers"],
    "connective_tissue": ["connective tissue"],
    "blood": ["blood", "bleeding", "haemorrhage", "hemorrhage"],
    "cystic_duct": ["cystic duct"],
    "l_hook_electrocautery": ["l-hook", "hook", "electrocautery", "diathermy"],
    "gallbladder": ["gallbladder", "gall bladder"],
    "hepatic_vein": ["hepatic vein"],
    "liver_ligament": ["liver ligament", "ligament"],
}

# Phase names contain structure words ("gallbladder dissection"); strip them first.
PHASE_WORDS = [p.replace("_", " ") for p in PHASES] + [
    "calot's triangle dissection", "clipping and cutting", "cleaning and coagulation",
]


def mentioned_structures(text: str) -> set[str]:
    text = text.lower()
    for phrase in sorted(PHASE_WORDS, key=len, reverse=True):
        text = text.replace(phrase, " ")
    found = set()
    # Longest phrases first, removing each match, so "liver ligament" is not also "liver".
    pairs = sorted(((w, c) for c, ws in STRUCTURE_WORDS.items() for w in ws),
                   key=lambda p: len(p[0]), reverse=True)
    for word, cls in pairs:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text):
            found.add(cls)
            text = re.sub(pattern, " ", text)
    return found


def normalise(text: str) -> str:
    """Lower case, one kind of quote mark and dash, single spaces, no outer quotes."""
    text = text.lower().translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " "}))
    return " ".join(text.split()).strip(' "\'')


def quote_problem(quote: str | None, sources: list[str]) -> str | None:
    """Why the quote fails, or None if it is a verbatim span of one of the sources."""
    if not quote:
        return "guideline claim needs a quote from its passage"
    n = len(normalise(quote).split())
    if not QUOTE_MIN_WORDS <= n <= QUOTE_MAX_WORDS:
        return f"quote has {n} words, needs {QUOTE_MIN_WORDS}-{QUOTE_MAX_WORDS}"
    if not any(normalise(quote) in normalise(s) for s in sources):
        return "quote does not appear verbatim in the cited passages"
    return None


def verify(claims: list[Claim], facts: list[FrameFacts], timeline: Timeline,
           retrieved_passages: dict[str, str]) -> VerificationReport:
    """retrieved_passages maps the id of each passage retrieved in this run to its text."""
    by_id = {f.frame_id: f for f in facts}
    events = {e.id: e for e in timeline.events}
    validated = validated_classes()
    return VerificationReport(results=[
        _check(c, by_id, events, validated, retrieved_passages) for c in claims
    ])


def _check(c: Claim, frames: dict, events: dict, validated: frozenset[str],
           passages: dict[str, str]) -> ClaimResult:
    invalid, unsupported = [], []

    # --- references must exist ------------------------------------------------
    if missing := [i for i in c.frame_ids if i not in frames]:
        invalid.append(f"unknown frame ids {missing}")
    if unknown := [s for s in c.structures if s not in CLASS_NAMES]:
        invalid.append(f"unknown structures {unknown}")
    if missing := [e for e in c.event_ids if e not in events]:
        invalid.append(f"unknown event ids {missing}")
    if c.phase is not None and c.phase not in PHASES:
        invalid.append(f"unknown phase '{c.phase}'")
    # Guideline and limitation claims talk about structures without asserting they
    # were seen, so they skip this check; guideline claims may not cite frames instead.
    if c.claim_type not in ("guideline", "limitation") and (
        extra := mentioned_structures(c.text) - set(c.structures)
    ):
        invalid.append(f"text names {sorted(extra)} but structures does not list them")

    # --- per-type requirements --------------------------------------------------
    needs = {
        "structures_visible": [("frame_ids", c.frame_ids), ("structures", c.structures)],
        "phase": [("frame_ids", c.frame_ids), ("phase", c.phase)],
        "event": [("event_ids", c.event_ids)],
        "guideline": [("passage_ids", c.passage_ids)],
        "limitation": [],
    }[c.claim_type]
    invalid += [f"{c.claim_type} claim needs {name}" for name, value in needs if not value]
    if c.claim_type == "guideline":
        if bad := [p for p in c.passage_ids if p not in passages]:
            invalid.append(f"passages {bad} were not retrieved in this run")
        if c.frame_ids or c.structures:
            invalid.append("guideline claims state guidance, not observations; "
                           "put observations in their own claim")
        if not bad and (problem := quote_problem(c.quote, [passages[p] for p in c.passage_ids])):
            invalid.append(problem)

    if invalid:
        return ClaimResult(claim_id=c.id, status="invalid", basis="none", reasons=invalid)

    # --- does the evidence back it? -------------------------------------------------
    cited = [frames[i] for i in c.frame_ids]
    if c.claim_type == "event" and not cited:
        cited = [f for f in frames.values()
                 if any(events[e].start_frame <= f.frame_id <= events[e].end_frame
                        for e in c.event_ids)]

    if c.claim_type not in ("guideline", "limitation"):
        for s in c.structures:
            n = sum(f.visible(s) for f in cited)
            if not cited or n / len(cited) < MIN_SUPPORT:
                unsupported.append(f"{s} visible in {n}/{len(cited)} cited frames")

    if c.claim_type == "phase":
        n = sum(f.phase == c.phase for f in cited)
        if n / len(cited) < MIN_SUPPORT:
            unsupported.append(f"phase {c.phase} in {n}/{len(cited)} cited frames")

    if c.claim_type == "event" and c.frame_ids:
        for e in (events[i] for i in c.event_ids):
            if not any(e.start_frame <= f.frame_id <= e.end_frame for f in cited):
                unsupported.append(f"cited frames do not overlap event {e.id}")

    if unsupported:
        return ClaimResult(claim_id=c.id, status="unsupported", basis="none", reasons=unsupported)
    return ClaimResult(claim_id=c.id, status="supported",
                       basis=_basis(c, events, validated), reasons=[])


def _basis(c: Claim, events: dict, validated: frozenset[str]) -> Basis:
    if c.claim_type == "guideline":
        return "retrieved_source"
    if c.claim_type == "limitation":
        return "none"
    unvalidated = (
        c.claim_type == "phase"  # phase model has no accuracy measured in this project
        or any(s not in validated for s in c.structures)
        or any(not events[e].validated for e in c.event_ids)
    )
    return "unvalidated_model" if unvalidated else "validated_model"
