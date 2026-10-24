"""The report is a list of claims, not prose.

Each claim says what kind of statement it is and points at the evidence it rests
on (frames, structures, events, guideline passages), so code can check it.
Prose is only rendered from claims that passed verification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "structures_visible",  # these structures are visible in these frames
    "phase",  # these frames belong to this surgical phase
    "event",  # an event from the timeline (instrument in view, possible bleeding)
    "guideline",  # a statement backed by a retrieved guideline passage
    "limitation",  # something the system could not determine (no factual assertion)
]


class Claim(BaseModel):
    id: str
    claim_type: ClaimType
    text: str = Field(description="One sentence for the report")
    frame_ids: list[int] = Field(default_factory=list)
    structures: list[str] = Field(default_factory=list)
    phase: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    passage_ids: list[str] = Field(default_factory=list)
    quote: str | None = Field(
        default=None, description="Guideline claims: a verbatim span from a cited passage")


Status = Literal["supported", "unsupported", "invalid"]
# supported:   the evidence backs the claim
# unsupported: the references exist but the evidence does not back the claim
# invalid:     the claim is malformed or cites things that do not exist

Basis = Literal["validated_model", "unvalidated_model", "retrieved_source", "none"]
# What a supported claim rests on. The final report labels unvalidated_model claims
# (phase estimates, blood, cystic duct, ...) as unverified.


class ClaimResult(BaseModel):
    claim_id: str
    status: Status
    basis: Basis
    reasons: list[str]


class VerificationReport(BaseModel):
    results: list[ClaimResult]

    @property
    def passed(self) -> bool:
        return all(r.status == "supported" for r in self.results)

    def failures(self) -> list[ClaimResult]:
        return [r for r in self.results if r.status != "supported"]
