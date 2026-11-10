"""Structured outputs the LLM must produce. Kept flat and small for 7-8B models.

Proposal uses one list per action type with ALL needed fields required, so Ollama's
JSON-schema constrained decoding cannot omit e.g. the ROI or weight.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Metric = Literal['Dmean', 'Dmax', 'D2', 'D95', 'D98', 'V20', 'V30', 'V60', 'V95%Rx']
Direction = Literal['decrease', 'increase', 'at_most', 'at_least', 'keep']


SCHEMA_VERSION = 'v2'
Change = Literal['reduce_by', 'reduce_to', 'increase_by', 'increase_to', 'reduce', 'increase', 'keep']


class GoalItem(BaseModel):
    """v2: one explicit verb + one value (v1 used direction/amount/target and confused 'by' vs 'to')."""
    roi: str = Field(description='Structure name exactly as listed, e.g. ESOPHAGUS, CORD, PTV, LUNGS_NOT_GTV')
    metric: Metric = Field(description='Dose metric. D* in Gy, V* in percent of ROI volume')
    change: Change = Field(description="reduce_by X: lower by X | reduce_to X: bring to X or below ('below X', "
                                        "'under X', 'at most X', 'X or less') | increase_by X | increase_to X: "
                                        "at least X | reduce / increase: no number given | keep: must not get worse")
    value: Optional[float] = Field(None, description='The number in the request (Gy for D*, percent for V*). '
                                                     'For keep: allowed tolerance if stated. Empty if no number.')

    # Derived v1 view used by goals.py / scoring
    @property
    def direction(self) -> str:
        return {'reduce_by': 'decrease', 'reduce': 'decrease', 'reduce_to': 'at_most', 'increase_by': 'increase',
                'increase': 'increase', 'increase_to': 'at_least', 'keep': 'keep'}[self.change]

    @property
    def amount(self) -> float | None:
        return self.value if self.change in ('reduce_by', 'increase_by', 'keep') else None

    @property
    def target(self) -> float | None:
        return self.value if self.change in ('reduce_to', 'increase_to') else None


class LimitChange(BaseModel):
    roi: str = Field(description='ROI whose dose LIMIT should change, e.g. CORD')
    kind: Literal['max', 'mean'] = Field('max', description='max-dose limit or mean-dose limit')
    direction: Literal['loosen', 'tighten', 'set'] = Field(
        description="loosen: relax / raise / allow more dose; tighten: make the limit stricter; set: other")
    value: Optional[float] = Field(None, description='New limit in total Gy, empty if not stated')


class PhysicistGoal(BaseModel):
    intent: Literal['optimize', 'change_limit', 'question', 'revert', 'reset', 'out_of_scope'] = Field(
        description='optimize: change the plan toward a dose goal; change_limit: relax / loosen / raise / '
                    'tighten a dose LIMIT or constraint; question: only asks about the current plan; '
                    'revert: go back to previous plan; reset: go back to protocol defaults; '
                    'out_of_scope: anything else')
    limit: Optional[LimitChange] = Field(None, description='Only for intent=change_limit')
    objectives: list[GoalItem] = Field(default_factory=list, description='What the physicist wants improved')
    protections: list[GoalItem] = Field(default_factory=list,
                                        description="Things that must not get worse, e.g. PTV D95 change=keep (value = tolerance if stated)")
    summary: str = Field('', description='One-sentence restatement of the request')


class SetWeight(BaseModel):
    term_id: str = Field(description='Objective id from the knob table, e.g. quadratic:ESOPHAGUS')
    weight: float = Field(description='New weight (>= 0)')


class AddObjective(BaseModel):
    roi: str = Field(description='OAR name, e.g. ESOPHAGUS')
    kind: Literal['mean', 'quadratic', 'linear_overdose']
    weight: float = Field(description='Objective weight (> 0)')
    dose_gy: float = Field(0.0, description='Threshold in total Gy; only used by linear_overdose (else 0)')


class SetConstraint(BaseModel):
    roi: str = Field(description='ROI name')
    kind: Literal['max', 'mean']
    limit_gy: float = Field(description='Hard limit in total Gy')


class Proposal(BaseModel):
    reasoning: str = Field(description='ONE short sentence: why these changes move the goal metrics')
    set_weights: list[SetWeight] = Field(default_factory=list)
    add_objectives: list[AddObjective] = Field(default_factory=list)
    set_constraints: list[SetConstraint] = Field(default_factory=list)

    def n_actions(self) -> int:
        return len(self.set_weights) + len(self.add_objectives) + len(self.set_constraints)


# Backwards-compatible flat action (used by tests / tools)
class Action(BaseModel):
    action: Literal['set_weight', 'add_objective', 'set_constraint', 'remove']
    term_id: Optional[str] = None
    roi: Optional[str] = None
    kind: Optional[str] = None
    weight: Optional[float] = None
    dose_gy: Optional[float] = None
    limit_gy: Optional[float] = None


def proposal_actions(p: Proposal) -> list[Action]:
    out = [Action(action='set_weight', term_id=a.term_id, weight=a.weight) for a in p.set_weights]
    out += [Action(action='add_objective', roi=a.roi, kind=a.kind, weight=a.weight,
                   dose_gy=a.dose_gy if a.kind == 'linear_overdose' else None) for a in p.add_objectives]
    out += [Action(action='set_constraint', roi=a.roi, kind=a.kind, limit_gy=a.limit_gy) for a in p.set_constraints]
    return out


def make_proposal_schema(term_ids: list[str], oar_rois: list[str], constraint_rois: list[str],
                         overdose_rois: list[str]):
    """Per-session Proposal schema whose ids/ROIs are enums -> constrained decoding cannot invent them."""
    from pydantic import create_model
    TermId = Literal[tuple(term_ids)]
    Oar = Literal[tuple(sorted(set(oar_rois) | set(overdose_rois)))]
    ConsRoi = Literal[tuple(constraint_rois)]
    SW = create_model('SetWeight', term_id=(TermId, Field(description='Existing objective id')),
                      weight=(float, Field(description='New weight (>= 0)')))
    AO = create_model('AddObjective', roi=(Oar, ...), kind=(Literal['mean', 'quadratic', 'linear_overdose'], ...),
                      weight=(float, Field(description='Objective weight (> 0)')),
                      dose_gy=(float, Field(0.0, description='Threshold Gy; only for linear_overdose (else 0)')))
    SC = create_model('SetConstraint', roi=(ConsRoi, ...), kind=(Literal['max', 'mean'], ...),
                      limit_gy=(float, Field(description='Hard limit in total Gy')))
    return create_model('Proposal', __base__=Proposal,
                        set_weights=(list[SW], Field(default_factory=list)),
                        add_objectives=(list[AO], Field(default_factory=list)),
                        set_constraints=(list[SC], Field(default_factory=list)))
