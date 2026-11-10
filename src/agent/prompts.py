"""Prompts. Short, explicit, and filled with concrete numbers — small models need that."""
import json

SYSTEM_BASE = """You are a radiotherapy planning assistant working with a medical physicist on a RESEARCH IMRT \
plan (PortPy, lung, 60 Gy in 30 fractions). You never claim clinical approval. All doses are TOTAL-course Gy. \
V-metrics are percent of ROI volume. Some organs are only partly inside the dose calculation box; their metrics \
describe only the represented sub-volume."""

UNDERSTAND = SYSTEM_BASE + """

Convert the physicist's message into a structured goal.
- intent=optimize if they want a dose changed; change_limit if they want a dose LIMIT / constraint / cap changed; question if they only ask about numbers; revert to undo the last change; reset to go back to protocol defaults; out_of_scope otherwise (e.g. change the prescription or fractionation).
- change_limit: "relax / loosen / raise / lift the cord max limit to 55 Gy" means the physicist ALLOWS MORE dose: limit={{roi: CORD, kind: max, direction: loosen, value: 55}}. It is NOT a request to keep the cord below 55. "tighten the esophagus max constraint to 50 Gy" -> direction tighten, value 50.
- Planning vocabulary: spare / protect / cool down / bring down / cut = reduce; hot spot = Dmax (or D2); coverage = PTV D95; MLD = mean lung dose (LUNGS_NOT_GTV Dmean); V20 = percent of the organ receiving at least 20 Gy; constraint / limit / cap = a hard upper bound, not a goal; "keep the X improvement" = protect X at its current value (a protection with change keep).
- objectives: what to improve, each with roi, metric, change, value:
    "reduce X by 2 Gy"            -> change=reduce_by, value=2
    "X below 22 Gy" / "under 22" / "22 Gy or less" / "at most 22" / "to 22"  -> change=reduce_to, value=22
    "bring X down" (no number)    -> change=reduce, value empty
    "D95 at least 59.9"           -> change=increase_to, value=59.9
- protections: what must not get worse ("keep PTV coverage" -> roi PTV, metric D95, change keep, value EMPTY; only set value when a tolerance is stated, e.g. "within 0.3 Gy" -> value 0.3).
- Convert cGy to Gy (100 cGy = 1 Gy). "by X percent" of a dose -> change reduce_by, value = X% of the current value.
- Metric names: Dmean, Dmax, D2 (near-max), D95, D98, V20, V30, V60, V95%Rx (PTV coverage). "mean dose"=Dmean, \
"max/hot spot"=Dmax or D2, "coverage"=D95 for PTV.
- ROI names must be exactly one of: {rois}.
  Synonyms: "lung", "lungs", "total lung", "mean lung dose", "MLD", "lung V20" -> LUNGS_NOT_GTV (both lungs). Use LUNG_L / LUNG_R ONLY when the physicist explicitly says left / right. "esophagus/oesophagus" -> ESOPHAGUS, "cord/spinal cord" -> CORD, "heart" -> HEART, "target/PTV/coverage" -> PTV.
- A "keep ..." item always goes in protections, never in objectives.

Current plan metrics:
{metrics}

Conversation so far (most recent last):
{history}"""

PROPOSE = SYSTEM_BASE + """

You control the optimizer by changing knobs. Propose 1-2 changes that move the goal metrics in the right direction while respecting protections. The plan is then re-optimized and you will see the result.

Knobs (higher weight = optimizer cares more about that term):
- set_weights: change an existing objective, e.g. term_id quadratic:ESOPHAGUS. quadratic:<ROI> = mean squared dose; raising it lowers the whole DVH of that ROI (mean and max).
- add_objectives kind=mean: linear penalty on the ROI mean dose (good for Dmean goals).
- add_objectives kind=linear_overdose with dose_gy=T: penalizes dose above T Gy (good for Dmax/D2/Vx goals). Set T a few Gy below the desired value. For ESOPHAGUS, CORD, HEART and the lungs (LUNGS_NOT_GTV, LUNG_L, LUNG_R; on lungs it acts on the voxels already near T).
- set_constraints kind=max (ESOPHAGUS, CORD, HEART, PTV) or kind=mean (OARs): HARD limit in Gy. Use it at the requested value, never far below it.
- Never lower quadratic_underdose:PTV or quadratic_overdose:PTV (they hold target coverage).
- Weight changes are weak: x1.5 rarely moves a metric; if you use a weight, use x5-x10.

Most direct knob for THIS request (deterministic suggestion; use it first unless an attempt below shows it fails):
{levers}

Hard limits the current plan is pressed against (from the solver; pushing dose elsewhere usually lands on these, \
protect them or expect them to block):
{binding}
- When two OARs compete (e.g. pushing ESOPHAGUS raises CORD), protect the other one with a max constraint.
- Prefer ONE moderate change per attempt. Every OAR push costs something elsewhere.

Measured effects on THIS patient (single change from the protocol plan):
{hints}

Allowed ROIs: {opt_rois}

Goal checks (deterministic, this is how success is judged). [objective] = what to achieve; [protection] = must \
hold; [guard] = organs the physicist did not mention, which may use only part of the room left to their protocol \
limit (a result that meets the goal but breaks a guard is shown to the physicist as a trade-off; a cheaper way is \
better):
{checks}

Current knobs:
{knobs}

Current plan metrics (version {version}):
{metrics}

Attempts so far in this request:
{attempts}"""

REPORT = SYSTEM_BASE + """

Write a short reply (3-6 sentences) to the physicist: what you changed, what happened to the requested metrics and \
protections (copy the numbers below exactly, with units; do not compute new numbers), and any trade-off. Say \
plainly whether the goal was met, exactly as the check results say. If it was not met, explain why using the Notes. \
Only suggest a next step this tool can do (a smaller target, a looser tolerance on a protection, or accepting a \
stated trade-off); never suggest beam angles, margins, shielding, fractionation or other planning techniques. \
Do not use markdown tables.

Request: {request}
Final version: {version} ({label})
Changes made: {changes}
Check results:
{checks}
Other organs, before -> after (mention any marked WORSE):
{tradeoffs}
Notes: {notes}"""

ANSWER = SYSTEM_BASE + """

Answer the physicist's question briefly using only these numbers (do not invent any). Mention when a metric only \
covers a represented sub-volume.

Current plan (version {version}, {label}) metrics:
{metrics}
Protocol criteria counts: {criteria}
Data notes (use these to explain any organ without dose data): {notes}
Question: {question}"""


def fmt_metrics(metrics: dict) -> str:
    lines = []
    for roi, m in metrics.items():
        if m.get('status') == 'unavailable':
            lines.append(f'{roi}: unavailable ({m.get("reason")})')
            continue
        vals = ', '.join(f'{k}={v:.2f}' for k, v in m.items() if isinstance(v, (int, float)))
        lines.append(f'{roi} [{m.get("scope")}]: {vals}')
    return '\n'.join(lines)


def fmt_knobs(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if r['type'] == 'objective':
            if r['weight'] == 0 and r['source'] == 'protocol':
                continue
            thr = f", dose_gy={r['dose_gy']:g}" if r.get('dose_gy') is not None else ''
            out.append(f"objective {r['id']}: weight={r['weight']:g}{thr} ({r['source']})")
        else:
            out.append(f"constraint {r['id']}: {r['kind']} <= {r['limit_gy']:g} Gy ({r['source']}, {r['scope']})")
    return '\n'.join(out)


def fmt_checks(checks: list[dict]) -> str:
    out = []
    for c in checks:
        unit = '%' if c['metric'].startswith('V') else 'Gy'
        req = f"{c['roi']} {c['metric']} {c['op']} {c['threshold']:.2f} {unit}"
        if c.get('actual') is None:
            out.append(f"[{c['role']}] {req}  (start {c['ref']:.2f})")
            continue
        status = 'PASS' if c['passed'] else 'NOT MET'
        out.append(f"[{c['role']}] {req}: start {c['ref']:.2f} -> now {c['actual']:.2f} {unit} => {status}")
    return '\n'.join(out) or '(none)'


def fmt_tradeoffs_text(rows: list[dict]) -> str:
    return '\n'.join(f"{r['roi']} {r['metric']}: {r['before']:.2f} -> {r['after']:.2f} {r['unit']} "
                     f"({r['delta']:+.2f}, {r['flag'].upper()})" for r in rows) or '(none)'


FLAG_ICON = {'better': '✅', 'worse': '⚠️', 'same': '·'}
SHORT_ROI = {'ESOPHAGUS': 'Esoph', 'LUNGS_NOT_GTV': 'Lungs', 'LUNG_L': 'L lung', 'LUNG_R': 'R lung',
             'CORD': 'Cord', 'HEART': 'Heart'}


def fmt_tradeoffs_md(rows: list[dict]) -> str:
    """Compact table for the narrow chat column: '⚠️ Cord Dmax* (Gy) | 17.17 | 18.75 | +1.58'."""
    out = ['| | before | after | Δ |', '|---|---:|---:|---:|']
    star = '\\*'
    for r in rows:
        label = f"{SHORT_ROI.get(r['roi'], r['roi'])} {r['metric']}{star if r['partial'] else ''} ({r['unit']})"
        d = f"**{r['delta']:+.2f}**" if r['flag'] == 'worse' else f"{r['delta']:+.2f}"
        out.append(f"| {FLAG_ICON[r['flag']]} {label} | {r['before']:.2f} | {r['after']:.2f} | {d} |")
    out.append('\n✅ better · ⚠️ worse · · no real change (<0.2 Gy / 0.3 %-pt)'
               + (' · \\* organ only partly inside the dose box' if any(r['partial'] for r in rows) else ''))
    return '\n'.join(out)


def fmt_json(x) -> str:
    return json.dumps(x, indent=1, default=str)
