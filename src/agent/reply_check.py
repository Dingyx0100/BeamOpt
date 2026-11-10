"""Deterministic checks on the LLM-written reply before it reaches the physicist.

1. Every number in the reply must come from the facts the LLM was given (or a difference of two of them).
2. Comparisons such as "18.7 Gy is below 0.5 Gy" and "rose from 17.2 to 16.9" must be arithmetically true.
3. A reply may not claim success when the deterministic check says the goal was not met.
4. Sentences suggesting things this tool cannot do (beam angles, margins, shielding, ...) are removed.
"""
from __future__ import annotations

import re

NUM = r'-?\d+(?:\.\d+)?'
OUT_OF_SCOPE = re.compile(
    r'beam angle|gantry|beam arrangement|beam configuration|number of beams|couch|margin|shield|\bblock(?:s|ing)?\b|'
    r'fractionation|hypofraction|\bvmat\b|\barcs?\b|proton|re-?contour|re-?plan|immobili|breath[- ]?hold|gating|'
    r'motion management|collimator|wedge|dose engine|\bimrt technique', re.I)
SUCCESS_CLAIM = re.compile(
    r'\b(successfully|achieved the (?:goal|target|request(?:ed)? (?:change|reduction))|'
    r'(?:goal|target|request)\b[^.;]{0,40}?\b(?:was|is|has been)\s+(?:fully\s+)?(?:met|achieved|reached)|'
    r'(?:met|meeting|meets|satisf(?:y|ies|ying)|fulfill(?:ing|s)?) (?:the|your) (?:[\w-]+ )?(?:goal|target|request)|'
    r'now (?:meets|satisfies) (?:the|your) (?:goal|target|request))', re.I)
NEGATION = re.compile(r"\b(not|n't|no|unable|could not|couldn't|without|fail(?:ed)?|partially|partly)\b", re.I)
UP = r'(?:rose|rises|increased|increases|went up|goes up|higher|grew)'
DOWN = r'(?:fell|falls|dropped|drops|decreased|decreases|reduced|lowered|went down|goes down|lower)'


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(NUM, text or '')]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', (text or '').strip()) if s]


def _allowed(facts: list[float]) -> list[float]:
    base = sorted({round(abs(f), 3) for f in facts})
    out = set(base)
    if len(base) <= 160:   # differences of facts ("dropped by 0.8 Gy") are legitimate
        for i, a in enumerate(base):
            for b in base[i + 1:]:
                out.add(round(b - a, 3))
    return sorted(out)


def unknown_numbers(reply: str, facts: list[float]) -> list[str]:
    allowed = _allowed(facts)
    bad = []
    for tok in re.findall(NUM, reply or ''):
        v = abs(float(tok))
        if v <= 10 and float(v).is_integer():
            continue                                    # counts / attempt numbers / versions / small integers
        decimals = len(tok.split('.')[1]) if '.' in tok else 0
        tol = 0.5 * 10 ** -decimals + 1e-6              # "0.2" matches 0.20 or 0.24; "18" matches 17.5..18.5
        if not any(abs(v - a) <= tol for a in allowed):
            bad.append(tok)
    return bad


def false_comparisons(reply: str) -> list[str]:
    bad = []
    for s in _sentences(reply):
        for m in re.finditer(rf'({NUM})\s*(?:gy|%)?[^.;]{{0,40}}?\b(below|under|less than|lower than|within)\b'
                             rf'[^.;\d]{{0,30}}?({NUM})\s*(?:gy|%)', s, re.I):
            a, word, b = float(m.group(1)), m.group(2).lower(), float(m.group(3))
            if word != 'within' and a > b + 1e-6:
                bad.append(m.group(0))
        for m in re.finditer(rf'({NUM})\s*(?:gy|%)?[^.;]{{0,40}}?\b(above|over|more than|higher than|exceed\w*)\b'
                             rf'[^.;\d]{{0,30}}?({NUM})\s*(?:gy|%)', s, re.I):
            if float(m.group(1)) < float(m.group(3)) - 1e-6:
                bad.append(m.group(0))
        for m in re.finditer(rf'\b({UP}|{DOWN})\b[^.;]{{0,30}}?\bfrom\s+({NUM})\s*(?:gy|%)?\s*(?:to|->|→)\s*({NUM})',
                             s, re.I):
            word, a, b = m.group(1).lower(), float(m.group(2)), float(m.group(3))
            up = re.fullmatch(UP, word, re.I) is not None
            if (up and b < a - 1e-6) or (not up and b > a + 1e-6):
                bad.append(m.group(0))
    return bad


def false_success(reply: str, outcome: str) -> list[str]:
    if outcome in ('met', 'needs_decision'):      # needs_decision: the goal WAS met, at a cost the physicist decides
        return []
    bad = []
    for s in _sentences(reply):
        if SUCCESS_CLAIM.search(s) and not NEGATION.search(s):
            bad.append(s)
    return bad


def strip_out_of_scope(reply: str) -> tuple[str, list[str]]:
    keep, dropped = [], []
    for s in _sentences(reply):
        (dropped if OUT_OF_SCOPE.search(s) else keep).append(s)
    return ' '.join(keep), dropped


def problems(reply: str, facts: list[float], outcome: str) -> list[str]:
    out = []
    unk = unknown_numbers(reply, facts)
    if unk:
        out.append('these numbers are not in the facts: ' + ', '.join(unk))
    out += [f'false comparison: "{c}"' for c in false_comparisons(reply)]
    out += [f'claims success but the goal was NOT met: "{c}"' for c in false_success(reply, outcome)]
    return out
