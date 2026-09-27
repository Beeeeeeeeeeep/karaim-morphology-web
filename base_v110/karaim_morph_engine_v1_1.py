#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Karaim Morphophonological Engine v1.1.0 — full verbal morphology

Rule-based nominal + verbal morphophonological analyzer for canonical KRPS Karaim.

Key v1.1 changes:
    1. closes 78 corpus-confirmed residual verbal-morphology occurrences from
       the v1.1 audit; five earlier screening hits are explicitly reclassified
       instead of being forced into grammar;
    2. expands UVCI, negative -мя-, present participles, nonfinite nominal
       stacking and selected outer nonfinite morphology under hard KRPS gates;
    3. adds narrowly audited voice/aspect bridges rather than globally enabling
       ambiguous one-segment derivational suffixes;
    4. introduces opt-in historical_profile='historical_western' and moves
       Western historical variants out of default contemporary morphology;
    5. preserves the canonical-input contract: no OCR repair, fuzzy matching,
       transliteration, case-folding or Unicode normalization is performed.

Input contract:
    - the engine receives already cleaned canonical Cyrillic KRPS Karaim data;
    - OCR detection/correction, transliteration and script repair are outside this module;
    - Latin-script Karaim is never accepted as an alternative parser orthography.

Scope:
    nominal morphology retained from v0.9.1-LHF1 plus source-grounded finite and
    non-finite verbal morphology, voice/derivation, modal morphology and selected
    analytic verbal constructions under hard lexical and dialect constraints.
"""
from __future__ import annotations

import argparse
import csv
import json
import copy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set, Any

VERSION = "1.1.0"
DIALECT_ALL = ("H", "T", "K")
DIALECT_ALIASES = {"H": "H", "T": "T", "K": "K"}

# ---------------------------------------------------------------------------
# Input representation
# ---------------------------------------------------------------------------
# v0.7 intentionally does NOT normalize Unicode, case, apostrophes, dashes,
# scripts or OCR artifacts. The morphology engine consumes the canonical KRPS
# representation produced upstream. Only surrounding record whitespace is
# removed when reading plain-text/CSV transport.

def prepare_surface(token: str) -> str:
    return token.strip()

def normalize_dialect(dialect: Optional[str]) -> Optional[str]:
    if dialect is None:
        return None
    d = str(dialect).strip().upper()
    if not d or d in {"UNKNOWN", "NONE", "?"}:
        return None
    if d not in DIALECT_ALIASES:
        raise ValueError(f"Unsupported dialect '{dialect}'. Use H, T, K or UNKNOWN.")
    return DIALECT_ALIASES[d]

def normalize_dialects(dialect: Optional[str] = None,
                       dialects: Optional[Iterable[str]] = None) -> List[str]:
    raw: List[str] = []
    if dialect is not None:
        raw.append(dialect)
    if dialects is not None:
        raw.extend(list(dialects))
    out: List[str] = []
    for x in raw:
        d = normalize_dialect(x)
        if d and d not in out:
            out.append(d)
    return [d for d in DIALECT_ALL if d in out]

# ---------------------------------------------------------------------------
# Phonological features
# ---------------------------------------------------------------------------
FRONT_VOWELS = set("эеияӧёӱю")
BACK_VOWELS = set("аыоу")
ROUNDED_VOWELS = set("оуӧёӱю")
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS
VOICELESS_FINALS = set("пфктсшщчцх")
VOICELESS_DIGRAPHS = ("къ", "хъ")
VOICED_OBSTRUENTS = set("бвгдзж")
SONORANTS = set("мнңнъйлр")

def vowels(s: str) -> List[str]:
    return [ch for ch in s if ch in ALL_VOWELS]


def harmony_features(stem: str) -> Dict[str, Optional[str]]:
    vs = vowels(stem)
    if not vs:
        return {"frontness": None, "rounding": None, "last_vowel": None}
    v = vs[-1]
    return {
        "frontness": "front" if v in FRONT_VOWELS else "back",
        "rounding": "rounded" if v in ROUNDED_VOWELS else "unrounded",
        "last_vowel": v,
    }


def final_segment_class(stem: str) -> str:
    t = stem.rstrip("'ьъ")
    if not t:
        return "unknown"
    if any(t.endswith(d) for d in VOICELESS_DIGRAPHS):
        return "voiceless_obstruent"
    ch = t[-1]
    if ch in ALL_VOWELS:
        return "vowel"
    if ch in VOICELESS_FINALS:
        return "voiceless_obstruent"
    if ch in VOICED_OBSTRUENTS:
        return "voiced_obstruent"
    if ch in SONORANTS:
        return "sonorant"
    return "other"


def expected_high_vowel(stem: str) -> Optional[set]:
    f = harmony_features(stem)
    if f["frontness"] is None:
        return None
    if f["frontness"] == "back" and f["rounding"] == "unrounded":
        return {"ы"}
    if f["frontness"] == "back" and f["rounding"] == "rounded":
        return {"у"}
    if f["frontness"] == "front" and f["rounding"] == "unrounded":
        return {"и"}
    return {"ӱ", "ю"}


def expected_low_vowel(stem: str) -> Optional[set]:
    f = harmony_features(stem)
    if f["frontness"] is None:
        return None
    return {"а"} if f["frontness"] == "back" else {"э", "е", "я"}


def suffix_harmony_score(stem: str, suffix: str, kind: str) -> Tuple[float, str]:
    if not stem:
        return -2.0, "empty_stem"
    sv = next((ch for ch in suffix if ch in ALL_VOWELS), None)
    if not sv:
        return 0.0, "no_vowel_to_validate"
    exp = expected_low_vowel(stem) if kind == "low" else expected_high_vowel(stem) if kind == "high" else None
    if exp is None:
        return 0.0, "harmony_unknown"
    if sv in exp:
        return 0.8, "harmony_match"
    stem_front = harmony_features(stem)["frontness"]
    suf_front = "front" if sv in FRONT_VOWELS else "back"
    if stem_front and stem_front != suf_front:
        return -1.1, "front_back_mismatch"
    return -0.25, "rounding_or_surface_mismatch"


def _case_onset_score(stem: str, suffix: str, category: str) -> Tuple[float, Optional[str]]:
    if category not in {"CASE.LOC", "CASE.ABL"} or not suffix:
        return 0.0, None
    c = suffix[0]
    cls = final_segment_class(stem)
    if c == "т" and cls == "voiceless_obstruent":
        return 0.5, "onset_voicing_match"
    if c == "д" and cls in {"vowel", "sonorant", "voiced_obstruent"}:
        return 0.5, "onset_voicing_match"
    if c == "т" and cls in {"vowel", "sonorant", "voiced_obstruent"}:
        return -0.45, "onset_voicing_mismatch"
    if c == "д" and cls == "voiceless_obstruent":
        return -0.45, "onset_voicing_mismatch"
    return 0.0, None

# ---------------------------------------------------------------------------
# Productive morphology: Cyrillic only
# ---------------------------------------------------------------------------
PL_FORMS = ["ляр", "лэр", "лер", "лар"]

POSS_FORMS: Dict[str, List[str]] = {
    "POSS.1PL": ["ымыз", "имиз", "умуз", "ӱмӱз", "юмюз", "мыз", "миз", "муз", "мӱз", "мюз"],
    "POSS.2PL.T": ["ыйыз", "ийиз", "уйуз", "ӱйӱз", "юйюз", "йыз", "йиз", "йуз", "йӱз"],
    "POSS.2PL.HK": ["ыныз", "иниз", "унуз", "ӱнӱз", "юнюз", "ныз", "низ", "нуз", "нӱз"],
    "POSS.1SG": ["ым", "им", "ум", "ӱм", "юм", "м"],
    "POSS.2SG.K": ["ынъ", "инъ", "унъ", "ӱнъ"],
    "POSS.2SG.HK": ["ын", "ин", "ун", "ӱн", "н"],
    "POSS.2SG.T": ["ый", "ий", "уй", "ӱй", "юй", "й"],
    "POSS.3.BUFFER_S": ["сы", "си", "су", "сӱ", "сю"],
    "POSS.3": ["ы", "и", "у", "ӱ", "ю"],
}
POSS_DIALECTS = {
    "POSS.1PL": list(DIALECT_ALL),
    "POSS.2PL.T": ["T"],
    "POSS.2PL.HK": ["H", "K"],
    "POSS.1SG": list(DIALECT_ALL),
    "POSS.2SG.K": ["K"],
    "POSS.2SG.HK": ["H", "K"],
    "POSS.2SG.T": ["T"],
    "POSS.3.BUFFER_S": list(DIALECT_ALL),
    "POSS.3": list(DIALECT_ALL),
}

CASE_FORMS: Dict[str, List[Tuple[str, str]]] = {
    "CASE.GEN": [(x, "high") for x in ["нинь", "нынь", "нюнь", "нӱнь", "нунь", "нын", "нин", "нун", "нӱн"]],
    "CASE.ACC": [(x, "high") for x in ["ны", "ни", "ну", "нӱ", "ню"]],
    "CASE.DAT": [(x, "low") for x in ["гъа", "къа", "гя", "кя", "гэ", "кэ", "ге", "ке", "га", "ка", "ха", "ча"]],
    "CASE.LOC": [(x, "low") for x in ["дя", "дэ", "да", "тя", "тэ", "та"]],
    "CASE.ABL": [(x, "low") for x in ["дянь", "тянь", "дэнь", "тэнь", "дан", "дэн", "тан", "тэн"]],
    "CASE.INS": [(x, "low") for x in ["бя", "ба"]],
}
# Hard restrictions only where v0.3 evidence is strong enough.
CASE_DIALECTS = {
    "CASE.GEN": list(DIALECT_ALL),
    "CASE.ACC": list(DIALECT_ALL),
    "CASE.DAT": list(DIALECT_ALL),
    "CASE.LOC": list(DIALECT_ALL),
    "CASE.ABL": list(DIALECT_ALL),
    "CASE.INS": ["H", "T"],
}

P3_CASE_COMBINED: Dict[str, List[Tuple[str, str, str]]] = {
    "CASE.ABL": [("ньдянь", "нь", "дянь"), ("ньдэнь", "нь", "дэнь"), ("ндянь", "н", "дянь"), ("ндэнь", "н", "дэнь"), ("ндан", "н", "дан"), ("ндэн", "н", "дэн")],
    "CASE.LOC": [("ньдя", "нь", "дя"), ("ньдэ", "нь", "дэ"), ("ндя", "н", "дя"), ("ндэ", "н", "дэ"), ("нда", "н", "да")],
    "CASE.DAT": [("нъя", "нъ", "я"), ("нъэ", "нъ", "э"), ("нъа", "нъ", "а"), ("нья", "нь", "я"), ("ньэ", "нь", "э"), ("ня", "н", "я"), ("нэ", "н", "э"), ("на", "н", "а")],
    "CASE.ACC": [("нь", "нь", ""), ("н", "н", "")],
}
POSS_NON3_DAT = [("я", "low"), ("э", "low"), ("а", "low")]

DERIV_FORMS: Dict[str, List[str]] = {
    "DERIV.NOMINAL_LIK": ["лыкъ", "лик", "лук", "люк", "лых", "лих", "лух", "люх"],
    "DERIV.ADJ_LI": ["лы", "ли", "лу", "лю"],
    "DERIV.PRIV_SIZ": ["сыз", "сиз", "суз", "сӱз", "сюз"],
    "DERIV.AGENT_CI": ["джы", "джи", "джу", "джю", "чы", "чи", "чу", "чю", "цы", "ци", "цу", "цю"],
    "DERIV.VN_UV": ["ув", "ӱв", "ив", "ыв"],
    "DERIV.VN_IS": ["ыш", "иш", "уш", "ӱш"],
}

VERB_INF_ENDINGS = ["мак", "мэк", "мах", "мях"]
VERB_PART_ENDINGS = ["гъан", "ган", "гэн", "ген", "къан", "кан", "кян"]
VERB_FINITE_ENDINGS = [
    "диляр", "дылар", "дюляр", "дулар", "тиляр", "тылар", "ды", "ди", "ду", "дю", "ты", "ти", "ту", "тю",
    "адырлар", "ядирляр", "эдирляр", "йдырлар", "йдирляр", "дырлар", "дирляр",
]

LHF_CATALOG_BASENAME = "lexicalized_historical_forms_v1.0.json"

def load_lhf_catalog(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load verified lexicalized historical forms by exact canonical surface.

    The loader intentionally performs no transliteration, case folding, Unicode
    normalization, OCR repair or fuzzy matching. Only parser_enabled + verified
    rows become active.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for row in data.get("entries", []):
        if not row.get("parser_enabled", False):
            continue
        if row.get("evidence_status") != "verified":
            continue
        surface = str(row.get("canonical_surface") or "").strip()
        if not surface:
            continue
        if surface in out:
            raise ValueError(f"Duplicate LHF canonical surface: {surface}")
        out[surface] = row
    return out

def _default_lhf_catalog() -> Dict[str, Dict[str, Any]]:
    p = Path(__file__).resolve().with_name(LHF_CATALOG_BASENAME)
    return load_lhf_catalog(p) if p.exists() else {}

LEXICALIZED = _default_lhf_catalog()

# ---------------------------------------------------------------------------
# KRPS lexicon: second hard constraint
# ---------------------------------------------------------------------------
KRPS_DIALECT_LABELS = {
    "dialekt łucko-halicki": "H",
    "dialekt trocki": "T",
    "dialekt krymski": "K",
}


def parse_krps_dialects(raw: str) -> List[str]:
    if not raw:
        return []
    found: List[str] = []
    for part in str(raw).split(";"):
        d = KRPS_DIALECT_LABELS.get(part.strip())
        if d and d not in found:
            found.append(d)
    return [d for d in DIALECT_ALL if d in found]


def is_atomic_stem_headword(headword: str) -> bool:
    """Can this headword validate a stem in productive nominal parsing?

    Complex dictionary labels/collocations remain available for exact lookup,
    but are not treated as productive single-token stems. Verbal dictionary
    stems ending in '-' are also excluded from nominal stem validation.
    """
    if not headword or headword.endswith("-"):
        return False
    forbidden = set(":;()[]= ")
    return not any(ch in forbidden or ch.isspace() for ch in headword)


@dataclass
class LexiconEntry:
    headword: str
    dialects: List[str]
    grammatical_forms: List[Dict[str, Any]] = field(default_factory=list)
    source_rows: List[int] = field(default_factory=list)
    atomic_stem: bool = True
    source_kind: str = "KRPS"
    source_ids: List[str] = field(default_factory=list)
    lexeme_type: Optional[str] = None


class KRPSLexicon:
    def __init__(self, entries: Dict[str, LexiconEntry], source: Optional[str] = None):
        self.entries = entries
        self.source = source
        self.atomic_stems: Set[str] = {h for h, e in entries.items() if e.atomic_stem}
        self.verb_headwords: Set[str] = {h for h in entries if h.endswith("-")}
        self.verb_bases: Dict[str, LexiconEntry] = {h[:-1]: e for h, e in entries.items() if h.endswith("-") and len(h) > 1}

    @classmethod
    def from_csv(cls, path: Path) -> "KRPSLexicon":
        entries: Dict[str, LexiconEntry] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"haslo_karaimskie", "dialekt", "forma_gramatyczna"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError("KRPS lexicon CSV missing columns: " + ", ".join(sorted(missing)))
            for rowno, row in enumerate(reader, start=2):
                headword = (row.get("haslo_karaimskie") or "").strip()
                if not headword:
                    continue
                dialects = parse_krps_dialects(row.get("dialekt") or "")
                gf_raw = (row.get("forma_gramatyczna") or "").strip()
                gf = None
                if gf_raw:
                    try:
                        gf = json.loads(gf_raw)
                    except json.JSONDecodeError:
                        gf = {"raw": gf_raw, "parse_error": True}
                if headword not in entries:
                    entries[headword] = LexiconEntry(
                        headword=headword,
                        dialects=dialects,
                        grammatical_forms=[gf] if gf else [],
                        source_rows=[rowno],
                        atomic_stem=is_atomic_stem_headword(headword),
                    )
                else:
                    e = entries[headword]
                    e.dialects = [d for d in DIALECT_ALL if d in set(e.dialects) | set(dialects)]
                    if gf and gf not in e.grammatical_forms:
                        e.grammatical_forms.append(gf)
                    e.source_rows.append(rowno)
        return cls(entries, str(path))

    def get(self, headword: str) -> Optional[LexiconEntry]:
        return self.entries.get(headword)

    def add_external_csv(self, path: Path) -> None:
        """Add vetted non-KRPS lexical items (e.g. proper names) as a separate layer.

        Expected columns: headword,dialects,lexeme_type. `dialects` uses H|T|K
        and may be empty. This is not spelling repair: only explicitly listed
        canonical items are added.
        """
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader=csv.DictReader(f)
            if "headword" not in (reader.fieldnames or []):
                raise ValueError("External lexicon CSV requires a headword column")
            for rowno,row in enumerate(reader,start=2):
                h=(row.get("headword") or "").strip()
                if not h:
                    continue
                ds=[]
                raw=(row.get("dialects") or "").replace(",","|")
                for x in raw.split("|"):
                    d=normalize_dialect(x) if x.strip() else None
                    if d and d not in ds: ds.append(d)
                source_id=(row.get("source_note") or "external_vetted").strip() or "external_vetted"
                lexeme_type=(row.get("lexeme_type") or "external").strip() or "external"
                if h not in self.entries:
                    self.entries[h]=LexiconEntry(h,ds,[],[-rowno],is_atomic_stem_headword(h),
                                                 source_kind="external_vetted",
                                                 source_ids=[source_id],lexeme_type=lexeme_type)
                else:
                    e=self.entries[h]
                    e.dialects=[d for d in DIALECT_ALL if d in set(e.dialects)|set(ds)]
                    if e.source_kind != "KRPS":
                        e.source_kind="external_vetted"
                        if source_id not in e.source_ids: e.source_ids.append(source_id)
                        e.lexeme_type=e.lexeme_type or lexeme_type
                if is_atomic_stem_headword(h): self.atomic_stems.add(h)
                if h.endswith("-"):
                    self.verb_headwords.add(h); self.verb_bases[h[:-1]]=self.entries[h]

    def add_attested_csv(self, path: Path) -> None:
        """Add exact, source-attested Karaim lexemes absent from KRPS.

        This is a deliberately stricter layer than ``add_external_csv``. Required
        columns are ``headword,dialects,lexeme_type,source_id,evidence_status``.
        Only rows with ``evidence_status=verified`` and a non-empty ``source_id``
        are admitted. Headwords are consumed literally: no normalization,
        transliteration, fuzzy matching or script conversion is performed.
        """
        required={"headword","dialects","lexeme_type","source_id","evidence_status"}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader=csv.DictReader(f)
            missing=required-set(reader.fieldnames or [])
            if missing:
                raise ValueError("Attested lexicon CSV missing columns: "+", ".join(sorted(missing)))
            for rowno,row in enumerate(reader,start=2):
                if (row.get("evidence_status") or "").strip().lower() != "verified":
                    continue
                h=(row.get("headword") or "").strip()
                source_id=(row.get("source_id") or "").strip()
                if not h or not source_id:
                    continue
                ds=[]
                raw=(row.get("dialects") or "").replace(",","|")
                for x in raw.split("|"):
                    d=normalize_dialect(x) if x.strip() else None
                    if d and d not in ds: ds.append(d)
                lexeme_type=(row.get("lexeme_type") or "attested").strip() or "attested"
                if h not in self.entries:
                    self.entries[h]=LexiconEntry(h,ds,[],[-100000-rowno],is_atomic_stem_headword(h),
                                                 source_kind="supplemental_attested",
                                                 source_ids=[source_id],lexeme_type=lexeme_type)
                else:
                    e=self.entries[h]
                    e.dialects=[d for d in DIALECT_ALL if d in set(e.dialects)|set(ds)]
                    if e.source_kind != "KRPS":
                        e.source_kind="supplemental_attested"
                        if source_id not in e.source_ids: e.source_ids.append(source_id)
                        e.lexeme_type=e.lexeme_type or lexeme_type
                if is_atomic_stem_headword(h): self.atomic_stems.add(h)
                if h.endswith("-"):
                    self.verb_headwords.add(h); self.verb_bases[h[:-1]]=self.entries[h]

    def dialect_compatible(self, entry: LexiconEntry, requested: Iterable[str]) -> bool:
        req = set(requested)
        if not req or not entry.dialects:
            return True
        return bool(req & set(entry.dialects))

    def resolve_candidate(self, stem: str, restored: Iterable[str], requested: Iterable[str]) -> Optional[Tuple[LexiconEntry, str]]:
        # exact stem always outranks a restored stem
        for lemma, match_type in [(stem, "exact_stem")] + [(x, "morphophonological_restoration") for x in restored]:
            e = self.entries.get(lemma)
            if e and e.atomic_stem and self.dialect_compatible(e, requested):
                return e, match_type
        return None

    def resolve_verb_candidate(self, stem: str, restored: Iterable[str], requested: Iterable[str]) -> Optional[Tuple[LexiconEntry, str]]:
        for base, match_type in [(stem, "exact_verb_stem")] + [(x, "morphophonological_restoration") for x in restored]:
            e = self.verb_bases.get(base)
            if e and self.dialect_compatible(e, requested):
                return e, match_type
        return None

    def derivational_chain(self, headword: str, max_depth: int = 6) -> List[Dict[str, Any]]:
        chain_rev: List[Dict[str, Any]] = []
        current = headword
        seen = set()
        for _ in range(max_depth):
            if current in seen:
                break
            seen.add(current)
            e = self.entries.get(current)
            if not e or not e.grammatical_forms:
                break
            rel = next((x for x in e.grammatical_forms if isinstance(x, dict) and x.get("od") and x.get("type")), None)
            if not rel:
                break
            parent = str(rel.get("od")).strip()
            typ = rel.get("type")
            child_base = current[:-1] if current.endswith("-") else current
            parent_base = parent[:-1] if parent.endswith("-") else parent
            surface_suffix = child_base[len(parent_base):] if child_base.startswith(parent_base) else None
            chain_rev.append({
                "from": parent, "to": current, "type": typ, "type_raw": rel.get("type_raw"),
                "surface_suffix": surface_suffix, "exact_append": surface_suffix is not None,
                "status": "verified_krps_relation"
            })
            current = parent
        chain = list(reversed(chain_rev))
        if chain:
            return chain
        # If KRPS has no explicit forma_gramatyczna relation, expose at most one
        # conservative productive voice hypothesis when BOTH parent and child
        # are independently attested verbal headwords. This is not promoted to
        # a verified KRPS relation.
        child_base = headword[:-1] if headword.endswith("-") else headword
        voice_forms = globals().get("VOICE_FORMS", {})
        best = None
        for morpheme, forms in voice_forms.items():
            for suff in sorted(set(forms), key=len, reverse=True):
                if child_base.endswith(suff) and len(child_base) > len(suff) + 1:
                    parent = child_base[:-len(suff)] + "-"
                    pe = self.entries.get(parent)
                    ce = self.entries.get(headword)
                    if pe and ce and parent.endswith("-"):
                        if pe.dialects and ce.dialects and not (set(pe.dialects) & set(ce.dialects)):
                            continue
                        inferred_type = {
                            "VOICE.PASS": "passive",
                            "VOICE.REFL": "reflexive",
                            "VOICE.RECIP": "reciprocal",
                            "VOICE.CAUS": "causative",
                        }.get(morpheme, morpheme.replace("VOICE.", "").lower())
                        cand={"from":parent,"to":headword,"type":inferred_type,
                              "type_raw":None,"surface_suffix":suff,"exact_append":True,
                              "status":"probable_productive_voice_inference"}
                        if best is None or len(suff) > len(best["surface_suffix"]):
                            best=cand
        return [best] if best else []

    def stats(self) -> Dict[str, int]:
        relation_counts: Dict[str, int] = {}
        for e in self.entries.values():
            for gf in e.grammatical_forms:
                if isinstance(gf, dict) and gf.get("type"):
                    relation_counts[gf["type"]] = relation_counts.get(gf["type"], 0) + 1
        return {
            "unique_headwords": len(self.entries),
            "atomic_nominal_stem_candidates": len(self.atomic_stems),
            "verb_headwords": len(self.verb_headwords),
            "headwords_with_dialect": sum(bool(e.dialects) for e in self.entries.values()),
            "headwords_with_grammatical_relation": sum(bool(e.grammatical_forms) for e in self.entries.values()),
            "grammatical_relation_rows": sum(relation_counts.values()),
            "source_kind_KRPS": sum(e.source_kind == "KRPS" for e in self.entries.values()),
            "source_kind_external_vetted": sum(e.source_kind == "external_vetted" for e in self.entries.values()),
            "source_kind_supplemental_attested": sum(e.source_kind == "supplemental_attested" for e in self.entries.values()),
            **{"relation_" + k: v for k, v in sorted(relation_counts.items())},
        }

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    text: str
    role: str
    morpheme: Optional[str] = None
    status: str = "rule"


@dataclass
class Candidate:
    stem_surface: str
    segments: List[Segment]
    score: float
    confidence: str
    notes: List[str] = field(default_factory=list)
    lemma_restore_candidates: List[str] = field(default_factory=list)
    dialect_profiles: List[str] = field(default_factory=lambda: list(DIALECT_ALL))
    resolved_lemma: Optional[str] = None
    lemma_dialects: List[str] = field(default_factory=list)
    lexicon_match_type: Optional[str] = None
    lexicon_source_rows: List[int] = field(default_factory=list)
    analysis_domain: str = "nominal"
    paradigm: Optional[str] = None
    derivational_chain: List[Dict[str, Any]] = field(default_factory=list)
    source_support: List[str] = field(default_factory=list)

    def category_sequence(self) -> List[str]:
        return [s.morpheme for s in self.segments if s.morpheme]

    def segmented(self) -> str:
        return " + ".join(s.text for s in self.segments if s.text)


@dataclass
class TokenResult:
    line: Optional[int]
    surface_original: str
    surface_canonical: str
    token_class: str
    decision: str
    best: Optional[Candidate]
    alternatives: List[Candidate]
    experimental_derivation: List[Dict[str, str]]
    history_ref: Optional[str] = None
    history_class: Optional[str] = None
    history_confidence: Optional[str] = None
    history_context_required: bool = False
    history_source_ids: List[str] = field(default_factory=list)
    requested_dialects: List[str] = field(default_factory=list)
    dialect_constraint_status: str = "not_applied_unknown"
    rejected_by_dialect_count: int = 0
    lexicon_constraint_status: str = "not_applied"
    rejected_by_lexicon_count: int = 0

# ---------------------------------------------------------------------------
# Helpers / detectors
# ---------------------------------------------------------------------------

def endswith_any(s: str, forms: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for f in sorted(set(forms), key=len, reverse=True):
        if s.endswith(f) and len(s) > len(f):
            yield s[:-len(f)], f


def intersect_profiles(*profiles: Iterable[str]) -> List[str]:
    out = set(DIALECT_ALL)
    for p in profiles:
        out &= set(p)
    return [d for d in DIALECT_ALL if d in out]


def restore_lemma_candidates(stem_surface: str, following_starts_with_vowel: bool) -> List[str]:
    """Licensed stem restorations, all gated by the hard lexicon.

    v0.6 adds palatalization-sign restoration and a wider final-obstruent
    reversal. These are candidate generators only: a restoration survives only
    when the resulting lemma exists in the lexicon and is dialect-compatible.
    """
    out: List[str] = []
    if following_starts_with_vowel:
        if stem_surface.endswith("гъ"):
            out.extend([stem_surface[:-2] + "къ", stem_surface[:-2] + "к", stem_surface[:-2] + "х"])
        if stem_surface.endswith("г"):
            out.append(stem_surface[:-1] + "к")
        if stem_surface.endswith("д"):
            out.append(stem_surface[:-1] + "т")
    # Soft sign may be present in the dictionary lemma but absent before an
    # inflectional suffix in the surface representation. Hard lexicon gating
    # prevents this from acting as free spelling repair.
    if stem_surface and not stem_surface.endswith("ь"):
        out.append(stem_surface + "ь")
    # A small structurally motivated internal-palatalization candidate, e.g.
    # эрянь -> эрьянь. Again this is accepted only through exact lexicon match.
    for i, ch in enumerate(stem_surface):
        if ch in "яюё" and i > 0 and stem_surface[i-1] != "ь":
            out.append(stem_surface[:i] + "ь" + stem_surface[i:])
    # High-vowel syncope in kinship/body-type stems, e.g. огъл- < огъыл/огъул.
    if stem_surface.endswith("гъл"):
        base=stem_surface[:-1]
        out.extend([base + "ыл", base + "ул"])
    return list(dict.fromkeys(x for x in out if x and x != stem_surface))


def detect_plural(s: str) -> List[Tuple[str, Segment, float, List[str]]]:
    out = []
    for stem, form in endswith_any(s, PL_FORMS):
        hs, hn = suffix_harmony_score(stem, form, "low")
        out.append((stem, Segment(form, "plural", "PL"), 1.6 + hs, [hn]))
    return out


def detect_possessive(s: str) -> List[Tuple[str, Segment, float, List[str], List[str]]]:
    out = []
    post_short = {"м", "н", "й", "мыз", "миз", "муз", "мӱз", "мюз", "ныз", "низ", "нуз", "нӱз", "йыз", "йиз", "йуз", "йӱз"}
    for cat, forms in POSS_FORMS.items():
        for stem, form in endswith_any(s, forms):
            base_cat = "POSS.3" if cat.startswith("POSS.3") else ("POSS.2SG" if cat.startswith("POSS.2SG") else cat)
            score, notes = 1.55, []
            profiles = list(POSS_DIALECTS[cat])
            if form in post_short:
                if final_segment_class(stem) == "vowel":
                    score += 0.2; notes.append("postvocalic_possessive_allomorph")
                else:
                    score -= 1.4; notes.append("postvocalic_possessive_allomorph_after_nonvowel")
            if cat == "POSS.3.BUFFER_S":
                if final_segment_class(stem) == "vowel":
                    score += 0.8; notes.append("p3_buffer_s_after_vowel")
                else:
                    score -= 1.0; notes.append("unexpected_p3_buffer_s")
            elif cat == "POSS.3":
                score -= 0.35
                if final_segment_class(stem) != "vowel":
                    score += 0.35; notes.append("p3_high_vowel_after_consonant")
                else:
                    score -= 0.8; notes.append("p3_expected_buffer_s_after_vowel")
            if cat.startswith("POSS.2PL") and len(stem) < 2:
                continue
            if any(ch in ALL_VOWELS for ch in form):
                hs, hn = suffix_harmony_score(stem, form, "high")
                score += hs; notes.append(hn)
            else:
                notes.append("no_overt_harmonic_vowel_in_suffix")
            out.append((stem, Segment(form, "possessive", base_cat), score, notes, profiles))
    return out


def detect_regular_case(s: str) -> List[Tuple[str, Segment, float, List[str], List[str]]]:
    out = []
    for cat, forms in CASE_FORMS.items():
        for form, hk in sorted(forms, key=lambda x: len(x[0]), reverse=True):
            if s.endswith(form) and len(s) > len(form):
                stem = s[:-len(form)]
                hs, hn = suffix_harmony_score(stem, form, hk)
                os, on = _case_onset_score(stem, form, cat)
                notes = [hn] + ([on] if on else [])
                out.append((stem, Segment(form, "case", cat), 1.65 + hs + os, notes, list(CASE_DIALECTS[cat])))
    return out


def detect_p3_special_case(s: str) -> List[Tuple[str, List[Segment], float, List[str], List[str]]]:
    out = []
    for cat, endings in P3_CASE_COMBINED.items():
        for ending, linker, case_surface in sorted(endings, key=lambda x: len(x[0]), reverse=True):
            if not s.endswith(ending) or len(s) <= len(ending) + 1:
                continue
            before = s[:-len(ending)]
            for stem, pseg, pscore, pnotes, profiles in detect_possessive(before):
                if pseg.morpheme != "POSS.3":
                    continue
                segs = [pseg]
                if linker:
                    if cat == "CASE.ACC":
                        segs.append(Segment(linker, "case", "CASE.ACC"))
                    else:
                        segs.append(Segment(linker, "linker", "LINK.N_AFTER_POSS3"))
                if case_surface:
                    segs.append(Segment(case_surface, "case", cat))
                score = pscore + 2.3
                notes = pnotes + ["p3_possessive_case_special"]
                if case_surface:
                    hk = "low" if cat in {"CASE.DAT", "CASE.LOC", "CASE.ABL"} else "high"
                    hs, hn = suffix_harmony_score(stem, case_surface, hk)
                    score += hs; notes.append(hn)
                out.append((stem, segs, score, notes, profiles))
    return out


def detect_non3poss_dative(s: str) -> List[Tuple[str, List[Segment], float, List[str], List[str]]]:
    out = []
    for form, hk in POSS_NON3_DAT:
        if not s.endswith(form) or len(s) <= len(form) + 2:
            continue
        before = s[:-len(form)]
        for stem, pseg, pscore, pnotes, profiles in detect_possessive(before):
            if pseg.morpheme == "POSS.3":
                continue
            hs, hn = suffix_harmony_score(stem, form, hk)
            out.append((stem, [pseg, Segment(form, "case", "CASE.DAT")], pscore + 1.9 + hs,
                        pnotes + [hn, "dative_after_non3_possessive_reduced_onset"], profiles))
    return out


def add_plural_prefix_to_analysis(stem_before_suffixes: str, suffix_segments: List[Segment], base_score: float,
                                  notes: List[str], profiles: List[str]) -> List[Candidate]:
    cands: List[Candidate] = []
    segs = [Segment(stem_before_suffixes, "stem")] + suffix_segments
    first_suffix = suffix_segments[0].text if suffix_segments else ""
    restores = restore_lemma_candidates(stem_before_suffixes, bool(first_suffix and first_suffix[0] in ALL_VOWELS))
    cands.append(Candidate(stem_before_suffixes, segs, base_score, "", list(notes), restores, list(profiles)))
    for stem, plseg, plscore, plnotes in detect_plural(stem_before_suffixes):
        cands.append(Candidate(stem, [Segment(stem, "stem"), plseg] + suffix_segments,
                               base_score + plscore + 1.2, "",
                               notes + plnotes + ["plural_before_following_inflection"],
                               restore_lemma_candidates(stem, False), list(profiles)))
    return cands


def experimental_derivation(stem: str) -> List[Dict[str, str]]:
    out = []
    for cat, forms in DERIV_FORMS.items():
        for base, f in endswith_any(stem, forms):
            if len(base) >= 2:
                out.append({"base_candidate": base, "suffix": f, "morpheme": cat, "status": "experimental_needs_lexical_validation"})
    out.sort(key=lambda x: len(x["suffix"]), reverse=True)
    return out[:5]


def verbal_class(token: str) -> Optional[str]:
    if token.endswith("-"):
        return "VERBAL_LEMMA_STEM"
    if any(token.endswith(x) for x in VERB_INF_ENDINGS):
        return "VERBAL_INFINITIVE"
    if any(token.endswith(x) for x in VERB_FINITE_ENDINGS):
        return "VERBAL_FINITE_OR_PARTICIPIAL"
    if any(token.endswith(x) for x in VERB_PART_ENDINGS):
        return "VERBAL_PARTICIPLE"
    return None


# ---------------------------------------------------------------------------
# v0.5 productive verbal morphology
# ---------------------------------------------------------------------------
# Source status:
#   CSATO2012_T = explicitly described for Trakai/Lithuanian Karaim.
#   ILRAN        = cross-dialect overview (four voices; 3 past categories;
#                  present, future; -GAn and -A/jdoğon participles).
#   COMPARATIVE  = comparative H/T/K simple-past paradigms summarized in
#                  recent scholarship based on Musaev/Prik/Pritsak.
#   HIST_W       = historical Western Karaim constructions (Németh).

PAST_DI = ["ды", "ди", "ду", "дю", "дӱ", "ты", "ти", "ту", "тю", "тӱ"]
COND_SA = ["са", "сэ", "се", "ся"]
NEG_MA = ["ма", "мэ", "ме", "мя"]
NEG_PRES = ["мы", "ми", "му", "мӱ", "мый", "мий", "муй", "мӱй", "мюй", "май", "мэй", "мей"]
PRESENT_A = ["а", "э", "е", "я"]
PRESENT_J = ["й"]
AOR_FULL = ["ар", "эр", "ер", "яр", "ыр", "ир", "ур", "ӱр", "юр", "р"]
AOR_SHORT = ["ы", "и", "у", "ӱ", "ю"]
NEG_AOR_3 = ["мас", "мэс", "мес", "маз", "мэз", "мез"]
OPT_GAY = ["гъай", "гъэй", "гъей", "гай", "гей", "гэй", "къай", "къэй", "къей", "кай", "кей", "кэй", "хай", "хэй"]
# Enclitic interrogative particle. Western Karaim sources attest mo/me/mia;
# Eastern/Crimean sources primarily use mI. It is attached only after an
# independently valid finite verbal analysis.
QUESTION_PARTICLES = [("мо", ["H","T"]), ("ме", ["H","T"]), ("мэ", ["H","T"]), ("мя", ["T"]),
                      ("мы", ["K"]), ("ми", ["K"])]
VOL_1SG = ["айым", "эйим", "ейим", "яйым", "яйим", "айын", "эйин", "ейин", "йын", "йин",
           "айим", "эим", "йим"]
VOL_1PL = ["айык", "эйик", "ейик", "айых", "эйих"]
VOL_3SG = ["сын", "син", "сун", "сӱн", "сюн", "сынь", "синь", "сӱнь"]
VOL_3PL = [x+y for x in VOL_3SG for y in ["лар", "лер", "ляр", "лэр"]]
IMP_GIN = ["гъын", "гъун", "гын", "гин", "гинь", "гӱн", "гун", "къын", "кын", "кин", "кинь", "кӱн", "кун", "хын", "хин", "чун", "чин"]
IMP_2PL = ["ыйыз", "ийиз", "уйуз", "ӱйӱз", "юйюз", "йыз", "йиз", "йуз", "йӱз",
           "ыныз", "иниз", "унуз", "ӱнӱз", "юнюз", "ныз", "низ", "нуз", "нӱз",
           "ынъыз", "инъиз", "унъуз", "ӱнъӱз", "нъыз", "нъиз"]
POTENTIAL = ["ал", "ял", "эл", "ел"]

INF_MAK = ["мак", "макъ", "мэк", "мек", "мах", "мях"]
INF_MAK_OBLIQUE = ["маг", "мэг", "мег", "магъ", "мэгъ", "мегъ", "мяг", "мягъ"]
VN_MA = ["ма", "мэ", "ме", "мя"]
VN_UV_VERB = ["ув", "ӱв", "ив", "ыв", "юв"]
VN_ISH_VERB = ["ыш", "иш", "уш", "ӱш", "юш"]
PTCP_GAN = ["гъан", "гъэн", "ган", "гэн", "ген", "гян", "гянь", "къан", "кан", "кэн", "кен", "кян", "кянь", "хан", "хян"]
PTCP_PRESENT = [
    "адогъон", "эдогъон", "едогъон", "ядогъон", "йдогъон",
    "адогъан", "эдогъан", "едогъан", "ядогъан", "йдогъан",
    "адагъан", "эдагъан", "едагъан", "ядаган", "йдаган",
    "адогон", "эдогон", "ядогон", "йдогон"
]
PTCP_UVCI = [
    "увчу", "увчы", "ӱвчӱ", "ӱвчи", "ивчи", "ивци", "увцу", "ӱвцӱ",
    "увджы", "ӱвджи", "ювчу", "ювчю", "ивчю",
    # v1.1: corpus-attested reduced/high-vowel allomorphs. These remain
    # lexicon-gated by the underlying KRPS verb root.
    "ывчу", "ывчы", "ивчу", "ивчӱ", "вчу", "вчи", "вци", "вцу", "вджы", "ивджи"
]
PTCP_MISH = ["мыш", "миш", "муш", "мӱш", "мюс", "мус"]
CVB_IP = ["ып", "ип", "уп", "ӱп", "юп", "п"]
CVB_IPTA = ["ыпта", "иптэ", "упта", "ӱптэ", "юптя", "ыптя", "иптя"]
CVB_NEG_MAY = ["май", "мэй", "мей"]
CVB_NEG_MASTAN = ["мастан", "мэстэн", "местен", "мастын", "мэстин"]
CVB_GANDA = ["гъанда", "ганда", "гэндэ", "гендэ", "гяньдя", "къанда", "канда", "кэндэ", "кендэ", "кяньдя"]
# v1.0 source- and corpus-grounded additional non-finite inventories.
NEG_INF_WEST = ["масха", "мэсхэ", "месхе", "маска", "мэскэ", "меске"]
CVB_NEG_MAYIN = ["майын", "мэйин", "мейин"]
CVB_NEG_MAYINCHA = ["майынча", "мэйинчэ", "мейинче"]
CVB_DOGAC = ["адогъац", "эдогъац", "едогъац", "ядогъац", "йдогъац",
             "адогъач", "эдогъач", "едогъач", "ядогъач", "йдогъач"]
CVB_GINCHA = ["гъынча", "гъунча", "гынча", "гинча", "гиньча", "гӱнча",
              "къынча", "кынча", "кинча", "киньча", "кӱнча", "хынча", "хинча"]

# v1.1 outer morphology licensed after non-finite forms. These markers are
# deliberately narrow: they are only considered after a valid KRPS verbal root
# and a recognized non-finite core.
NONFINITE_ADV_LEY = ["лей", "лэй"]
NONFINITE_LIMITATIVE = ["чакъ", "чаз"]

# v1.1 audited single-segment voice bridges. One-letter voice/causative
# morphology is too ambiguous for a global productive fallback, so only these
# corpus-attested derived stems are licensed.
AUDITED_SINGLE_VOICE_DERIV = {
    "йувл": ("йув", "л", "VOICE.PASS", ["K"]),
    "саклан": ("сакла", "н", "VOICE.REFL", ["H","T","K"]),
    "сакланъ": ("сакла", "нъ", "VOICE.REFL", ["H","T","K"]),
    "атлангъыздырт": ("атлангъыздыр", "т", "VOICE.CAUS", ["H","T","K"]),
    "секиргяле": ("секир", "гяле", "ASP.ITER", ["T"]),
}

HISTORICAL_PROFILES = {"historical_western"}
HIST_W_3PL_D = ["дляр", "длар", "длер", "длэр"]
HIST_W_AUX_3SG = ["эдь", "едь", "өдь", "эд", "ед", "ид"]
HIST_W_AUX_3PL = [a+p for a in ["эд", "ед", "иред", "ирэд"] for p in ["ляр","лер","лэр","лар"]]
HIST_W_PERSON_1SG = ["мень", "минь", "мин"]
HIST_W_PERSON_1PL = ["бизь"]
HIST_W_NEG_MAS = ["мясь", "месь"]
HIST_W_NEG_INF = ["мяська", "мяськя", "мяськэ", "мясьха", "меське", "меськэ"]
# Source-attested contracted conditional surfaces are kept as portmanteaux;
# forcing a finer boundary would overstate what can be recovered from spelling.
HIST_W_COND_PORTMANTEAU = [
    "сэйдыр", "сейдир", "сыйдныз", "сейдыйыз", "сады", "сайэди",
    "сэй", "сейль", "сэйдый", "сай", "сыйд", "сэйд", "сейд"
]

COP_PAST_3SG = ["эди", "еди", "иди", "эд", "ед", "ид", "эди́"]
COP_PAST_PERSONAL = [
    ("PERS.1SG", ["эдим", "едим", "идим"], ["H","T","K"]),
    ("PERS.2SG", ["эдий", "едий", "идий"], ["T"]),
    ("PERS.2SG", ["эдин", "един", "идин"], ["H","K"]),
    ("PERS.1PL", ["эдик", "едик", "идик", "эдих", "едих"], ["H","T","K"]),
    ("PERS.2PL", ["эдийиз", "едийиз", "идийиз"], ["T"]),
    ("PERS.2PL", ["эдиниз", "единиз", "идиниз"], ["H","K"]),
    ("PERS.3PL", ["эдиляр", "едиляр", "идиляр", "эдилар", "едилар"], ["H","T","K"]),
]

# Type-I (pronominal/predicative) personal markers. The short Trakai forms are
# directly attested by Csató. H and K variants are kept conservative and only
# license patterns well-supported by KRPS/cross-dialect descriptions.
TYPE1_PERSON = [
    ("PERS.1PL", ["быз", "биз", "буз", "бӱз", "бюз"], ["H","T","K"]),
    ("PERS.2PL", ["сыз", "сиз", "суз", "сӱз", "сюз"], ["H","T","K"]),
    ("PERS.1SG", ["мын", "мин", "мун", "мӱн", "мюн"], ["T","K"]),
    ("PERS.2SG", ["сын", "син", "сун", "сӱн", "сюн"], ["T","K"]),
    ("PERS.1SG", ["мэн", "мен"], ["H"]),
    ("PERS.2SG", ["сэн", "сен"], ["H"]),
    ("PERS.1SG", ["м"], ["H","T","K"]),
    ("PERS.2SG", ["с"], ["H","T"]),
]

# Type-II endings used after -DI and -SA. Comparative sources establish the
# major H/T/K opposition: T 2nd person has -j, H has -n, Crimean has -ŋ.
TYPE2_PERSON = [
    ("PERS.2PL", ["йыз", "йиз", "йуз", "йӱз", "юйюз"], ["T"]),
    ("PERS.2PL", ["ныз", "низ", "нуз", "нӱз", "нюз"], ["H"]),
    ("PERS.2PL", ["нъыз", "нъиз", "нъуз", "нъӱз", "ныз", "низ", "нуз", "нӱз"], ["K"]),
    ("PERS.3PL", ["лар", "лер", "ляр", "лэр"], ["H","T","K"]),
    ("PERS.1PL", ["къ", "к", "х"], ["H","T","K"]),
    ("PERS.2SG", ["й"], ["T"]),
    ("PERS.2SG", ["н"], ["H"]),
    ("PERS.2SG", ["нъ", "нь", "н"], ["K"]),
    ("PERS.1SG", ["м"], ["H","T","K"]),
]

# Present third-person copular markers / contractions. Trakai -t and -dlar are
# explicit in Csató; Western historical texts additionally attest -dyr/-dy/-d.
PRES_3_MARKERS = [
    ("PERS.3PL", ["дырлар", "дирляр", "дирлэр", "дурлар", "дюрляр", "дылар", "диляр", "дилэр", "дулар", "дюляр", "длар", "длер", "длэр", "тлар", "тлер", "тлэр", "тилэр"], ["H","T"]),
    ("PERS.3PL", ["дырлар", "дирлер", "дурлар", "дюрлер", "дылар", "дилер", "дулар", "дюлер"], ["K"]),
    ("PERS.3SG", ["дыр", "дир", "дур", "дюр", "ды", "ди", "ду", "дю", "д", "т", "дь", "ть"], ["H","T"]),
    ("PERS.3SG", ["дыр", "дир", "дур", "дюр", "ды", "ди", "ду", "дю", "д", "т"], ["K"]),
]

NEG_AOR_3_MARKERS = [
    ("PERS.3PL", ["тылар", "тиляр", "тилэр", "тулар", "тюляр", "дылар", "диляр", "дилэр", "дулар", "дюляр", "тлар", "тлер", "тлэр"], ["H","T"]),
    ("PERS.3SG", ["тыр", "тир", "тур", "тюр", "дыр", "дир", "дур", "дюр", "ты", "ти", "ту", "тю", "т", "ды", "ди", "д"], ["H","T","K"]),
    ("PERS.3PL", ["лар", "лер", "ляр", "лэр"], ["K"]),
]

# Voice allomorphs are used mainly to expose a derivational layer. When KRPS
# already has a derived verbal headword, the explicit KRPS relation outranks
# productive inference.
VOICE_FORMS = {
    "VOICE.PASS": ["ыл", "ил", "ул", "ӱл", "юл", "л", "ын", "ин", "ун", "ӱн", "н"],
    "VOICE.REFL": ["ын", "ин", "ун", "ӱн", "н", "нь"],
    "VOICE.RECIP": ["ыш", "иш", "уш", "ӱш", "ш"],
    "VOICE.CAUS": ["дыр", "дир", "дур", "дюр", "тыр", "тир", "тур", "тюр", "т", "ыр", "ир", "ур", "ӱр", "гъыз", "гиз", "гӱз", "къыз", "киз"],
}
VOICE_TYPE_TO_MORPHEME = {"passive":"VOICE.PASS", "reflexive":"VOICE.REFL", "causative":"VOICE.CAUS", "reciprocal":"VOICE.RECIP"}


def _suffix_matches(s: str, forms: Iterable[str], allow_empty_stem: bool = False):
    for f in sorted(set(forms), key=len, reverse=True):
        if s.endswith(f) and (allow_empty_stem or len(s) > len(f)):
            yield s[:-len(f)] if f else s, f


def _person_strips(s: str, table, include_zero: bool = True):
    out = []
    for cat, forms, profiles in table:
        for base, form in _suffix_matches(s, forms):
            out.append((base, Segment(form, "person", cat), profiles))
    if include_zero:
        out.append((s, Segment("", "person", "PERS.3SG"), list(DIALECT_ALL)))
    return out


def _present_person_strips(s: str):
    out = []
    # ordinary 1st/2nd person
    out.extend(_person_strips(s, TYPE1_PERSON, include_zero=False))
    # third persons carry a copular/predicative marker in Trakai/Western forms
    for cat, forms, profiles in PRES_3_MARKERS:
        for base, form in _suffix_matches(s, forms):
            out.append((base, Segment(form, "person", cat), profiles))
    # Crimean Kipchak descriptions/texts also attest bare -A/-y in 3SG and
    # -A/-y + lAr in 3PL. The following entries only expose the outer person
    # layer; _finite_present_candidates still requires an overt PRESENT_A/J
    # and an exact hard-lexicon verbal root.
    out.append((s, Segment("", "person", "PERS.3SG"), ["K"]))
    for base, form in _suffix_matches(s, ["лар","лер","ляр","лэр"]):
        out.append((base, Segment(form, "person", "PERS.3PL"), ["K"]))
    return out


def _aorist_person_strips(s: str):
    out = _person_strips(s, TYPE1_PERSON, include_zero=True)
    # 3PL plain -lar after full r-marked aorist
    for base, form in _suffix_matches(s, ["лар","лер","ляр","лэр"]):
        out.append((base, Segment(form, "person", "PERS.3PL"), list(DIALECT_ALL)))
    return out
