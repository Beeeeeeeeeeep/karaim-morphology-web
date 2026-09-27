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


def _verb_restorations(stem_surface: str, following_starts_with_vowel: bool = True) -> List[str]:
    out = restore_lemma_candidates(stem_surface, following_starts_with_vowel)
    if following_starts_with_vowel:
        if stem_surface.endswith("б"):
            out.append(stem_surface[:-1] + "п")
        # In the KRPS Cyrillic representation, stem-final palatalization can be
        # overt in the dictionary headword but not repeated before a vowel-initial
        # suffix. This is a grammatical orthographic/morphophonological
        # restoration, not text normalization: it is accepted only when the
        # restored verbal headword exists in the hard KRPS lexicon.
        if stem_surface and not stem_surface.endswith("ь") and stem_surface[-1] in "лнртдсз":
            out.append(stem_surface + "ь")
        # The converse alternation is attested in the supplied Cyrillic examples
        # (surface stem-final ь before inflection vs. a KRPS headword without it).
        # It is only a candidate restoration and survives exclusively when the
        # exact unsoftened verb is present in the hard lexicon.
        if stem_surface.endswith("ь") and len(stem_surface) > 1:
            out.append(stem_surface[:-1])
        if stem_surface.endswith("з"):
            # only a weak option; kept after exact forms and never accepted unless KRPS confirms it
            pass
    # Before -j/-й a dictionary stem in -е can surface with -э in the KRPS
    # material. This is a lexical-gated morphophonological restoration.
    if stem_surface.endswith("э"):
        out.append(stem_surface[:-1] + "е")
    # v1.1 audited surface alternation from the supplied corpus. Do not
    # generalize final hard-sign deletion to arbitrary tokens.
    if stem_surface == "тынъ":
        out.append("тын")
    # Crimean Karaim texts attest Oghuz/Ottoman-type voicing of inherited
    # initial k- > g- alongside Kipchak k-.  This is NOT spelling repair:
    # it is a K-only lexical restoration and survives only if the restored
    # k-initial verb is an attested KRPS Crimean headword.  Restrict it to
    # front-vowel sequences, which is the environment needed by the v0.7
    # corpus case гэч- ~ кэч-.
    if len(stem_surface) >= 2 and stem_surface[0] == "г" and stem_surface[1] in "эеиӧӱю":
        out.append("к" + stem_surface[1:])
    return list(dict.fromkeys(out))


def _resolve_verb(c: Candidate, lexicon: KRPSLexicon, requested: Iterable[str]) -> bool:
    resolved = lexicon.resolve_verb_candidate(c.stem_surface, c.lemma_restore_candidates, requested)
    if not resolved:
        return False
    entry, match_type = resolved
    if entry.dialects:
        compatible = intersect_profiles(c.dialect_profiles, entry.dialects)
        if not compatible:
            return False
        c.dialect_profiles = compatible
    c.resolved_lemma = entry.headword
    c.lemma_dialects = list(entry.dialects)
    c.lexicon_match_type = match_type
    c.lexicon_source_rows = list(entry.source_rows)
    c.derivational_chain = lexicon.derivational_chain(entry.headword)
    c.notes.append("verb_lexicon_hard_match:" + match_type)
    if match_type == "morphophonological_restoration" and c.stem_surface.startswith("г") and entry.headword.startswith("к") and "K" in entry.dialects:
        c.notes.append("crimean_initial_k_to_g_variant_restored")
        if "MURAT_CRIMEAN_K_G" not in c.source_support:
            c.source_support.append("MURAT_CRIMEAN_K_G")
    c.score += 1.5 * len(entry.headword) + (2.0 if match_type == "exact_verb_stem" else 0.5)
    return True


def _expand_krps_derivation(c: Candidate) -> None:
    """Expose exact append-only KRPS derivation in surface segmentation.

    This never infers an internal boundary. It only expands a stem when the
    headword's forma_gramatyczna chain says parent -> child and child is exactly
    parent+suffix in the canonical spelling.
    """
    if not c.derivational_chain or not c.segments:
        return
    stem_seg = c.segments[0]
    if stem_seg.role not in {"stem","verbal_stem"}:
        return
    lemma = c.resolved_lemma or ""
    lemma_base = lemma[:-1] if lemma.endswith("-") else lemma
    if stem_seg.text != lemma_base:
        return
    chain = c.derivational_chain
    if not all(x.get("exact_append") for x in chain):
        return
    root = chain[0]["from"]
    root_base = root[:-1] if root.endswith("-") else root
    pieces = [Segment(root_base, "root", "ROOT", "krps_relation")]
    current = root_base
    for rel in chain:
        suff = rel.get("surface_suffix") or ""
        if not suff:
            return
        mor = VOICE_TYPE_TO_MORPHEME.get(rel.get("type"), "DERIV." + str(rel.get("type","UNKNOWN")).upper())
        pieces.append(Segment(suff, "derivational", mor, rel.get("status","krps_relation")))
        current += suff
    if current == lemma_base:
        c.segments = pieces + c.segments[1:]
        c.notes.append("krps_derivational_chain_expanded")


PRODUCTIVE_VERB_DERIV = [
    # Only relatively distinctive surface allomorphs are allowed in the
    # fallback resolver. One-letter voice markers (-l/-n/-t/-š) and high-vowel
    # causatives are too ambiguous next to TAM/person morphology; those remain
    # available when KRPS itself attests the derived headword/relation.
    ("VOICE.CAUS", ["дыр","дир","дур","дюр","тыр","тир","тур","тюр","гъыз","гиз","гӱз","къыз","киз"], ["H","T","K"], "productive_voice"),
    ("VOICE.PASS", ["ыл","ил","ул","ӱл","юл"], ["H","T","K"], "productive_voice"),
    ("VOICE.REFL", ["ын","ин","ун","ӱн","юнь","инь"], ["H","T","K"], "productive_voice"),
    ("VOICE.RECIP", ["ыш","иш","уш","ӱш"], ["H","T","K"], "productive_voice"),
    # Denominal/deverbal -lAn- is independently described for Western Karaim
    # and is also attested in Crimean derivational material. It is treated as a
    # productive derivational bridge, never as a lemma by itself.
    ("DERIV.V.LAN", ["лан", "лен", "лянь", "лян"], ["H","T","K"], "productive_derivation"),
]


def _resolve_productive_derived_verb(c: Candidate, lexicon: KRPSLexicon,
                                      requested: Iterable[str], max_depth: int = 2) -> bool:
    """Resolve ROOT + productive verbal derivation(s) + inflection.

    v0.5 required the *derived* verbal stem to be a KRPS headword. v0.6 keeps
    the hard lexicon but allows a licensed productive derivation when the
    underlying root verb is a KRPS headword. The surface boundaries are kept
    only when they are literal suffix boundaries.
    """
    start = c.stem_surface
    # state: current surface, stripped derivations from outermost to innermost
    stack=[(start, [])]
    seen={start}
    solutions=[]
    while stack:
        cur, stripped = stack.pop(0)
        if stripped:
            restorations=_verb_restorations(cur, True)
            resolved=lexicon.resolve_verb_candidate(cur, restorations, requested)
            root_kind="verb"
            # v0.7: productive -lAn- is also a denominal verbalizer.  When the
            # innermost stripped derivation is DERIV.V.LAN, a hard-matched KRPS
            # nominal lemma may license the derived verbal stem.  No arbitrary
            # string is accepted: the nominal root must itself be an atomic KRPS
            # headword (or a lexicon-gated morphophonological restoration).
            derivs_probe=list(reversed(stripped))
            if not resolved and derivs_probe and derivs_probe[0][0] == "DERIV.V.LAN":
                resolved=lexicon.resolve_candidate(cur, restore_lemma_candidates(cur, True), requested)
                root_kind="nominal" if resolved else "verb"
            if resolved:
                entry, mt=resolved
                # Derivations were stripped outermost-first; render inner->outer.
                derivs=derivs_probe
                profiles=list(DIALECT_ALL)
                for d in derivs:
                    profiles=intersect_profiles(profiles,d[2])
                if entry.dialects:
                    profiles=intersect_profiles(profiles,entry.dialects)
                if requested:
                    profiles=intersect_profiles(profiles,requested)
                if profiles:
                    solutions.append((entry,mt,cur,derivs,profiles,root_kind))
            if len(stripped) >= max_depth:
                continue
        for mor, forms, profiles, status in PRODUCTIVE_VERB_DERIV:
            for base, suff in _suffix_matches(cur, forms):
                if len(base) < 2:
                    continue
                key=(base, tuple((x[0],x[1]) for x in stripped+[(mor,suff,profiles,status)]))
                if key in seen:
                    continue
                seen.add(key)
                stack.append((base, stripped+[(mor,suff,profiles,status)]))
    if not solutions:
        return False
    # Prefer fewer derivational steps, then the longest resolved lexical root.
    solutions.sort(key=lambda x:(len(x[3]), -len(x[0].headword)))
    entry, mt, root_surface, derivs, profiles, root_kind=solutions[0]
    deriv_segments=[]
    chain=[]
    cur=root_surface
    for mor,suff,dprof,status in derivs:
        deriv_segments.append(Segment(suff,"derivational",mor,status))
        chain.append({"from":cur if root_kind == "nominal" and not chain else cur+"-","to":cur+suff+"-","type":mor.replace("VOICE.","").replace("DERIV.V.","").lower(),
                      "surface_suffix":suff,"exact_append":True,"status":"probable_productive_derivation"})
        cur += suff
    c.stem_surface=root_surface
    c.segments=[Segment(root_surface,"root","ROOT","lexicon_root")]+deriv_segments+c.segments[1:]
    c.lemma_restore_candidates=_verb_restorations(root_surface,True)
    c.resolved_lemma=entry.headword
    c.lemma_dialects=list(entry.dialects)
    c.lexicon_match_type=mt+("+denominal_productive_derivation" if root_kind == "nominal" else "+productive_derivation")
    c.lexicon_source_rows=list(entry.source_rows)
    c.derivational_chain=chain
    c.dialect_profiles=profiles
    c.notes.append("productive_derived_stem_resolved_to_krps_root")
    if root_kind == "nominal":
        c.notes.append("denominal_LAN_root_hard_matched_in_krps")
    c.source_support=list(dict.fromkeys(c.source_support+["MUSAEV_WESTERN","FOLTYN_DERIVATION","ILRAN_PRODUCTIVE_LAN","ISIK_CRIMEAN"]))
    c.score += 0.8*len(entry.headword)+0.5*len(derivs)
    return True


def _resolve_audited_single_voice(c: Candidate, lexicon: KRPSLexicon, requested: Iterable[str]) -> bool:
    spec = AUDITED_SINGLE_VOICE_DERIV.get(c.stem_surface)
    if not spec:
        return False
    root, suffix, morpheme, profiles = spec
    active = intersect_profiles(profiles, requested) if list(requested) else list(profiles)
    if not active:
        return False
    entry = lexicon.verb_bases.get(root)
    if not entry or not lexicon.dialect_compatible(entry, active):
        return False
    if entry.dialects:
        active = intersect_profiles(active, entry.dialects)
        if not active:
            return False
    old_tail = c.segments[1:]
    c.stem_surface = root
    c.segments = [Segment(root, "root", "ROOT", "lexicon_root"),
                  Segment(suffix, "derivational", morpheme, "audited_corpus_voice")] + old_tail
    c.resolved_lemma = entry.headword
    c.lemma_dialects = list(entry.dialects)
    c.lexicon_match_type = "audited_single_voice_to_exact_verb_root"
    c.lexicon_source_rows = list(entry.source_rows)
    c.dialect_profiles = active
    c.derivational_chain = [{"from": root+"-", "to": root+suffix+"-",
                             "type": morpheme.replace("VOICE.","").lower(),
                             "surface_suffix": suffix, "exact_append": True,
                             "status": "verified_corpus_whitelist"}]
    c.notes.append("audited_single_segment_voice_whitelist")
    c.source_support = list(dict.fromkeys(c.source_support + ["KRPS_GLOSSES_2026"]))
    c.score += 1.5 * len(entry.headword) + 1.0
    return True


def _make_verb_candidate(stem: str, morph_segments: List[Segment], score: float,
                         profiles: Iterable[str], paradigm: str, notes: List[str],
                         source_support: List[str], lexicon: KRPSLexicon,
                         requested: Iterable[str], following_vowel: bool = True) -> Optional[Candidate]:
    profiles2 = intersect_profiles(profiles, requested) if list(requested) else [d for d in DIALECT_ALL if d in set(profiles)]
    if list(requested) and not profiles2:
        return None
    if not profiles2:
        profiles2 = list(profiles)
    c = Candidate(
        stem_surface=stem,
        segments=[Segment(stem, "verbal_stem")] + morph_segments,
        score=score, confidence="", notes=list(notes),
        lemma_restore_candidates=_verb_restorations(stem, following_vowel),
        dialect_profiles=profiles2,
        analysis_domain="verbal", paradigm=paradigm, source_support=source_support,
    )
    if not _resolve_verb(c, lexicon, requested):
        # v1.1: narrowly licensed one-segment voice/causative bridges from the
        # audited corpus. This runs before the broader productive derivation.
        if _resolve_audited_single_voice(c, lexicon, requested):
            return c
        # v0.6: a productive verbal derivation may intervene between a KRPS
        # lexical root and TAM/non-finite morphology.
        if _resolve_productive_derived_verb(c, lexicon, requested):
            return c
        # Synthetic potential -(y)Al- is grammatical rather than lexical; allow
        # stripping it once before the hard lexicon check.
        pot = next(((base, f) for base, f in _suffix_matches(stem, POTENTIAL)), None)
        if pot:
            base, f = pot
            c.stem_surface = base
            c.segments = [Segment(base, "verbal_stem"), Segment(f, "modality", "MOD.POT")] + morph_segments
            c.lemma_restore_candidates = _verb_restorations(base, True)
            c.notes.append("synthetic_potential_postverb")
            if not _resolve_verb(c, lexicon, requested):
                return None
        else:
            return None
    _expand_krps_derivation(c)
    return c


def _finite_past_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    # positive and NEG + DI
    for base_after_person, pseg, pprof in _person_strips(token, TYPE2_PERSON, include_zero=True):
        for before_di, di in _suffix_matches(base_after_person, PAST_DI):
            # positive
            c=_make_verb_candidate(before_di,[Segment(di,"tense","TENSE.PAST.DI"),pseg],7.0,pprof,
                                   "IND.PAST.DI",["simple_past"],["CSATO2012_T","COMPARATIVE_H_T_K"],lexicon,requested,True)
            if c: out.append(c)
            # negative: stem + MA + DI
            for stem, neg in _suffix_matches(before_di, NEG_MA):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(di,"tense","TENSE.PAST.DI"),pseg],8.0,pprof,
                                       "IND.PAST.DI.NEG",["negative_before_past"],["CSATO2012_T","ILRAN"],lexicon,requested,True)
                if c: out.append(c)
    return out


def _finite_present_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    for before_person,pseg,pprof in _present_person_strips(token):
        # affirmative -A after consonant, -j after vowel
        for before_tm, tm in _suffix_matches(before_person, PRESENT_A + PRESENT_J):
            bare_k3 = (pseg.morpheme == "PERS.3SG" and pseg.text == "" and pprof == ["K"])
            pres_score = 3.9 if bare_k3 else 6.8
            notes = ["present_A_or_j"] + (["crimean_bare_3sg_surface_ambiguous"] if bare_k3 else [])
            support = ["CSATO2012_T","ILRAN"] + (["ISIK_CRIMEAN_PRESENT_TABLE"] if bare_k3 else [])
            c=_make_verb_candidate(before_tm,[Segment(tm,"tense","TENSE.PRES"),pseg],pres_score,pprof,
                                   "IND.PRESENT",notes,support,lexicon,requested,True)
            if c: out.append(c)
        # contracted negative present: -mI / -mAy + person.  Do not use the
        # newly added Crimean zero-3SG present slot here: bare -mAy is also a
        # core negative converb and must keep the established single-token
        # reading unless an overt present/person marker disambiguates it.
        if not (pseg.morpheme == "PERS.3SG" and pseg.text == "" and pprof == ["K"]):
            for stem, negpr in _suffix_matches(before_person, NEG_PRES):
                c=_make_verb_candidate(stem,[Segment(negpr,"negation","NEG.PRES"),pseg],7.2,pprof,
                                       "IND.PRESENT.NEG",["negative_present_portmanteau"],["CSATO2012_T","HIST_W"],lexicon,requested,True)
                if c: out.append(c)
    # v1.1 corpus-supported Western/H form ROOT + -mAy + -sIn.  H normally
    # uses -sen in the canonical Type-I table, so this is kept as a narrow
    # source-conditioned exception rather than changing TYPE1_PERSON globally.
    for pform in ["сын", "син"]:
        for before_p,_ in _suffix_matches(token,[pform]):
            for stem,negpr in _suffix_matches(before_p,["май","мэй","мей"]):
                c=_make_verb_candidate(stem,[Segment(negpr,"negation","NEG.PRES"),Segment(pform,"person","PERS.2SG")],
                                       8.1,["H"],"IND.PRESENT.NEG",["western_H_mAy_sIn_corpus_variant"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)
    return out


def _finite_aorist_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    for before_person,pseg,pprof in _aorist_person_strips(token):
        # positive full aorist/future
        for stem,aor in _suffix_matches(before_person,AOR_FULL):
            c=_make_verb_candidate(stem,[Segment(aor,"tense","TENSE.AOR.FUT"),pseg],6.4,pprof,
                                   "IND.AOR_FUT",["aorist_functions_as_future"],["CSATO2012_T","ILRAN"],lexicon,requested,True)
            if c: out.append(c)
        # short high-vowel aorist before 1/2-person marker (e.g. al-y-m)
        if pseg.morpheme not in {"PERS.3SG","PERS.3PL"}:
            for stem,aor in _suffix_matches(before_person,AOR_SHORT):
                c=_make_verb_candidate(stem,[Segment(aor,"tense","TENSE.AOR.FUT"),pseg],6.5,pprof,
                                       "IND.AOR_FUT",["short_high_vowel_aorist_before_person"],["CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
        # negative 1/2 person: -MA + person, no overt r
        if pseg.morpheme in {"PERS.1SG","PERS.2SG","PERS.1PL","PERS.2PL"}:
            for stem,neg in _suffix_matches(before_person,NEG_MA):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment("","tense","TENSE.AOR.FUT.NEG"),pseg],6.8,pprof,
                                       "IND.AOR_FUT.NEG",["negative_aorist_r_absent_in_1_2_person"],["CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
    # Negative 3rd-person forms: -mAs-t / -mAs-tlar in Trakai/Western, plain -mAs in broader Turkic/K.
    for pcat,pforms,pprof in NEG_AOR_3_MARKERS:
        for before_p,pform in _suffix_matches(token,pforms):
            for stem,neg3 in _suffix_matches(before_p,NEG_AOR_3):
                c=_make_verb_candidate(stem,[Segment(neg3,"negation","NEG+AOR"),Segment(pform,"person",pcat)],7.0,pprof,
                                       "IND.AOR_FUT.NEG",["negative_aorist_3rd"],["CSATO2012_T","COMPARATIVE_H_T_K"],lexicon,requested,True)
                if c: out.append(c)
    # Crimean/plain 3SG -mAs
    for stem,neg3 in _suffix_matches(token,NEG_AOR_3):
        c=_make_verb_candidate(stem,[Segment(neg3,"negation","NEG+AOR"),Segment("","person","PERS.3SG")],6.0,["H","K"],
                               "IND.AOR_FUT.NEG",["plain_negative_aorist_3sg"],["COMPARATIVE_H_T_K"],lexicon,requested,True)
        if c: out.append(c)

    # v1.1 corpus example: r-final Trakai stem + Western-shaped 1SG -мэн.
    # Keep this as a narrow source-conditioned mixed ending, not a TYPE1 change.
    for before_p,pform in _suffix_matches(token,["мэн","мен"]):
        if before_p.endswith("р"):
            c=_make_verb_candidate(before_p,[Segment("","tense","TENSE.AOR.FUT","haplology_after_r"),Segment(pform,"person","PERS.1SG")],
                                   6.2,["T"],"IND.AOR_FUT",["aorist_r_haplology_after_r_final_stem","source_conditioned_T_men"],
                                   ["KRPS_GLOSSES_2026"],lexicon,requested,False)
            if c: out.append(c)

    # v1.1: lexicon-gated aorist haplology after stems already ending in -r.
    # Restrict to Trakai pending broader source support.
    for before_person,pseg,pprof in _person_strips(token,TYPE1_PERSON,include_zero=False):
        if pseg.morpheme in {"PERS.1SG","PERS.2SG","PERS.1PL","PERS.2PL"} and before_person.endswith("р"):
            c=_make_verb_candidate(before_person,[Segment("","tense","TENSE.AOR.FUT", "haplology_after_r"),pseg],
                                   5.9,intersect_profiles(pprof,["T"]),"IND.AOR_FUT",
                                   ["aorist_r_haplology_after_r_final_stem"],["KRPS_GLOSSES_2026"],lexicon,requested,False)
            if c: out.append(c)
    return out


def _compound_past_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    """Recognize one-token/fused reflexes of analytical past constructions.

    Spaced constructions such as `al-yr e-di-m` are represented in the spec but
    require sequence-level parsing; this word parser only handles fused spellings.
    """
    out=[]
    endings=[("PERS.3SG",x,["H","T","K"]) for x in COP_PAST_3SG]
    endings += [(cat,x,prof) for cat,forms,prof in COP_PAST_PERSONAL for x in forms]
    for pcat,auxsurf,prof in sorted(endings,key=lambda z:len(z[1]),reverse=True):
        for pre,_ in _suffix_matches(token,[auxsurf]):
            pseg=Segment("","person",pcat)
            aux=Segment(auxsurf,"auxiliary","AUX.COP.PAST")
            # Aorist + edi: intraterminal / habitual past
            for stem,aor in _suffix_matches(pre,AOR_FULL):
                c=_make_verb_candidate(stem,[Segment(aor,"tense","TENSE.AOR"),aux,pseg],7.3,prof,
                                       "IND.PAST.INTRATERMINAL",["fused_aorist_EDI"],["CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
            # -GAn edi pluperfect I
            for stem,gan in _suffix_matches(pre,PTCP_GAN):
                c=_make_verb_candidate(stem,[Segment(gan,"participle","PTCP.PAST.GAN"),aux,pseg],7.4,prof,
                                       "IND.PLUPERFECT.GAN_EDI",["fused_GAN_EDI"],["CSATO2012_T","HIST_W"],lexicon,requested,True)
                if c: out.append(c)
    return out

def _conditional_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    for before_person,pseg,pprof in _person_strips(token,TYPE2_PERSON,include_zero=True):
        for stem,sa in _suffix_matches(before_person,COND_SA):
            c=_make_verb_candidate(stem,[Segment(sa,"mood","MOOD.COND"),pseg],6.8,pprof,
                                   "MOOD.CONDITIONAL",["hypothetical_SA"],["CSATO2012_T","ILRAN"],lexicon,requested,True)
            if c: out.append(c)
            for negstem,neg in _suffix_matches(stem,NEG_MA):
                c=_make_verb_candidate(negstem,[Segment(neg,"negation","NEG"),Segment(sa,"mood","MOOD.COND"),pseg],7.2,pprof,
                                       "MOOD.CONDITIONAL.NEG",[],["CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
    return out


def _imperative_vol_opt_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    # Imperative 2SG bare stem.
    c=_make_verb_candidate(token,[Segment("","mood","MOOD.IMP"),Segment("","person","PERS.2SG")],3.0,["H","T","K"],
                           "MOOD.IMPERATIVE.2SG",["bare_stem_imperative"],["CSATO2012_T","ILRAN"],lexicon,requested,False)
    if c: out.append(c)

    # Elaborated 2SG -GIN. Trakai is directly described by Csató; the supplied
    # KRPS examples and Eastern historical descriptions also license H/K.
    for stem,form in _suffix_matches(token,IMP_GIN):
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.IMP"),Segment("","person","PERS.2SG")],6.4,["H","T","K"],
                               "MOOD.IMPERATIVE.2SG",["elaborated_GIN_imperative"],["CSATO2012_T","NEMETH_EASTERN","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)
        # Negative imperative: ROOT + -MA + -GIN.  The positive candidate above
        # cannot replace this analysis because the negative boundary is explicit.
        for nstem,neg in _suffix_matches(stem,NEG_MA):
            c=_make_verb_candidate(nstem,[Segment(neg,"negation","NEG"),Segment(form,"mood","MOOD.IMP"),Segment("","person","PERS.2SG")],
                                   8.9,["H","T","K"],"MOOD.IMPERATIVE.2SG.NEG",["negative_GIN_imperative"],
                                   ["KRPS_GLOSSES_2026","CSATO2012_T","NEMETH_EASTERN"],lexicon,requested,True)
            if c: out.append(c)

    for stem,form in _suffix_matches(token,IMP_2PL):
        profiles=["T"] if "й" in form else ["H","K"]
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.IMP"),Segment("","person","PERS.2PL")],6.2,profiles,
                               "MOOD.IMPERATIVE.2PL",[],["CSATO2012_T","COMPARATIVE_H_T_K"],lexicon,requested,True)
        if c: out.append(c)

    # Voluntative.
    for stem,form in _suffix_matches(token,VOL_1SG):
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.VOL"),Segment("","person","PERS.1SG")],6.2,["T","H"],
                               "MOOD.VOLUNTATIVE",[],["CSATO2012_T"],lexicon,requested,True)
        if c: out.append(c)
    for stem,form in _suffix_matches(token,VOL_1PL):
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.VOL"),Segment("","person","PERS.1PL")],6.2,["T","H"],
                               "MOOD.VOLUNTATIVE",[],["CSATO2012_T"],lexicon,requested,True)
        if c: out.append(c)
    for stem,form in _suffix_matches(token,VOL_3PL):
        pl=next((x for x in ["ляр","лэр","лер","лар"] if form.endswith(x)),"")
        syn=form[:-len(pl)] if pl else form
        c=_make_verb_candidate(stem,[Segment(syn,"mood","MOOD.VOL"),Segment(pl,"person","PERS.3PL")],6.4,["H","T","K"],
                               "MOOD.VOLUNTATIVE",[],["CSATO2012_T","ILRAN"],lexicon,requested,True)
        if c: out.append(c)
    for stem,form in _suffix_matches(token,VOL_3SG):
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.VOL"),Segment("","person","PERS.3SG")],6.0,["H","T","K"],
                               "MOOD.VOLUNTATIVE",[],["CSATO2012_T","ILRAN","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)

    # Negative voluntative: NEG precedes the mood ending.
    for form,pcat,profiles in [(x,"PERS.1SG",["H","T","K"]) for x in VOL_1SG] + [(x,"PERS.1PL",["H","T","K"]) for x in VOL_1PL]:
        for pre,moodform in _suffix_matches(token,[form]):
            for stem,neg in _suffix_matches(pre,NEG_MA):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(moodform,"mood","MOOD.VOL"),Segment("","person",pcat)],
                                       8.6,profiles,"MOOD.VOLUNTATIVE.NEG",["negative_voluntative"],["KRPS_CORPUS","CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
    for vol in VOL_3SG:
        for pre,moodform in _suffix_matches(token,[vol]):
            for stem,neg in _suffix_matches(pre,NEG_MA):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(moodform,"mood","MOOD.VOL"),Segment("","person","PERS.3SG")],
                                       8.6,["H","T","K"],"MOOD.VOLUNTATIVE.NEG",["negative_voluntative"],["CSATO2012_T","KRPS_CORPUS"],lexicon,requested,True)
                if c: out.append(c)
    for vol in VOL_3PL:
        for pre,moodform in _suffix_matches(token,[vol]):
            for stem,neg in _suffix_matches(pre,NEG_MA):
                pl=next((x for x in ["ляр","лэр","лер","лар"] if moodform.endswith(x)),"")
                syn=moodform[:-len(pl)] if pl else moodform
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(syn,"mood","MOOD.VOL"),Segment(pl,"person","PERS.3PL")],
                                       8.8,["H","T","K"],"MOOD.VOLUNTATIVE.NEG",["negative_voluntative"],["CSATO2012_T","KRPS_CORPUS"],lexicon,requested,True)
                if c: out.append(c)

    # Optative -Gej/-QAy + type-I personal endings; zero/3pl allowed too.
    for before_person,pseg,pprof in _aorist_person_strips(token):
        for pre,opt in _suffix_matches(before_person,OPT_GAY):
            c=_make_verb_candidate(pre,[Segment(opt,"mood","MOOD.OPT"),pseg],6.2,pprof,
                                   "MOOD.OPTATIVE",[],["CSATO2012_T","COMPARATIVE_H_T_K","KRPS_GLOSSES_2026"],lexicon,requested,True)
            if c: out.append(c)
            # Explicit negative optative: ROOT + MA + Gej/QAy + PERSON.
            for stem,neg in _suffix_matches(pre,NEG_MA):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(opt,"mood","MOOD.OPT"),pseg],8.7,pprof,
                                       "MOOD.OPTATIVE.NEG",["negative_optative"],["KRPS_GLOSSES_2026","CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)

    # Trakai 3SG optative can carry abbreviated -t / palatalized -t'.
    for before_t,tform in _suffix_matches(token,["ть","т"]):
        for pre,opt in _suffix_matches(before_t,OPT_GAY):
            c=_make_verb_candidate(pre,[Segment(opt,"mood","MOOD.OPT"),Segment(tform,"person","PERS.3SG")],6.7,["T"],
                                   "MOOD.OPTATIVE",["optional_3sg_t"],["CSATO2012_T","KRPS_GLOSSES_2026"],lexicon,requested,True)
            if c: out.append(c)
            for stem,neg in _suffix_matches(pre,NEG_MA):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(opt,"mood","MOOD.OPT"),Segment(tform,"person","PERS.3SG")],8.9,["T"],
                                       "MOOD.OPTATIVE.NEG",["negative_optative","optional_3sg_t"],["CSATO2012_T","KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)

    # v1.1: source/corpus-attested optative followed by overt 3SG copular
    # marker -dyr/-dir. Keep it Western H/T and lexicon-gated.
    for pcat,pforms,pprof in PRES_3_MARKERS:
        if pcat != "PERS.3SG":
            continue
        full_opt_p3 = [x for x in pforms if x in {"дыр","дир","дур","дюр"}]
        for before_p,pform in _suffix_matches(token,full_opt_p3):
            for stem,opt in _suffix_matches(before_p,OPT_GAY):
                profiles=intersect_profiles(pprof,["H","T"])
                c=_make_verb_candidate(stem,[Segment(opt,"mood","MOOD.OPT"),Segment(pform,"person","PERS.3SG")],
                                       8.2,profiles,"MOOD.OPTATIVE",["optative_with_overt_3sg_copular_marker"],
                                       ["KRPS_GLOSSES_2026","CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)

    # Polite/past optative: -Gey + -DI + type-II person.
    for before_person,pseg,pprof in _person_strips(token,TYPE2_PERSON,include_zero=True):
        for before_di,di in _suffix_matches(before_person,PAST_DI):
            for stem,opt in _suffix_matches(before_di,OPT_GAY):
                profiles=intersect_profiles(pprof,["T"])
                if not profiles:
                    continue
                c=_make_verb_candidate(stem,[Segment(opt,"mood","MOOD.OPT"),Segment(di,"tense","TENSE.PAST.DI"),pseg],6.7,profiles,
                                       "MOOD.OPTATIVE.PAST",["polite_past_optative"],["CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
    return out

def _nominal_tail_variants(token: str):
    """Return possible outer nominal inflection stripped from a non-finite verb.

    No lexical decision is made here. The inner verbal stem must later pass the
    hard KRPS verb lexicon, which strongly limits oversegmentation.
    """
    out=[(token,[],0.0,list(DIALECT_ALL),[])]
    for b,case,sc,notes,prof in detect_regular_case(token):
        out.append((b,[case],sc,prof,notes))
        # v1.1: ordinary PL + CASE after a non-finite core (without possessive).
        for br,pl,pls,pln in detect_plural(b):
            out.append((br,[pl,case],sc+pls+0.6,prof,notes+pln+["plural_before_case"]))
        for bp,pseg,ps,pnotes,pprof in detect_possessive(b):
            ip=intersect_profiles(prof,pprof)
            if ip:
                out.append((bp,[pseg,case],sc+ps+0.4,ip,notes+pnotes))
                for br,pl,pls,pln in detect_plural(bp):
                    out.append((br,[pl,pseg,case],sc+ps+pls+0.8,ip,notes+pnotes+pln))
    for b,segs,sc,notes,prof in detect_p3_special_case(token):
        out.append((b,segs,sc,prof,notes))
        for br,pl,pls,pln in detect_plural(b):
            out.append((br,[pl]+segs,sc+pls+0.8,prof,notes+pln))
    for b,pseg,ps,pnotes,prof in detect_possessive(token):
        out.append((b,[pseg],ps,prof,pnotes))
        for br,pl,pls,pln in detect_plural(b):
            out.append((br,[pl,pseg],ps+pls+0.8,prof,pnotes+pln))
    for b,pl,ps,pnotes in detect_plural(token):
        out.append((b,[pl],ps,list(DIALECT_ALL),pnotes))
    # dedupe
    seen=set(); ded=[]
    for x in out:
        key=(x[0],tuple((s.text,s.morpheme) for s in x[1]))
        if key not in seen:
            seen.add(key); ded.append(x)
    return ded


def _nonfinite_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]

    # v1.1: infinitive/verbal-noun + privative -sIz.
    for core,priv in _suffix_matches(token,DERIV_FORMS["DERIV.PRIV_SIZ"]):
        core_entry = lexicon.get(core)
        if core_entry and core_entry.atomic_stem and not core.endswith("-"):
            continue
        for stem,inf in _suffix_matches(core,INF_MAK):
            c=_make_verb_candidate(stem,[Segment(inf,"infinitive","NONFIN.INF"),Segment(priv,"derivational","DERIV.PRIV_SIZ")],
                                   8.1,["H","T","K"],"INF+PRIV",["nonfinite_plus_privative"],
                                   ["KRPS_GLOSSES_2026"],lexicon,requested,True)
            if c: out.append(c)

    # v1.1: selected source/corpus-supported outer morphology after -GAn.
    for tail in NONFINITE_ADV_LEY:
        for core,_ in _suffix_matches(token,[tail]):
            for stem,gan in _suffix_matches(core,PTCP_GAN):
                c=_make_verb_candidate(stem,[Segment(gan,"participle","PTCP.PAST.GAN"),Segment(tail,"derivational","DERIV.ADV.LEY")],
                                       8.0,["H","T","K"],"PTCP.PAST.ADV",["participle_adverbial_ley"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)
    for tail in NONFINITE_LIMITATIVE:
        for core,_ in _suffix_matches(token,[tail]):
            for stem,gan in _suffix_matches(core,PTCP_GAN):
                c=_make_verb_candidate(stem,[Segment(gan,"participle","PTCP.PAST.GAN"),Segment(tail,"derivational","DERIV.NONFIN.LIMIT")],
                                       7.8,["H","T","K"],"PTCP.PAST.LIMIT",["participle_outer_limitative"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)
    # Past participle + overt predicative/copular 3SG.
    for pcat,pforms,pprof in PRES_3_MARKERS:
        if pcat != "PERS.3SG":
            continue
        full_pred = [x for x in pforms if x in {"дыр","дир","дур","дюр"}]
        for core,pform in _suffix_matches(token,full_pred):
            for stem,gan in _suffix_matches(core,PTCP_GAN):
                c=_make_verb_candidate(stem,[Segment(gan,"participle","PTCP.PAST.GAN"),Segment(pform,"copula","COP.PRED.3SG")],
                                       8.3,pprof,"PTCP.PAST.PRED",["participle_predicative_copula"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)

    # Western negative infinitive -mAsQA. Keep it as a dedicated portmanteau
    # morpheme: synchronically it is a conventional negative infinitive, not a
    # productive guess from ordinary nominal case stripping.
    for stem,form in _suffix_matches(token,NEG_INF_WEST):
        c=_make_verb_candidate(stem,[Segment(form,"infinitive","NONFIN.INF.NEG.WESTERN")],8.8,["H","T"],
                               "INF.NEG.WESTERN",["western_negative_infinitive_mAsQA"],
                               ["NEMETH2015_NW","NEMETH2016_EAST_WEST","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)

    # Additional converbs securely represented in the supplied Cyrillic corpus.
    for stem,form in _suffix_matches(token,CVB_NEG_MAYINCHA):
        c=_make_verb_candidate(stem,[Segment(form,"converb","CVB.NEG.MAYINCHA")],9.0,["H","T","K"],
                               "CVB.NEG.MAYINCHA",["negative_until_without_converb"],["KRPS_GLOSSES_2026","TURKIC_COMPARATIVE"],lexicon,requested,True)
        if c: out.append(c)
    # -mAyIn is surface-ambiguous with a negative voluntative; retain it, but
    # below the explicit voluntative score so single-token legacy decisions do
    # not silently change. Sentence context can promote it later.
    for stem,form in _suffix_matches(token,CVB_NEG_MAYIN):
        c=_make_verb_candidate(stem,[Segment(form,"converb","CVB.NEG.MAYIN")],8.2,["H","T","K"],
                               "CVB.NEG.MAYIN",["surface_ambiguous_with_negative_voluntative"],["KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)
    for stem,form in _suffix_matches(token,CVB_GINCHA):
        c=_make_verb_candidate(stem,[Segment(form,"converb","CVB.GINCHA")],7.5,["H","T","K"],
                               "CVB.GINCHA",["terminative_temporal_converb"],["KRPS_GLOSSES_2026","TURKIC_COMPARATIVE"],lexicon,requested,True)
        if c: out.append(c)

    for core,tail,tail_score,tail_prof,tail_notes in _nominal_tail_variants(token):
        rules = [
            ("INF", INF_MAK, "infinitive", "NONFIN.INF", 7.0, ["H","T","K"], ["ILRAN","KRPS","KRPS_GLOSSES_2026"]),
            ("PTCP.PAST", PTCP_GAN, "participle", "PTCP.PAST.GAN", 7.1, ["H","T","K"], ["CSATO2012_T","ILRAN"]),
            ("PTCP.PRESENT", PTCP_PRESENT, "participle", "PTCP.PRESENT.AJDOGON", 7.4, ["H","T","K"], ["CSATO2012_T","ILRAN"]),
            ("PTCP.UVCI", PTCP_UVCI, "participle", "PTCP.UVCI", 6.2, ["H","T","K"], ["KRPS","CSATO2012_T","KARAIM_LT_UVCI"]),
            ("PTCP.MISH", PTCP_MISH, "participle", "PTCP.MISH", 5.2, ["K"], ["ISIK2024","PRIK1976"]),
            ("CVB.IPTA", CVB_IPTA, "converb", "CVB.IPTA", 6.5, ["H","T","K"], ["CSATO2012_T"]),
            ("CVB.IP", CVB_IP, "converb", "CVB.IP", 6.0, ["H","T","K"], ["CSATO2012_T"]),
            ("CVB.NEG.MAY", CVB_NEG_MAY, "converb", "CVB.NEG.MAY", 6.5, ["H","T","K"], ["COMPARATIVE_H_T_K"]),
            ("CVB.NEG.MASTAN", CVB_NEG_MASTAN, "converb", "CVB.NEG.MASTAN", 6.5, ["H","T","K"], ["KRPS_CORPUS"]),
            ("CVB.GANDA", CVB_GANDA, "converb", "CVB.GANDA", 6.8, ["H","T","K"], ["CSATO2012_T"]),
            ("VN.UV", VN_UV_VERB, "verbal_noun", "VN.UV", 5.8, ["H","T","K"], ["KRPS"]),
            ("VN.ISH", VN_ISH_VERB, "verbal_noun", "VN.ISH", 5.8, ["H","T","K"], ["KRPS"]),
            ("VN.MA", VN_MA, "verbal_noun", "VN.MA", 5.2, ["H","T","K"], ["CSATO2012_T","ILRAN"]),
        ]
        for par,forms,role,morph,base_score,prof,support in rules:
            if tail and par.startswith("CVB"):
                continue
            active_forms = list(forms) + (INF_MAK_OBLIQUE if (par == "INF" and tail) else [])
            for stem,form in _suffix_matches(core,active_forms):
                segs=[Segment(form,role,morph)] + tail
                tail_penalty = 3.5 if (par == "VN.MA" and tail) else 0.0
                if par == "PTCP.UVCI" and form in {"вчу","вчи","вци","вцу","вджы"}:
                    # Reduced allomorphs are valid after vowel-final stems but
                    # should not outrank an equally valid fuller -IvČI parse.
                    tail_penalty += 2.0
                # Oblique infinitive stems in -mag/-magъ are highly ambiguous
                # with NEG -ma + imperative -GIN; retain them as candidates but
                # keep explicit mood morphology ahead in ranking.
                if par == "INF" and form in INF_MAK_OBLIQUE:
                    tail_penalty += 3.0
                c=_make_verb_candidate(stem,segs,base_score+tail_score-tail_penalty,intersect_profiles(prof,tail_prof),par,
                                       tail_notes+(["nonfinite_with_nominal_tail"] if tail else []),support,lexicon,requested,True)
                if c: out.append(c)
                # negative participle/infinitive: stem + MA + marker
                if par in {"INF","PTCP.PAST","PTCP.PRESENT","PTCP.UVCI"}:
                    for nstem,neg in _suffix_matches(stem,NEG_MA):
                        c=_make_verb_candidate(nstem,[Segment(neg,"negation","NEG"),Segment(form,role,morph)]+tail,
                                               base_score+tail_score+0.7,intersect_profiles(prof,tail_prof),par+".NEG",
                                               tail_notes+["negative_nonfinite"],support,lexicon,requested,True)
                        if c: out.append(c)
        # present participle + -č converb
        for pre,ch in _suffix_matches(core,["ч","ц"]):
            for stem,form in _suffix_matches(pre,PTCP_PRESENT):
                c=_make_verb_candidate(stem,[Segment(form,"participle","PTCP.PRESENT.AJDOGON"),Segment(ch,"converb","CVB.PTCP.C")]+tail,
                                       7.2+tail_score,intersect_profiles(["H","T"],tail_prof),"CVB.PRESENT_PTCP_C",tail_notes,
                                       ["CSATO2012_T"],lexicon,requested,True)
                if c: out.append(c)
    # bare -A/j converb, deliberately lower-scored due to homography
    for stem,form in _suffix_matches(token,PRESENT_A+PRESENT_J):
        c=_make_verb_candidate(stem,[Segment(form,"converb","CVB.A")],4.0,["H","T","K"],"CVB.A",
                               ["high_surface_ambiguity"],["CSATO2012_T"],lexicon,requested,True)
        if c: out.append(c)
    return out

def _historical_western_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    """Opt-in historical Western Karaim morphology.

    Nothing in this layer is generated during default contemporary analysis.
    The layer is deliberately exact/conservative and still requires a KRPS
    verbal root compatible with H/T.
    """
    out=[]
    west=["H","T"]

    # North-Western inherited 2SG -sen after overt aorist.
    for before_person,pform in _suffix_matches(token,["сэн","сен"]):
        for stem,aor in _suffix_matches(before_person,AOR_FULL):
            c=_make_verb_candidate(stem,[Segment(aor,"tense","TENSE.AOR.FUT"),Segment(pform,"person","PERS.2SG")],
                                   7.8,["T"],"IND.AOR_FUT.HISTORICAL_WESTERN",
                                   ["historical_nw_2sg_sen","historical_profile:historical_western"],
                                   ["NEMETH2015_NW_SEN"],lexicon,requested,True)
            if c: out.append(c)

    # Continuative present A/j + dIr/dy/d + ordinary person.
    for before_person,pseg,pprof in _person_strips(token,TYPE1_PERSON,include_zero=True):
        for dbase,dform in _suffix_matches(before_person,["дыр","дир","дур","дюр","ды","ди","д"]):
            for stem,aform in _suffix_matches(dbase,PRESENT_A+PRESENT_J):
                c=_make_verb_candidate(stem,[Segment(aform,"converb","CVB.A"),Segment(dform,"aspect","ASP.CONT"),pseg],
                                       8.0,intersect_profiles(pprof,west),"IND.PRESENT.CONT.HISTORICAL_WESTERN",
                                       ["historical_profile:historical_western","western_continuative_A_dyr"],
                                       ["NEMETH2019_CONT"],lexicon,requested,True)
                if c: out.append(c)

    # Abbreviated 3PL -dlar/-dler, directly or after an A/j present linker.
    for base,dform in _suffix_matches(token,HIST_W_3PL_D):
        c=_make_verb_candidate(base,[Segment(dform,"person","PERS.3PL")],8.2,west,
                               "IND.PRESENT.CONT.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","abbreviated_western_3pl_dlar"],
                               ["NEMETH2019_CONT","KRPS_GLOSSES_2026"],lexicon,requested,False)
        if c: out.append(c)
        for stem,aform in _suffix_matches(base,PRESENT_A+PRESENT_J):
            c=_make_verb_candidate(stem,[Segment(aform,"converb","CVB.A"),Segment(dform,"person","PERS.3PL")],8.6,west,
                                   "IND.PRESENT.CONT.HISTORICAL_WESTERN",
                                   ["historical_profile:historical_western","abbreviated_western_3pl_dlar"],
                                   ["NEMETH2019_CONT","KRPS_GLOSSES_2026"],lexicon,requested,True)
            if c: out.append(c)

    # Historical Western perfect conditional: migrate the conservative v1.0
    # -SA + (e)DI + Type-II analysis into the opt-in profile.
    for before_person,pseg,pprof in _person_strips(token,TYPE2_PERSON,include_zero=True):
        for pre,form in _suffix_matches(before_person,["сайды","сэйди","сейди","сыйды","сийди","суду","сюдю"]):
            profiles=intersect_profiles(pprof,west)
            c=_make_verb_candidate(pre,[Segment(form,"mood","MOOD.COND.PERF.HIST_W"),pseg],8.5,profiles,
                                   "MOOD.CONDITIONAL.PERF.HISTORICAL_WESTERN",
                                   ["historical_profile:historical_western","fused_SA_EDI"],
                                   ["HIST_W","GULTEKIN2002"],lexicon,requested,True)
            if c: out.append(c)

    # Contracted -SA + edi spellings retained as exact historical portmanteaux.
    for stem,form in _suffix_matches(token,HIST_W_COND_PORTMANTEAU):
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.COND.PERF.HIST_W")],8.4,west,
                               "MOOD.CONDITIONAL.PERF.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","contracted_SA_EDI"],
                               ["GULTEKIN2002","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)

    # Fused aorist + shortened edi; including 3PL auxiliaries.
    hist_aux = list(HIST_W_AUX_3SG) + ["эдляр","едляр","эдлер","едлер","эдлэр","едлэр","эдлар","едлар",
                                             "эдьляр","едьляр","эдьлер","едьлер"]
    for pre,auxsurf in _suffix_matches(token,hist_aux):
        for stem,aor in _suffix_matches(pre,AOR_FULL):
            pcat="PERS.3PL" if auxsurf.endswith(("ляр","лер","лэр","лар")) else "PERS.3SG"
            c=_make_verb_candidate(stem,[Segment(aor,"tense","TENSE.AOR"),Segment(auxsurf,"auxiliary","AUX.COP.PAST.HIST_W"),Segment("","person",pcat)],
                                   8.4,west,"IND.PAST.INTRATERMINAL.HISTORICAL_WESTERN",
                                   ["historical_profile:historical_western","fused_short_EDI"],
                                   ["NEMETH2019_CONT","KRPS_GLOSSES_2026"],lexicon,requested,True)
            if c: out.append(c)

    # Western historical -(I)p edi pluperfect.
    endings=[("PERS.3SG",x,west) for x in COP_PAST_3SG]
    endings += [(cat,x,intersect_profiles(prof,west)) for cat,forms,prof in COP_PAST_PERSONAL for x in forms]
    for pcat,auxsurf,prof in endings:
        if not prof: continue
        for pre,_ in _suffix_matches(token,[auxsurf]):
            for stem,ip in _suffix_matches(pre,CVB_IP):
                c=_make_verb_candidate(stem,[Segment(ip,"converb","CVB.IP"),Segment(auxsurf,"auxiliary","AUX.COP.PAST"),Segment("","person",pcat)],
                                       8.5,prof,"IND.PLUPERFECT.P_EDI.HISTORICAL_WESTERN",
                                       ["historical_profile:historical_western","fused_P_EDI"],
                                       ["NEMETH_P_EDI"],lexicon,requested,True)
                if c: out.append(c)

    # Historical person spellings. First try ordinary present A/j + marker,
    # then a direct marker after a vowel-final lexical stem (attested contraction).
    hist_person=[("PERS.1SG",HIST_W_PERSON_1SG,west),("PERS.1PL",HIST_W_PERSON_1PL,["T"])]
    for pcat,forms,prof in hist_person:
        for before_p,pform in _suffix_matches(token,forms):
            for stem,aform in _suffix_matches(before_p,PRESENT_A+PRESENT_J):
                c=_make_verb_candidate(stem,[Segment(aform,"tense","TENSE.PRES"),Segment(pform,"person",pcat)],8.0,prof,
                                       "IND.PRESENT.HISTORICAL_WESTERN",
                                       ["historical_profile:historical_western","historical_person_orthography"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)
            c=_make_verb_candidate(before_p,[Segment("","tense","TENSE.PRES.HIST_CONTRACTED"),Segment(pform,"person",pcat)],7.3,prof,
                                   "IND.PRESENT.HISTORICAL_WESTERN",
                                   ["historical_profile:historical_western","historical_person_direct_after_vowel_stem"],
                                   ["KRPS_GLOSSES_2026"],lexicon,requested,False)
            if c: out.append(c)

    # Palatalized Western negative aorist / negative infinitive family.
    for stem,form in _suffix_matches(token,HIST_W_NEG_INF):
        c=_make_verb_candidate(stem,[Segment(form,"infinitive","NONFIN.INF.NEG.HIST_W")],8.7,west,
                               "INF.NEG.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","palatalized_mAsQA"],
                               ["NEMETH2015_NW","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)
    for stem,neg in _suffix_matches(token,HIST_W_NEG_MAS):
        c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG+AOR.HIST_W"),Segment("","person","PERS.3SG")],8.0,west,
                               "IND.AOR_FUT.NEG.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","palatalized_mAs"],
                               ["KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)
    for pform in ["тирляр","тьляр","тирлер","тьлер"]:
        for pre,_ in _suffix_matches(token,[pform]):
            for stem,neg in _suffix_matches(pre,HIST_W_NEG_MAS):
                c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG+AOR.HIST_W"),Segment(pform,"person","PERS.3PL")],8.6,west,
                                       "IND.AOR_FUT.NEG.HISTORICAL_WESTERN",
                                       ["historical_profile:historical_western","palatalized_mAs_3pl"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)

    # Western historical -mIš retained as nonproductive/petrified.
    for stem,mish in _suffix_matches(token,["мись","миш"]):
        c=_make_verb_candidate(stem,[Segment(mish,"participle","PTCP.MISH.HIST_W")],7.7,west,
                               "PTCP.MISH.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","western_mish_petrified"],
                               ["NEMETH2015_NW","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)

    # Trakai-like -sIn recorded in an H-labelled historical example.
    for pform in ["сын","син"]:
        for pre,_ in _suffix_matches(token,[pform]):
            for stem,aform in _suffix_matches(pre,PRESENT_A+PRESENT_J):
                c=_make_verb_candidate(stem,[Segment(aform,"tense","TENSE.PRES"),Segment(pform,"person","PERS.2SG")],7.6,["H"],
                                       "IND.PRESENT.HISTORICAL_WESTERN",
                                       ["historical_profile:historical_western","dialect_mixed_H_sIn"],
                                       ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                if c: out.append(c)

    # A few historical spellings preserve a final -ыи marker whose TAM value
    # is not safely recoverable from the surface alone. Expose morphology but
    # explicitly leave TAM underspecified.
    for stem,form in _suffix_matches(token,["ыи"]):
        c=_make_verb_candidate(stem,[Segment(form,"historical_marker","HIST.W.FINAL_Y")],6.8,["H","T"],
                               "VERB.HISTORICAL_WESTERN.UNDERSPECIFIED",
                               ["historical_profile:historical_western","tam_underspecified"],
                               ["KRPS_GLOSSES_2026"],lexicon,requested,False)
        if c: out.append(c)

    # Contracted optative -QAy surface (e.g. болъай).
    for stem,form in _suffix_matches(token,["ъай"]):
        c=_make_verb_candidate(stem,[Segment(form,"mood","MOOD.OPT.HIST_W")],7.5,["H"],
                               "MOOD.OPTATIVE.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","contracted_QAy"],
                               ["KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)

    # Historical Western -doGAč converb, including explicit negative MA.
    for pre,form in _suffix_matches(token,CVB_DOGAC):
        c=_make_verb_candidate(pre,[Segment(form,"converb","CVB.DOGAC")],8.2,west,
                               "CVB.DOGAC.HISTORICAL_WESTERN",
                               ["historical_profile:historical_western","western_doGAc_converb"],
                               ["NEMETH2015_NW","KRPS_GLOSSES_2026"],lexicon,requested,True)
        if c: out.append(c)
        for stem,neg in _suffix_matches(pre,NEG_MA):
            c=_make_verb_candidate(stem,[Segment(neg,"negation","NEG"),Segment(form,"converb","CVB.DOGAC")],8.9,west,
                                   "CVB.DOGAC.NEG.HISTORICAL_WESTERN",
                                   ["historical_profile:historical_western","negative_western_doGAc_converb"],
                                   ["NEMETH2015_NW","KRPS_GLOSSES_2026"],lexicon,requested,True)
            if c: out.append(c)

    # Hyphenated historical auxiliary/person constructions: parse the left
    # verbal core independently, then attach the explicitly written historical
    # segment. This is exact punctuation-aware parsing, not normalization.
    if token.count("-")==1:
        left,right=token.split("-",1)
        if left and right:
            if right in set(COP_PAST_3SG) | {x for _,forms,_ in COP_PAST_PERSONAL for x in forms}:
                core=generate_verbal_candidates(left,lexicon,requested,historical_profile="historical_western",_allow_question=False)
                # Historical -gяй spelling before explicitly hyphenated edi.
                for stem,opt in _suffix_matches(left,["гяй"]):
                    c=_make_verb_candidate(stem,[Segment(opt,"mood","MOOD.OPT.HIST_W")],8.0,west,
                                           "MOOD.OPTATIVE.HISTORICAL_WESTERN",
                                           ["historical_profile:historical_western","historical_gяй"],
                                           ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                    if c: core.append(c)
                for base_c in core:
                    hc=copy.deepcopy(base_c)
                    hc.segments.append(Segment("-"+right,"auxiliary","AUX.COP.PAST.HIST_W"))
                    hc.paradigm=(base_c.paradigm or "VERB")+".AUX_EDI.HISTORICAL_WESTERN"
                    hc.notes.append("historical_profile:historical_western")
                    hc.score += 0.6
                    out.append(hc)
            elif right in ["сэн","сен"]:
                for stem,aor in _suffix_matches(left,AOR_FULL):
                    c=_make_verb_candidate(stem,[Segment(aor,"tense","TENSE.AOR.FUT"),Segment("-"+right,"person","PERS.2SG")],8.0,["H","T"],
                                           "IND.AOR_FUT.HISTORICAL_WESTERN",
                                           ["historical_profile:historical_western","hyphenated_person"],
                                           ["KRPS_GLOSSES_2026"],lexicon,requested,True)
                    if c: out.append(c)
    return _dedupe_candidates(out)


def _potential_alternatives(cands: List[Candidate], lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    import copy
    extra=[]
    for c in cands:
        if not c.segments or c.analysis_domain != "verbal":
            continue
        first=c.segments[0]
        if first.role not in {"verbal_stem","root"}:
            continue
        # Only expand the directly inflected stem, not an already KRPS-expanded root.
        surface_stem=c.stem_surface
        for base,pot in _suffix_matches(surface_stem,POTENTIAL):
            resolved=lexicon.resolve_verb_candidate(base,_verb_restorations(base,True),requested)
            if not resolved:
                continue
            e,mt=resolved
            profiles=intersect_profiles(c.dialect_profiles,["H","T"],e.dialects or DIALECT_ALL)
            if not profiles:
                continue
            alt=copy.deepcopy(c)
            # find original stem segment only when it is still present literally
            if alt.segments and alt.segments[0].text == surface_stem:
                alt.segments=[Segment(base,"verbal_stem"),Segment(pot,"modality","MOD.POT")]+alt.segments[1:]
            else:
                continue
            alt.stem_surface=base
            alt.resolved_lemma=e.headword
            alt.lemma_dialects=list(e.dialects)
            alt.lexicon_match_type="exact_verb_stem+productive_potential"
            alt.lexicon_source_rows=list(e.source_rows)
            alt.derivational_chain=lexicon.derivational_chain(e.headword)
            alt.dialect_profiles=profiles
            alt.notes.append("productive_potential_preferred_over_lexicalized_extended_stem")
            alt.source_support=list(dict.fromkeys(alt.source_support+["CSATO2012_T"]))
            alt.score += 2.5
            extra.append(alt)
    return cands+extra

def generate_verbal_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str], historical_profile: Optional[str] = None, _allow_question: bool = True) -> List[Candidate]:
    if token.endswith("-"):
        return []
    cands=[]

    # Enclitic interrogative particle is outside TAM/person morphology. Parse the
    # finite core first, then append Q.PARTICLE only when the core already has a
    # valid finite/mood analysis under both hard constraints.
    if _allow_question:
        for qform,qprofiles in QUESTION_PARTICLES:
            if token.endswith(qform) and len(token) > len(qform):
                core=token[:-len(qform)]
                core_cands=generate_verbal_candidates(core,lexicon,requested,historical_profile=historical_profile,_allow_question=False)
                for base_c in core_cands:
                    if not (base_c.paradigm or "").startswith(("IND.","MOOD.")):
                        continue
                    # A bare 2SG imperative is homographic with the dictionary
                    # stem; using it as the core of an attached question particle
                    # creates many false positives (e.g. a lexical -mA ending).
                    # Leave that construction to sentence-level disambiguation.
                    if base_c.paradigm == "MOOD.IMPERATIVE.2SG" and "bare_stem_imperative" in base_c.notes:
                        continue
                    profiles=intersect_profiles(base_c.dialect_profiles,qprofiles)
                    if not profiles:
                        continue
                    qc=copy.deepcopy(base_c)
                    qc.segments.append(Segment(qform,"clitic","Q.PARTICLE"))
                    qc.dialect_profiles=profiles
                    qc.paradigm=(base_c.paradigm or "") + ".INTERROGATIVE"
                    qc.notes.append("enclitic_interrogative_particle")
                    qc.source_support=list(dict.fromkeys(qc.source_support+["CSATO2012_T","NEMETH2012_WESTERN","CRIMEAN_MURAT2023"]))
                    qc.score += 0.7
                    cands.append(qc)
    cands += _finite_past_candidates(token,lexicon,requested)
    cands += _finite_present_candidates(token,lexicon,requested)
    cands += _finite_aorist_candidates(token,lexicon,requested)
    cands += _compound_past_candidates(token,lexicon,requested)
    cands += _conditional_candidates(token,lexicon,requested)
    cands += _imperative_vol_opt_candidates(token,lexicon,requested)
    cands += _nonfinite_candidates(token,lexicon,requested)
    if historical_profile == "historical_western":
        cands += _historical_western_candidates(token,lexicon,requested)
    cands = _potential_alternatives(cands, lexicon, requested)
    # Deduplicate with preference for richer analyses and longer lexically resolved stems.
    cands=_dedupe_candidates(cands)
    for c in cands:
        if c.analysis_domain == "verbal":
            if c.category_sequence():
                c.confidence = "high" if c.score >= 13 else "medium_candidate"
    return cands


# ---------------------------------------------------------------------------
# v0.6 closed classes, nominal derivation and predicative morphology
# ---------------------------------------------------------------------------

# Closed-class forms are encoded explicitly because personal/demonstrative
# pronouns are not reliably analyzable by the ordinary noun declension rules.
# The table is intentionally conservative; each form points to a KRPS lemma.
PRONOUN_FORMS = {
    "сэнинъ": {"lemma":"сень", "dialects":["T"], "segments":[("сэ","pronoun_stem","PRON.2SG"),("нинъ","case","CASE.GEN")]},
    "онъа":   {"lemma":"ол",   "dialects":["H","T"], "segments":[("онъ","pronoun_oblique_stem","PRON.3SG.OBL"),("а","case","CASE.DAT")]},
    "кимге":  {"lemma":"ким",  "dialects":["H","T"], "segments":[("ким","pronoun_stem","PRON.INTERROG"),("ге","case","CASE.DAT")]},
    "майя":   {"lemma":"мень", "dialects":["T"], "segments":[("майя","pronoun_oblique_stem","PRON.1SG.DAT")]},
}

PREDICATIVE_FORMS = [
    ("дыр", ["H","T","K"]), ("дир", ["H","T","K"]), ("дур", ["H","T","K"]), ("дюр", ["H","T","K"]),
    ("ды", ["H","T"]), ("ди", ["H","T"]), ("д", ["H","T"]), ("т", ["T"]),
]

NOMINAL_DERIV_RULES = [
    ("DERIV.ADJ.LI", ["лы","ли","лу","лю"], ["H","T","K"], "adjectival", ["MUSAEV_WESTERN","FOLTYN_DERIVATION","ISIK_CRIMEAN"]),
    # Crimean +ǮA/equative-type derivative; Western cognate allomorphs are
    # included but still hard-gated by the lexical base and dialect.
    ("DERIV.EQUATIVE.CA", ["джа","джэ","ча","чэ","ца","цэ"], ["H","T","K"], "derivational", ["ISIK_CRIMEAN","MUSAEV_WESTERN"]),
    # Comparative: Trakai -rAx/-ryAk, South-Western -rAk/-rek.
    ("DERIV.COMPAR.RAK", ["рах","рях"], ["T"], "comparative", ["MUSAEV_WESTERN","CSATO2012_T"]),
    ("DERIV.COMPAR.RAK", ["рак","рек"], ["H"], "comparative", ["MUSAEV_WESTERN"]),
]


def generate_pronoun_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    meta=PRONOUN_FORMS.get(token)
    if not meta:
        return []
    entry=lexicon.get(meta["lemma"])
    if not entry:
        return []
    profiles=intersect_profiles(meta["dialects"], entry.dialects or DIALECT_ALL)
    if requested:
        profiles=intersect_profiles(profiles,requested)
    if not profiles:
        return []
    segs=[Segment(t,r,m,"closed_class_paradigm") for t,r,m in meta["segments"]]
    c=Candidate(meta["segments"][0][0],segs,35.0,"high",["closed_class_pronoun_paradigm"],[],profiles,
                resolved_lemma=entry.headword,lemma_dialects=list(entry.dialects),lexicon_match_type="closed_class_lemma",
                lexicon_source_rows=list(entry.source_rows),analysis_domain="pronominal",paradigm="PRON.DECLENSION",
                source_support=["MUSAEV_WESTERN"])
    return [c]


def generate_predicative_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    for suff,profiles in sorted(PREDICATIVE_FORMS,key=lambda x:len(x[0]),reverse=True):
        if not token.endswith(suff) or len(token)<=len(suff)+1:
            continue
        base=token[:-len(suff)]
        # The predicate base must be an attested non-verbal KRPS lexeme (or an
        # explicitly supplied external lexeme), preventing verbal -DIr analyses.
        e=lexicon.get(base)
        if not e or base.endswith("-") or not e.atomic_stem:
            continue
        p=intersect_profiles(profiles,e.dialects or DIALECT_ALL)
        if requested: p=intersect_profiles(p,requested)
        if not p: continue
        c=Candidate(base,[Segment(base,"stem"),Segment(suff,"copula","COP.PRED.3SG")],12.5,"high",
                    ["nominal_predicate_copula"],[],p,resolved_lemma=e.headword,lemma_dialects=list(e.dialects),
                    lexicon_match_type="exact_predicate_base",lexicon_source_rows=list(e.source_rows),
                    analysis_domain="predicate",paradigm="COP.PRED.3SG",source_support=["NEMETH_COPULA","CSATO_COPULA"])
        out.append(c)
    return out


def generate_nominal_derivation_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out=[]
    for morph,forms,profiles,role,support in NOMINAL_DERIV_RULES:
        for base,suff in _suffix_matches(token,forms):
            e=lexicon.get(base)
            if not e or not e.atomic_stem or base.endswith("-"):
                # lexical-gated soft-sign restoration may still identify the base
                resolved=lexicon.resolve_candidate(base,restore_lemma_candidates(base,False),requested)
                if not resolved: continue
                e,mt=resolved
            else:
                mt="exact_stem"
            p=intersect_profiles(profiles,e.dialects or DIALECT_ALL)
            if requested: p=intersect_profiles(p,requested)
            if not p: continue
            surface_base=base
            c=Candidate(surface_base,[Segment(surface_base,"stem"),Segment(suff,role,morph)],24.0,"high",
                        ["productive_nominal_derivation"],restore_lemma_candidates(base,False),p,
                        resolved_lemma=e.headword,lemma_dialects=list(e.dialects),lexicon_match_type=mt+"+productive_derivation",
                        lexicon_source_rows=list(e.source_rows),analysis_domain="derivational",paradigm=morph,
                        source_support=support)
            out.append(c)
    return _dedupe_candidates(out)


# v0.8: productive nominal derivation may precede ordinary nominal inflection.
# The first implemented family is -LIK + PL + POSS. It is deliberately
# conservative: the lexical ROOT must be an exact non-verbal KRPS headword,
# plural must be overt, and a possessive suffix must follow. This avoids turning
# every apparent -LIK ending into a derivational analysis.
LIK_CHAIN_FORMS = ["лыкъ", "лык", "лик", "лук", "люк", "лых", "лих", "лух", "люх"]

def generate_lik_inflection_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    out: List[Candidate] = []
    # ROOT + LIK + PL + POSS
    for before_poss, pseg, pscore, pnotes, poss_profiles in detect_possessive(token):
        for derived_stem, plseg, plscore, plnotes in detect_plural(before_poss):
            for lik in sorted(LIK_CHAIN_FORMS, key=len, reverse=True):
                if not derived_stem.endswith(lik) or len(derived_stem) <= len(lik) + 1:
                    continue
                root = derived_stem[:-len(lik)]
                entry = lexicon.get(root)
                # Exact root only. No restoration/fuzzy path at the derivational boundary.
                if not entry or not entry.atomic_stem or root.endswith("-"):
                    continue
                profiles = intersect_profiles(poss_profiles, entry.dialects or DIALECT_ALL)
                if requested:
                    profiles = intersect_profiles(profiles, requested)
                if not profiles:
                    continue
                segs = [
                    Segment(root, "stem"),
                    Segment(lik, "derivational", "DERIV.NOMINAL_LIK", "productive_derivation"),
                    plseg, pseg
                ]
                score = 10.0 + plscore + pscore
                notes = ["v08_lik_before_inflection", "exact_krps_root_required"] + plnotes + pnotes
                out.append(Candidate(
                    root, segs, score, "high", notes, [], profiles,
                    resolved_lemma=entry.headword, lemma_dialects=list(entry.dialects),
                    lexicon_match_type="exact_root+productive_LIK+inflection",
                    lexicon_source_rows=list(entry.source_rows), analysis_domain="derivational",
                    paradigm="DERIV.NOMINAL_LIK+PL+POSS",
                    source_support=["KRPS_PRODUCTIVE_LIK", "MUSAEV_KARAIM_LIK", "CSATO2012_T_POSS"]
                ))
    return _dedupe_candidates(out)

def generate_v06_special_candidates(token: str, lexicon: KRPSLexicon, requested: Iterable[str]) -> List[Candidate]:
    return _dedupe_candidates(generate_pronoun_candidates(token,lexicon,requested)+
                              generate_predicative_candidates(token,lexicon,requested)+
                              generate_nominal_derivation_candidates(token,lexicon,requested)+
                              generate_lik_inflection_candidates(token,lexicon,requested))

# ---------------------------------------------------------------------------
# Ranking / constraints
# ---------------------------------------------------------------------------

def _assign_confidence(c: Candidate) -> None:
    cats = c.category_sequence()
    special = "p3_possessive_case_special" in c.notes
    if len(cats) >= 2 or special:
        c.confidence = "high" if c.score >= 5.0 else "medium"
    else:
        c.confidence = "medium_candidate" if c.score >= 2.1 else "low_candidate"


def _dedupe_candidates(cands: List[Candidate]) -> List[Candidate]:
    seen, out = set(), []
    for c in sorted(cands, key=lambda x: (-x.score, -len(x.category_sequence()), x.segmented())):
        key = (tuple((s.text, s.morpheme) for s in c.segments), tuple(c.dialect_profiles))
        if key in seen:
            continue
        seen.add(key)
        _assign_confidence(c)
        out.append(c)
    return out


def apply_dialect_constraint(cands: List[Candidate], dialects: Optional[Iterable[str]]) -> Tuple[List[Candidate], int, str]:
    allowed = normalize_dialects(dialects=dialects)
    if not allowed:
        return cands, 0, "not_applied_unknown"
    allowed_set = set(allowed)
    kept: List[Candidate] = []
    rejected = 0
    for c in cands:
        matched = [d for d in DIALECT_ALL if d in allowed_set and d in c.dialect_profiles]
        if matched:
            c.notes.append("dialect_constraint_match:" + "|".join(matched))
            kept.append(c)
        else:
            rejected += 1
    return kept, rejected, "applied:" + "|".join(allowed)

def apply_lexicon_constraint(cands: List[Candidate], lexicon: KRPSLexicon,
                             requested_dialects: Iterable[str]) -> Tuple[List[Candidate], int]:
    """Hard-filter candidate analyses through the KRPS headword lexicon."""
    kept: List[Candidate] = []
    rejected = 0
    for c in cands:
        resolved = lexicon.resolve_candidate(c.stem_surface, c.lemma_restore_candidates, requested_dialects)
        if resolved is None:
            rejected += 1
            continue
        entry, match_type = resolved
        if entry.dialects:
            compatible = intersect_profiles(c.dialect_profiles, entry.dialects)
            if not compatible:
                rejected += 1
                continue
            c.dialect_profiles = compatible
        c.resolved_lemma = entry.headword
        c.lemma_dialects = list(entry.dialects)
        c.lexicon_match_type = match_type
        c.lexicon_source_rows = list(entry.source_rows)
        c.notes.append("lexicon_hard_match:" + match_type)
        # Lexicon-aware anti-oversegmentation prior: when several analyses have
        # real KRPS lemmas, prefer the analysis retaining the longer lexical
        # stem. This is ranking only; it never rescues a non-lexical stem.
        c.score += (1.0 if match_type == "exact_stem" else 0.35) + 1.5 * len(entry.headword)
        c.notes.append("lexicon_longest_stem_prior")
        kept.append(c)
    return _dedupe_candidates(kept), rejected

# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

def analyze_token(token: str, lexicon: KRPSLexicon, line: Optional[int] = None,
                  dialect: Optional[str] = None, dialects: Optional[Iterable[str]] = None,
                  historical_profile: Optional[str] = None) -> TokenResult:
    original = token.rstrip("\n")
    norm = prepare_surface(original)
    requested = normalize_dialects(dialect=dialect, dialects=dialects)
    dstatus = "applied:" + "|".join(requested) if requested else "not_applied_unknown"

    if not norm:
        return TokenResult(line, original, norm, "EMPTY", "no_analysis", None, [], [],
                           requested_dialects=requested, dialect_constraint_status=dstatus,
                           lexicon_constraint_status="applied_hard")

    # LHF v1.0: source-verified lexicalized/frozen forms guard the productive parser.
    # Matching is exact on the canonical project surface; Latin transliterations are
    # metadata only and can never activate this branch.
    if norm in LEXICALIZED:
        meta = LEXICALIZED[norm]
        meta_dialects = list(meta.get("dialects") or [])
        if requested and meta_dialects and not (set(requested) & set(meta_dialects)):
            return TokenResult(line, original, norm, "LEXICALIZED_HISTORICAL", "dialect_conflict_no_analysis", None, [], [],
                               history_ref=meta.get("id"), history_class=meta.get("class"),
                               history_confidence=meta.get("confidence"),
                               history_context_required=meta.get("activation_mode") == "context_required",
                               history_source_ids=list(meta.get("source_ids") or []),
                               requested_dialects=requested, dialect_constraint_status=dstatus,
                               rejected_by_dialect_count=1, lexicon_constraint_status="applied_hard")
        context_required = meta.get("activation_mode") == "context_required"
        confidence = "medium" if context_required else "high"
        notes = ["lexicalized_historical_form", "lhf_catalog:" + str(meta.get("id"))]
        if context_required:
            notes.append("context_required:" + json.dumps(meta.get("context_trigger"), ensure_ascii=False))
            notes.append("productive_segmentation_blocked_pending_sequence_context")
        c = Candidate(norm, [Segment(norm, "lemma")], 100.0, confidence, notes, [], meta_dialects or list(DIALECT_ALL))
        # The LHF catalog itself licenses the canonical lexicalized lemma.
        # A KRPS headword may enrich provenance, but absence from KRPS (e.g. исне)
        # must not erase the resolved lemma supplied by the verified LHF catalog.
        c.resolved_lemma = norm
        c.lemma_dialects = list(meta_dialects)
        entry = lexicon.get(norm)
        if entry:
            c.lemma_dialects = list(entry.dialects or meta_dialects)
            c.lexicon_match_type = "exact_headword+lhf_catalog"
            c.lexicon_source_rows = list(entry.source_rows)
        else:
            c.lexicon_match_type = "lhf_exact_catalog"
        c.analysis_domain = "lexicalized_historical"
        c.paradigm = "LHF.CONTEXT_REQUIRED" if context_required else "LHF.ATOMIC"
        c.source_support = list(meta.get("source_ids") or [])
        return TokenResult(line, original, norm,
                           "LEXICALIZED_HISTORICAL_CONTEXT_REQUIRED" if context_required else "LEXICALIZED_HISTORICAL",
                           "lexicalized_context_required" if context_required else "lexicalized",
                           c, [], experimental_derivation(norm), history_ref=meta.get("id"),
                           history_class=meta.get("class"), history_confidence=meta.get("confidence"),
                           history_context_required=context_required,
                           history_source_ids=list(meta.get("source_ids") or []),
                           requested_dialects=requested, dialect_constraint_status=dstatus,
                           lexicon_constraint_status="lhf_catalog_exact")

    # Dictionary verbal stems themselves can expose an attested derivational chain.
    if norm.endswith("-"):
        entry=lexicon.get(norm)
        if entry and lexicon.dialect_compatible(entry,requested):
            base=norm[:-1]
            c=Candidate(base,[Segment(base,"verbal_stem")],100.0,"high",["krps_exact_verbal_headword"],[],list(entry.dialects or DIALECT_ALL),
                        resolved_lemma=norm,lemma_dialects=list(entry.dialects),lexicon_match_type="exact_headword",lexicon_source_rows=list(entry.source_rows),
                        analysis_domain="verbal",paradigm="VERB.LEMMA",derivational_chain=lexicon.derivational_chain(norm),source_support=["KRPS"])
            _expand_krps_derivation(c)
            return TokenResult(line,original,norm,"VERBAL_LEMMA_STEM","lexicon_exact_verbal_lemma",c,[],experimental_derivation(base),
                               requested_dialects=requested,dialect_constraint_status=dstatus,lexicon_constraint_status="applied_hard")
        return TokenResult(line,original,norm,"VERBAL_LEMMA_STEM","lexicon_unresolved_verbal_lemma",None,[],experimental_derivation(norm[:-1]),
                           requested_dialects=requested,dialect_constraint_status=dstatus,lexicon_constraint_status="applied_hard",rejected_by_lexicon_count=1)

    # v0.5: full productive verbal generation is run before exact non-verbal
    # precedence, because KRPS can list infinitives/verbal nouns as headwords too.
    if historical_profile is not None and historical_profile not in HISTORICAL_PROFILES:
        raise ValueError("Unsupported historical_profile. Use None or 'historical_western'.")
    verbal = generate_verbal_candidates(norm, lexicon, requested, historical_profile=historical_profile)
    special = generate_v06_special_candidates(norm, lexicon, requested)

    exact_entry = lexicon.get(norm)
    exact_candidate = None
    if exact_entry and lexicon.dialect_compatible(exact_entry, requested):
        exact_candidate = Candidate(norm,[Segment(norm,"lemma")],100.0,"high",["krps_exact_headword"],[],list(exact_entry.dialects or DIALECT_ALL),
                                    resolved_lemma=norm,lemma_dialects=list(exact_entry.dialects),lexicon_match_type="exact_headword",lexicon_source_rows=list(exact_entry.source_rows),
                                    analysis_domain="lexical",paradigm="LEXEME",source_support=["KRPS"])
        if verbal:
            for c in verbal:
                c.notes.append("surface_is_also_krps_headword")

    # Build nominal candidates using the v0.4 engine.
    nominal: List[Candidate] = []
    for stem,segs,score,notes,profiles in detect_p3_special_case(norm):
        nominal.extend(add_plural_prefix_to_analysis(stem,segs,score,notes,profiles))
    for stem,segs,score,notes,profiles in detect_non3poss_dative(norm):
        nominal.extend(add_plural_prefix_to_analysis(stem,segs,score,notes,profiles))
    for before_case,case_seg,case_score,case_notes,case_profiles in detect_regular_case(norm):
        nominal.extend(add_plural_prefix_to_analysis(before_case,[case_seg],case_score,case_notes,case_profiles))
        for before_poss,pseg,pscore,pnotes,poss_profiles in detect_possessive(before_case):
            profiles=intersect_profiles(case_profiles,poss_profiles)
            if not profiles: continue
            bonus=0.15
            if pseg.morpheme == "POSS.1PL" or pseg.morpheme.startswith("POSS.2PL"): bonus=0.9
            if pseg.morpheme.startswith("POSS.2SG"): bonus=-1.8
            nominal.extend(add_plural_prefix_to_analysis(before_poss,[pseg,case_seg],case_score+pscore+bonus,
                                                            case_notes+pnotes+["possessive_before_case"],profiles))
    for before_poss,pseg,pscore,pnotes,profiles in detect_possessive(norm):
        nominal.extend(add_plural_prefix_to_analysis(before_poss,[pseg],pscore,pnotes,profiles))
    for stem,plseg,plscore,plnotes in detect_plural(norm):
        nominal.append(Candidate(stem,[Segment(stem,"stem"),plseg],plscore,"",plnotes,restore_lemma_candidates(stem, False),list(DIALECT_ALL)))

    nominal,rejected_dialect,dstatus2=apply_dialect_constraint(nominal,requested)
    nominal=_dedupe_candidates(nominal)
    nominal,rejected_lexicon=apply_lexicon_constraint(nominal,lexicon,requested)

    # Exact lexical headword blocks accidental nominal oversegmentation, as in
    # v0.4. A source-supported verbal analysis, however, is retained as the main
    # analysis for recognizable verbal morphology (e.g. an infinitive that also
    # has its own KRPS headword).
    if verbal:
        # A bare 2SG imperative is formally identical to the verbal stem and can
        # be homographic with an independent nominal headword. Without syntax,
        # preserve the lexical reading as primary when *all* verbal readings are bare imperatives.
        only_bare_imp = all(c.paradigm == "MOOD.IMPERATIVE.2SG" for c in verbal)
        # v0.6 fallback derivation is deliberately conservative. If the surface
        # token itself is an exact KRPS lexeme and every verbal reading depends
        # on an inferred productive derivation (rather than an attested derived
        # verbal headword), the exact lexeme stays primary and the verbal reading
        # is exposed only as an alternative.
        only_inferred_derivation = all("+productive_derivation" in (c.lexicon_match_type or "") for c in verbal)
        if exact_candidate and (only_bare_imp or only_inferred_derivation):
            verbal_sorted=_dedupe_candidates(verbal)
            return TokenResult(line,original,norm,"LEXICAL_OR_IMPERATIVE","ambiguous_exact_lexeme_or_bare_imperative",
                               exact_candidate,verbal_sorted[:5],experimental_derivation(norm),
                               requested_dialects=requested,dialect_constraint_status=dstatus,
                               rejected_by_dialect_count=rejected_dialect,lexicon_constraint_status="applied_hard",
                               rejected_by_lexicon_count=rejected_lexicon)
        candidates=_dedupe_candidates(verbal + special + nominal)
        best=candidates[0]
        alternatives=candidates[1:6]
        if exact_candidate:
            alternatives=(alternatives+[exact_candidate])[:6]
        return TokenResult(line,original,norm,"VERBAL_FORM","segmented_verbal_high" if best.confidence=="high" else "segmented_verbal_candidate",
                           best,alternatives,experimental_derivation((best.resolved_lemma or best.stem_surface).rstrip("-")),
                           requested_dialects=requested,dialect_constraint_status=dstatus,
                           rejected_by_dialect_count=rejected_dialect,lexicon_constraint_status="applied_hard",
                           rejected_by_lexicon_count=rejected_lexicon)

    if exact_candidate:
        return TokenResult(line,original,norm,"KRPS_LEMMA","lexicon_exact_lemma",exact_candidate,[],experimental_derivation(norm),
                           requested_dialects=requested,dialect_constraint_status=dstatus,
                           lexicon_constraint_status="applied_hard")

    if special or nominal:
        combined=_dedupe_candidates(special+nominal)
        best=combined[0]
        token_class={"pronominal":"PRONOUN_FORM","predicate":"PREDICATIVE_FORM","derivational":"DERIVED_FORM"}.get(best.analysis_domain,"NOMINAL_OR_OTHER")
        return TokenResult(line,original,norm,token_class,"segmented_high" if best.confidence=="high" else "segmented_candidate",
                           best,combined[1:6],experimental_derivation(best.resolved_lemma or best.stem_surface),
                           requested_dialects=requested,dialect_constraint_status=dstatus2,
                           rejected_by_dialect_count=rejected_dialect,lexicon_constraint_status="applied_hard",
                           rejected_by_lexicon_count=rejected_lexicon)

    decision="no_lexically_valid_productive_analysis"
    if rejected_lexicon: decision="lexicon_constraint_removed_all_candidates"
    elif requested and rejected_dialect: decision="dialect_constraint_removed_all_candidates"
    return TokenResult(line,original,norm,"UNRESOLVED_OR_LEXICAL",decision,None,[],experimental_derivation(norm),
                       requested_dialects=requested,dialect_constraint_status=dstatus,
                       rejected_by_dialect_count=rejected_dialect,lexicon_constraint_status="applied_hard",
                       rejected_by_lexicon_count=rejected_lexicon)

# ---------------------------------------------------------------------------
# Serialization / CLI
# ---------------------------------------------------------------------------

def candidate_to_dict(c: Optional[Candidate]) -> Optional[dict]:
    if c is None:
        return None
    d = asdict(c)
    d["category_sequence"] = c.category_sequence()
    d["segmented"] = c.segmented()
    return d


def result_to_dict(r: TokenResult) -> dict:
    return {
        "line": r.line,
        "surface_original": r.surface_original,
        "surface_canonical": r.surface_canonical,
        "token_class": r.token_class,
        "decision": r.decision,
        "requested_dialects": r.requested_dialects,
        "dialect_constraint_status": r.dialect_constraint_status,
        "rejected_by_dialect_count": r.rejected_by_dialect_count,
        "lexicon_constraint_status": r.lexicon_constraint_status,
        "rejected_by_lexicon_count": r.rejected_by_lexicon_count,
        "best": candidate_to_dict(r.best),
        "alternatives": [candidate_to_dict(c) for c in r.alternatives],
        "experimental_derivation": r.experimental_derivation,
        "history_ref": r.history_ref,
        "history_class": r.history_class,
        "history_confidence": r.history_confidence,
        "history_context_required": r.history_context_required,
        "history_source_ids": r.history_source_ids,
    }



def _sequence_tokenize(text: str) -> List[str]:
    """Whitespace tokenization for analytic verbal constructions only.

    This intentionally does not normalize spelling/script; it merely removes
    surrounding punctuation so canonical word forms can be passed unchanged to
    the single-token analyzer.
    """
    import re
    out=[]
    for raw in str(text).split():
        tok=re.sub(r'^[\s\.,;:!?…„“”«»()\[\]{}]+|[\s\.,;:!?…„“”«»()\[\]{}]+$', '', raw)
        if tok:
            out.append(tok)
    return out


def analyze_sequence(text: str, lexicon: KRPSLexicon, dialect: Optional[str] = None,
                     dialects: Optional[Iterable[str]] = None, historical_profile: Optional[str] = None) -> Dict[str, Any]:
    """Analyze selected spaced/analytic verbal constructions.

    The sequence layer never repairs tokens. It reuses analyze_token() and only
    combines adjacent analyses that are independently legal under the same hard
    dialect/lexicon constraints.
    """
    requested=normalize_dialects(dialect=dialect,dialects=dialects)
    toks=_sequence_tokenize(text)
    results=[analyze_token(t,lexicon,line=i+1,dialects=requested,historical_profile=historical_profile) for i,t in enumerate(toks)]
    seqs=[]
    cop_surfaces=set(COP_PAST_3SG) | {x for _,forms,_ in COP_PAST_PERSONAL for x in forms}
    for i in range(len(toks)-1):
        a,b=toks[i],toks[i+1]
        ra,rb=results[i],results[i+1]
        ca=ra.best
        # AOR + EDI: intraterminal/habitual past.
        if ca and ca.analysis_domain=="verbal" and ca.paradigm in {"IND.AOR_FUT","IND.AOR_FUT.NEG"} and b in cop_surfaces:
            seqs.append({"start":i,"end":i+1,"tokens":[a,b],"paradigm":"IND.PAST.INTRATERMINAL.ANALYTIC",
                         "head_lemma":ca.resolved_lemma,"source_support":["CSATO2012_T"]})
        # GAN + EDI: pluperfect.
        if ca and ca.analysis_domain=="verbal" and ca.paradigm in {"PTCP.PAST","PTCP.PAST.NEG"} and b in cop_surfaces:
            seqs.append({"start":i,"end":i+1,"tokens":[a,b],"paradigm":"IND.PLUPERFECT.GAN_EDI.ANALYTIC",
                         "head_lemma":ca.resolved_lemma,"source_support":["CSATO2012_T","HIST_W"]})
        # (I)P + EDI: Western second pluperfect.
        if historical_profile == "historical_western" and ca and ca.analysis_domain=="verbal" and ca.paradigm=="CVB.IP" and b in cop_surfaces and (not requested or "H" in requested):
            seqs.append({"start":i,"end":i+1,"tokens":[a,b],"paradigm":"IND.PLUPERFECT.P_EDI.ANALYTIC.HISTORICAL_WESTERN",
                         "head_lemma":ca.resolved_lemma,"source_support":["HIST_W"],"historical_profile":"historical_western"})
        # Analytic potential: finite potential of bol- + lexical verb in VN.MA.
        if ca and ca.resolved_lemma=="бол-" and "MOD.POT" in ca.category_sequence() and rb.best and rb.best.paradigm=="VN.MA":
            seqs.append({"start":i,"end":i+1,"tokens":[a,b],"paradigm":"MOD.POT.ANALYTIC",
                         "head_lemma":rb.best.resolved_lemma,"aux_lemma":"бол-","source_support":["CSATO2012_T"]})
    return {"engine_version":VERSION,"tokens":toks,"token_results":[result_to_dict(r) for r in results],"sequence_analyses":seqs}

def analyze_file(path: Path, lexicon: KRPSLexicon,
                 dialect: Optional[str] = None, dialects: Optional[Iterable[str]] = None,
                 historical_profile: Optional[str] = None) -> List[TokenResult]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [analyze_token(tok, lexicon=lexicon, line=i + 1, dialect=dialect, dialects=dialects, historical_profile=historical_profile)
            for i, tok in enumerate(lines)]


def write_csv(results: List[TokenResult], path: Path) -> None:
    fields = ["line", "surface_original", "surface_canonical", "requested_dialects", "dialect_constraint_status",
              "rejected_by_dialect_count", "lexicon_constraint_status", "rejected_by_lexicon_count",
              "token_class", "decision", "confidence", "segmented", "categories",
              "stem_surface", "resolved_lemma", "lemma_dialects", "lexicon_match_type", "lexicon_source_rows",
              "lemma_restore_candidates", "candidate_dialect_profiles", "analysis_domain", "paradigm",
              "derivational_chain", "source_support", "notes", "history_ref", "history_class",
              "history_confidence", "history_context_required", "history_source_ids", "derivation_candidates"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results:
            b = r.best
            w.writerow({
                "line": r.line,
                "surface_original": r.surface_original,
                "surface_canonical": r.surface_canonical,
                "requested_dialects": "|".join(r.requested_dialects) if r.requested_dialects else "UNKNOWN",
                "dialect_constraint_status": r.dialect_constraint_status,
                "rejected_by_dialect_count": r.rejected_by_dialect_count,
                "lexicon_constraint_status": r.lexicon_constraint_status,
                "rejected_by_lexicon_count": r.rejected_by_lexicon_count,
                "token_class": r.token_class,
                "decision": r.decision,
                "confidence": b.confidence if b else "",
                "segmented": b.segmented() if b else "",
                "categories": " > ".join(b.category_sequence()) if b else "",
                "stem_surface": b.stem_surface if b else "",
                "resolved_lemma": b.resolved_lemma if b else "",
                "lemma_dialects": "|".join(b.lemma_dialects) if b else "",
                "lexicon_match_type": b.lexicon_match_type if b else "",
                "lexicon_source_rows": "|".join(map(str, b.lexicon_source_rows)) if b else "",
                "lemma_restore_candidates": " | ".join(b.lemma_restore_candidates) if b else "",
                "candidate_dialect_profiles": " | ".join(b.dialect_profiles) if b else "",
                "analysis_domain": b.analysis_domain if b else "",
                "paradigm": b.paradigm if b else "",
                "derivational_chain": json.dumps(b.derivational_chain, ensure_ascii=False) if b else "",
                "source_support": " | ".join(b.source_support) if b else "",
                "notes": " | ".join(b.notes) if b else "",
                "history_ref": r.history_ref or "",
                "history_class": r.history_class or "",
                "history_confidence": r.history_confidence or "",
                "history_context_required": r.history_context_required,
                "history_source_ids": " | ".join(r.history_source_ids),
                "derivation_candidates": json.dumps(r.experimental_derivation, ensure_ascii=False),
            })


def main() -> None:
    ap = argparse.ArgumentParser(description="Karaim nominal + full verbal morphophonological analyzer v1.1")
    ap.add_argument("input", type=Path, help="UTF-8 file, one pre-cleaned KRPS Karaim token per line")
    ap.add_argument("--dialect", choices=["H", "T", "K", "UNKNOWN"], default="UNKNOWN",
                    help="Single hard dialect constraint (convenience option)")
    ap.add_argument("--dialects", default=None,
                    help="Comma-separated allowed KRPS dialects, e.g. H,T. Overrides --dialect when supplied.")
    ap.add_argument("--json", dest="json_path", type=Path)
    ap.add_argument("--csv", dest="csv_path", type=Path)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--lexicon-csv", type=Path, required=True,
                    help="Full verified KRPS export with haslo_karaimskie,dialekt,forma_gramatyczna")
    ap.add_argument("--external-lexicon-csv", type=Path, default=None,
                    help="Optional vetted canonical non-KRPS lexemes, e.g. proper names")
    ap.add_argument("--attested-lexicon-csv", type=Path, default=None,
                    help="Optional exact source-attested Karaim supplement; requires verified evidence rows")
    ap.add_argument("--lhf-catalog", type=Path, default=None,
                    help="Optional LHF catalog override. Exact canonical surfaces only; defaults to lexicalized_historical_forms_v1.0.json next to the engine.")
    ap.add_argument("--historical-profile", choices=["historical_western"], default=None,
                    help="Opt-in historical morphology. Historical Western variants are never active by default.")
    args = ap.parse_args()

    global LEXICALIZED
    if args.lhf_catalog:
        LEXICALIZED = load_lhf_catalog(args.lhf_catalog)

    lexicon = KRPSLexicon.from_csv(args.lexicon_csv)
    if args.external_lexicon_csv:
        lexicon.add_external_csv(args.external_lexicon_csv)
    if args.attested_lexicon_csv:
        lexicon.add_attested_csv(args.attested_lexicon_csv)
    if args.dialects:
        requested = [x.strip() for x in args.dialects.split(",") if x.strip()]
        dialect = None
    else:
        requested = []
        dialect = None if args.dialect == "UNKNOWN" else args.dialect
    results = analyze_file(args.input, lexicon=lexicon, dialect=dialect, dialects=requested, historical_profile=args.historical_profile)

    if args.json_path:
        effective = normalize_dialects(dialect=dialect, dialects=requested)
        args.json_path.write_text(json.dumps({"engine_version": VERSION, "dialects": effective or ["UNKNOWN"],
                                              "historical_profile": args.historical_profile,
                                              "lexicon_stats": lexicon.stats(),
                                              "results": [result_to_dict(r) for r in results]},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_path:
        write_csv(results, args.csv_path)
    if args.do_print:
        for r in results:
            b = r.best
            print(r.line, r.surface_original, f"[{'|'.join(r.requested_dialects) if r.requested_dialects else 'UNKNOWN'}]", "=>",
                  b.segmented() if b else "—", b.category_sequence() if b else [], r.decision,
                  f"rejected_dialect={r.rejected_by_dialect_count}", f"rejected_lexicon={r.rejected_by_lexicon_count}")


if __name__ == "__main__":
    main()