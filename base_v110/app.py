from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

import karaim_morph_engine_v1_1 as engine

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load the exact project data shipped with the app.
LEXICON = engine.KRPSLexicon.from_csv(DATA_DIR / "eksport_glowy_dialekt_forma(2).csv")
EXTERNAL = DATA_DIR / "krps_external_lexemes_v0.9.csv"
if EXTERNAL.exists():
    LEXICON.add_external_csv(EXTERNAL)
engine.LEXICALIZED = engine.load_lhf_catalog(DATA_DIR / "lexicalized_historical_forms_v1.0.json")

LHF_META: Dict[str, Dict[str, Any]] = {}
_lhf_raw = json.loads((DATA_DIR / "lexicalized_historical_forms_v1.0.json").read_text(encoding="utf-8"))
for _entry in _lhf_raw.get("entries", []):
    LHF_META[_entry.get("canonical_surface", "")] = _entry

app = FastAPI(title="Karaim Morphology Explorer", version="1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
_jinja = Environment(
    loader=FileSystemLoader(BASE_DIR / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


class AnalyzeRequest(BaseModel):
    word: str = Field(min_length=1, max_length=120)
    dialect: str = Field(default="UNKNOWN")


DIALECT_LABELS = {
    "H": "dialekt łucko-halicki (zachodni)",
    "T": "dialekt trocki",
    "K": "dialekt krymski",
}

CONFIDENCE_LABELS = {
    "high": "wysoka",
    "medium-high": "średnio wysoka",
    "medium": "średnia",
    "low-medium": "średnio niska",
    "low": "niska",
}

ROLE_GROUP = {
    "root": "root",
    "lemma": "root",
    "stem": "root",
    "verbal_stem": "root",
    "derivational": "derivation",
    "aspect": "derivation",
    "modality": "derivation",
    "negation": "negation",
    "tense": "tense",
    "mood": "mood",
    "person": "person",
    "participle": "nonfinite",
    "converb": "nonfinite",
    "infinitive": "nonfinite",
    "plural": "plural",
    "possessive": "possessive",
    "case": "case",
    "copula": "auxiliary",
    "auxiliary": "auxiliary",
    "historical_marker": "historical",
    "linker": "other",
    "clitic": "other",
    "closed_class_paradigm": "other",
}

GROUP_LABELS = {
    "root": "rdzeń / temat leksykalny",
    "derivation": "słowotwórstwo, strona lub aspekt",
    "negation": "negacja",
    "tense": "czas",
    "mood": "tryb",
    "person": "osoba i liczba",
    "nonfinite": "forma nieosobowa czasownika",
    "plural": "liczba mnoga",
    "possessive": "dzierżawczość",
    "case": "przypadek",
    "auxiliary": "kopuła / czasownik pomocniczy",
    "historical": "marker historyczny",
    "other": "inny element gramatyczny",
}

EXACT_MORPHEME = {
    "ROOT": ("rdzeń", "Podstawa leksykalna, od której budowana jest analizowana forma."),
    "VOICE.CAUS": ("przyrostek kauzatywny (sprawczy)", "Tworzy znaczenie: spowodować, że ktoś lub coś wykona daną czynność albo znajdzie się w danym stanie."),
    "VOICE.PASS": ("marker strony biernej", "Wskazuje, że podmiot jest odbiorcą lub obiektem czynności."),
    "VOICE.REFL": ("marker strony zwrotnej", "Wskazuje, że czynność jest skierowana na wykonawcę."),
    "VOICE.RECIP": ("marker strony wzajemnej", "Wskazuje czynność wykonywaną wzajemnie przez co najmniej dwóch uczestników."),
    "NEG": ("marker przeczenia", "Neguje znaczenie czasownika."),
    "NEG.PRES": ("marker przeczenia w czasie teraźniejszym", "Buduje przeczącą formę czasu teraźniejszego."),
    "TENSE.PRES": ("marker czasu teraźniejszego", "Umieszcza czynność w czasie teraźniejszym."),
    "TENSE.AOR": ("marker aorystu", "Tworzy formę aorystyczną, używaną m.in. do znaczeń ogólnych lub habitualnych."),
    "TENSE.AOR.FUT": ("marker aorystu / czasu przyszłego", "Tworzy formę aorystyczną lub przyszłościową zależnie od dialektu i kontekstu."),
    "TENSE.AOR.FUT.NEG": ("przeczący aoryst / czas przyszły", "Łączy znaczenie przeczenia z formą aorystyczną lub przyszłościową."),
    "TENSE.PAST.DI": ("marker czasu przeszłego na -DI", "Oznacza zakończoną czynność w przeszłości."),
    "MOOD.IMP": ("tryb rozkazujący", "Wyraża polecenie, nakaz lub wezwanie."),
    "MOOD.IMPERATIVE.2SG": ("tryb rozkazujący, 2. osoba liczby pojedynczej", "Wyraża polecenie skierowane do jednej osoby."),
    "MOOD.IMPERATIVE.2SG.NEG": ("przeczący tryb rozkazujący, 2. osoba liczby pojedynczej", "Wyraża zakaz lub polecenie niewykonywania czynności."),
    "MOOD.IMPERATIVE.2PL": ("tryb rozkazujący, 2. osoba liczby mnogiej", "Wyraża polecenie skierowane do wielu osób."),
    "MOOD.OPT": ("tryb życzący", "Wyraża życzenie lub pożądany stan rzeczy."),
    "MOOD.OPTATIVE": ("tryb życzący", "Wyraża życzenie lub pożądany stan rzeczy."),
    "MOOD.CONDITIONAL": ("tryb warunkowy", "Wyraża czynność zależną od warunku."),
    "MOOD.COND": ("tryb warunkowy", "Wyraża czynność zależną od warunku."),
    "MOOD.VOL": ("tryb woluntatywny", "Wyraża wolę lub zamiar wykonania czynności."),
    "MOOD.VOLUNTATIVE": ("tryb woluntatywny", "Wyraża wolę lub zamiar wykonania czynności."),
    "MOD.POT": ("marker możliwości", "Wyraża możliwość lub zdolność wykonania czynności."),
    "PERS.1SG": ("1. osoba liczby pojedynczej", "Wskazuje mówiącego jako wykonawcę czynności."),
    "PERS.2SG": ("2. osoba liczby pojedynczej", "Wskazuje jednego adresata jako wykonawcę czynności."),
    "PERS.3SG": ("3. osoba liczby pojedynczej", "Wskazuje jednego uczestnika poza mówiącym i adresatem."),
    "PERS.1PL": ("1. osoba liczby mnogiej", "Wskazuje grupę obejmującą mówiącego."),
    "PERS.2PL": ("2. osoba liczby mnogiej", "Wskazuje wielu adresatów."),
    "PERS.3PL": ("3. osoba liczby mnogiej", "Wskazuje wielu uczestników poza mówiącym i adresatem."),
    "NONFIN.INF": ("bezokolicznik", "Tworzy nieosobową formę czasownika odpowiadającą bezokolicznikowi."),
    "INF": ("bezokolicznik", "Tworzy nieosobową formę czasownika odpowiadającą bezokolicznikowi."),
    "PTCP.UVCI": ("imiesłów / nazwa wykonawcy z rodziny -(I)vČI", "Tworzy formę związaną z wykonawcą czynności lub imiesłowem odczasownikowym."),
    "PTCP.PAST": ("imiesłów przeszły", "Tworzy imiesłów odnoszący się do czynności wcześniejszej."),
    "PTCP.PAST.GAN": ("imiesłów przeszły na -GAN", "Tworzy imiesłów przeszły z rodziny -GAN."),
    "PTCP.PRESENT": ("imiesłów teraźniejszy", "Tworzy imiesłów odnoszący się do trwającej lub charakterystycznej czynności."),
    "CVB.IP": ("imiesłów przysłówkowy na -(I)p", "Łączy czynność z kolejnym zdarzeniem lub czynnością."),
    "CVB.A": ("imiesłów przysłówkowy na -A/-y", "Tworzy formę nieosobową używaną do łączenia czynności."),
    "CVB.GINCHA": ("imiesłów przysłówkowy na -GINčA", "Wyraża relację czasową lub graniczną względem kolejnej czynności."),
    "VN.MA": ("rzeczownik odczasownikowy na -mA", "Nominalizuje czynność, dzięki czemu forma może przyjmować morfologię rzeczownikową."),
    "VN.UV": ("rzeczownik odczasownikowy na -Uv", "Nominalizuje czynność."),
    "VN.ISH": ("rzeczownik odczasownikowy na -(I)š", "Nominalizuje czynność lub proces."),
    "PL": ("liczba mnoga", "Wskazuje więcej niż jeden desygnat."),
    "POSS.1SG": ("dzierżawczość: 1. osoba liczby pojedynczej", "Oznacza relację „mój / moja / moje”."),
    "POSS.2SG": ("dzierżawczość: 2. osoba liczby pojedynczej", "Oznacza relację „twój / twoja / twoje”."),
    "POSS.3": ("dzierżawczość: 3. osoba", "Oznacza relację „jego / jej / ich” zależnie od kontekstu."),
    "CASE.GEN": ("dopełniacz", "Oznacza relację dopełniaczową, często odpowiadającą polskiemu „kogo? czego?”."),
    "CASE.ACC": ("biernik", "Oznacza bezpośredni obiekt czynności: „kogo? co?”."),
    "CASE.DAT": ("celownik", "Oznacza odbiorcę, cel lub kierunek: „komu? czemu? dokąd?”."),
    "CASE.LOC": ("miejscownik / przypadek lokatywny", "Oznacza miejsce lub położenie."),
    "CASE.ABL": ("ablatyw", "Oznacza źródło, oddalenie lub ruch od czegoś."),
    "CASE.INS": ("narzędnik", "Oznacza narzędzie, sposób lub towarzyszenie."),
    "COP.PRED.3SG": ("kopuła orzeczeniowa, 3. osoba liczby pojedynczej", "Tworzy orzeczenie imienne, zbliżone funkcją do polskiego „jest”."),
    "ASP.ITER": ("aspekt iteratywny", "Wskazuje czynność powtarzaną lub wielokrotną."),
    "DERIV.PRIV_SIZ": ("przyrostek prywatywny „bez”", "Tworzy znaczenie braku lub pozbawienia danej właściwości/czynności."),
}

PARADIGM_LABELS = {
    "INF": "bezokolicznik",
    "PTCP.UVCI": "imiesłów / nazwa wykonawcy z rodziny -(I)vČI",
    "MOOD.IMPERATIVE.2SG.NEG": "przeczący tryb rozkazujący, 2. osoba liczby pojedynczej",
    "IND.AOR_FUT": "aoryst / czas przyszły",
    "IND.AOR_FUT.HISTORICAL_WESTERN": "historyczny zachodni aoryst / czas przyszły",
    "LHF.ATOMIC": "zleksykalizowana forma historyczna traktowana współcześnie jako całość",
}

LHF_POLISH = {
    "LHF-001": {
        "now": "Współcześnie forma jest traktowana jako niepodzielna, zleksykalizowana forma honoryfikatywna. Parser nie odtwarza wewnątrz niej dawnych granic morfemów.",
        "history": "Historycznie forma wywodzi się z hebrajskiego wyrazu oznaczającego „honor / chwałę” oraz dawnej morfologii dzierżawczej. W toku rozwoju została skrócona i ponownie zinterpretowana jako forma grzecznościowa.",
    },
    "LHF-002": {
        "now": "Współcześnie forma jest traktowana jako samodzielny postpozycjon. Nie należy jej rozcinać na produktywne końcówki rzeczownikowe.",
        "history": "Historycznie jest to skrócona postać dawnej odmienionej formy dzierżawczej; później została zgramatykalizowana i zaczęła funkcjonować jako postpozycjon.",
    },
    "LHF-003": {
        "now": "W konstrukcji z czasownikiem „эт-/эть-” forma jest traktowana jako stały składnik złożenia. Sam pojedynczy wyraz wymaga kontekstu, dlatego końcowe -у nie jest automatycznie interpretowane jako produktywny karaimski sufiks.",
        "history": "Jest to zapożyczenie z ukraińskiego, w którym utrwaliła się dawna forma przypadka źródłowego i została zachowana jako stały element konstrukcji czasownikowej.",
    },
    "LHF-004": {
        "now": "Współcześnie jest to zleksykalizowana forma grzecznościowa. Dawna morfologia dzierżawcza nie jest traktowana jako produktywny rozbiór synchroniczny.",
        "history": "Forma należy do tej samej rodziny honoryfikatywnej co „кануз” i wywodzi się ostatecznie z hebrajskiego wyrazu oznaczającego „honor”.",
    },
    "LHF-005": {
        "now": "Wyraz jest traktowany jako lemat występujący tylko w liczbie mnogiej, a nie jako zwykłe połączenie niepoświadczonej podstawy z produktywnym sufiksem liczby mnogiej.",
        "history": "Jest to zapożyczenie, w którym karaimska końcówka liczby mnogiej utrwaliła się jako część całej jednostki leksykalnej.",
    },
}


def _group_for_segment(seg: Dict[str, Any]) -> str:
    role = (seg.get("role") or "").lower()
    code = seg.get("morpheme") or ""
    if role in ROLE_GROUP:
        return ROLE_GROUP[role]
    if code.startswith("CASE."):
        return "case"
    if code.startswith("POSS."):
        return "possessive"
    if code.startswith("PERS."):
        return "person"
    if code.startswith("VOICE.") or code.startswith("DERIV.") or code.startswith("ASP.") or code.startswith("MOD."):
        return "derivation"
    if code.startswith("PTCP.") or code.startswith("CVB.") or code.startswith("VN.") or code.startswith("NONFIN.") or code.startswith("INF"):
        return "nonfinite"
    if code.startswith("TENSE."):
        return "tense"
    if code.startswith("MOOD."):
        return "mood"
    if code.startswith("NEG"):
        return "negation"
    return "other"


def _morpheme_info(code: Optional[str], role: str) -> Dict[str, str]:
    if code in EXACT_MORPHEME:
        label, description = EXACT_MORPHEME[code]
        return {"label": label, "description": description}
    if not code:
        group = ROLE_GROUP.get(role, "other")
        return {"label": GROUP_LABELS[group], "description": "Element rozpoznany na podstawie jego funkcji w analizie."}

    if code.startswith("PERS."):
        bits = code.split(".")[-1]
        person = bits[0] if bits and bits[0].isdigit() else "?"
        number = "liczby mnogiej" if bits.endswith("PL") else "liczby pojedynczej"
        return {"label": f"{person}. osoba {number}", "description": "Wskazuje osobę i liczbę wykonawcy czynności."}
    if code.startswith("POSS."):
        return {"label": "końcówka dzierżawcza", "description": "Wskazuje relację posiadania lub przynależności."}
    if code.startswith("CASE."):
        return {"label": "końcówka przypadka", "description": "Wskazuje funkcję składniową lub relację przestrzenną/semantyczną formy."}
    if code.startswith("PTCP."):
        return {"label": "imiesłów", "description": "Tworzy nieosobową formę czasownika o właściwościach czasownikowych i nominalnych/przymiotnikowych."}
    if code.startswith("CVB."):
        return {"label": "imiesłów przysłówkowy", "description": "Tworzy nieosobową formę łączącą daną czynność z inną czynnością lub okolicznością."}
    if code.startswith("VN."):
        return {"label": "rzeczownik odczasownikowy", "description": "Przekształca czynność w formę zachowującą się częściowo jak rzeczownik."}
    if code.startswith("DERIV."):
        return {"label": "przyrostek słowotwórczy", "description": "Tworzy nową podstawę lub zmienia kategorię/znaczenie wyrazu."}
    if code.startswith("VOICE."):
        return {"label": "marker strony czasownika", "description": "Zmienia relację między wykonawcą czynności a jej uczestnikami."}
    if code.startswith("MOOD."):
        return {"label": "marker trybu", "description": "Określa stosunek mówiącego do czynności, np. rozkaz, warunek lub życzenie."}
    if code.startswith("TENSE.") or code.startswith("IND."):
        return {"label": "marker czasu / formy oznajmującej", "description": "Określa relację czasową i sposób przedstawienia zdarzenia."}
    if code.startswith("NEG"):
        return {"label": "marker przeczenia", "description": "Neguje znaczenie czasownika."}
    if code.startswith("COP") or code.startswith("AUX"):
        return {"label": "kopuła lub czasownik pomocniczy", "description": "Pomaga budować złożoną formę gramatyczną lub orzeczenie."}
    return {"label": "element gramatyczny", "description": "Element rozpoznany przez regułę morfologiczną parsera."}


def _paradigm_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value in PARADIGM_LABELS:
        return PARADIGM_LABELS[value]
    # Human-readable fallback without exposing unexplained abbreviations as the main label.
    value_upper = value.upper()
    if "HISTORICAL_WESTERN" in value_upper or "HIST_W" in value_upper:
        return "historyczny wariant zachodnio-karaimski"
    if "IMPERATIVE" in value_upper or "MOOD.IMP" in value_upper:
        return "tryb rozkazujący"
    if "OPT" in value_upper:
        return "tryb życzący"
    if "COND" in value_upper:
        return "tryb warunkowy"
    if "PAST" in value_upper:
        return "forma czasu przeszłego"
    if "PRESENT" in value_upper or "PRES" in value_upper:
        return "forma czasu teraźniejszego"
    if "AOR" in value_upper or "FUT" in value_upper:
        return "aoryst / forma przyszłościowa"
    if "PTCP" in value_upper:
        return "imiesłów"
    if "CVB" in value_upper:
        return "imiesłów przysłówkowy"
    if "INF" in value_upper:
        return "bezokolicznik"
    return "forma morfologiczna rozpoznana przez parser"


def _segment_view(seg: Dict[str, Any]) -> Dict[str, Any]:
    info = _morpheme_info(seg.get("morpheme"), seg.get("role") or "")
    return {
        "text": seg.get("text") or "",
        "role": seg.get("role") or "",
        "morpheme": seg.get("morpheme"),
        "status": seg.get("status") or "",
        "group": _group_for_segment(seg),
        "label": info["label"],
        "description": info["description"],
        "visible": bool(seg.get("text")),
    }


def _analysis_signature(result: Dict[str, Any]) -> Any:
    best = result.get("best")
    if not best:
        return None
    return (
        best.get("segmented"),
        best.get("resolved_lemma"),
        best.get("paradigm"),
        tuple(best.get("category_sequence") or []),
    )


def _build_explanation(raw: Dict[str, Any], historical: bool = False) -> List[str]:
    best = raw.get("best")
    if not best:
        decision = raw.get("decision", "")
        if "dialect_conflict" in decision:
            return ["Parser znalazł możliwą analizę, ale została odrzucona przez wybrane ograniczenie dialektalne."]
        if "lexicon_constraint" in decision:
            return ["Parser wygenerował kandydatów morfologicznych, ale żaden nie spełnił twardego warunku zgodności z leksykonem KRPS."]
        return ["Parser nie znalazł bezpiecznej analizy spełniającej aktualne reguły morfologiczne, leksykalne i dialektalne."]

    lines: List[str] = []
    lemma = best.get("resolved_lemma") or best.get("stem_surface")
    if lemma:
        lines.append(f"Rozpoznana podstawa słownikowa to „{lemma}”.")

    segments = [_segment_view(s) for s in best.get("segments") or []]
    for s in segments:
        if s["group"] == "root":
            continue
        if s["text"]:
            lines.append(f"Segment „{s['text']}” pełni funkcję: {s['label']}. {s['description']}")
        else:
            lines.append(f"Analiza zawiera także kategorię „{s['label']}”, która nie ma w tej formie osobnego widocznego segmentu. {s['description']}")

    dialects = best.get("dialect_profiles") or []
    if dialects:
        dlabels = [DIALECT_LABELS.get(d, d) for d in dialects]
        lines.append("Analiza jest zgodna z profilem: " + ", ".join(dlabels) + ".")

    if historical:
        lines.append("Ta analiza korzysta z jawnie włączonego profilu historycznego zachodniego. Reguły tego profilu nie są używane w domyślnej analizie współczesnej.")

    return lines


def _candidate_view(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidate:
        return None
    segs = [_segment_view(s) for s in candidate.get("segments") or []]
    return {
        "segmented": candidate.get("segmented"),
        "resolved_lemma": candidate.get("resolved_lemma"),
        "confidence": CONFIDENCE_LABELS.get(candidate.get("confidence"), candidate.get("confidence") or "nieokreślona"),
        "analysis_domain": candidate.get("analysis_domain"),
        "paradigm_label": _paradigm_label(candidate.get("paradigm")),
        "segments": segs,
        "features": [s for s in segs if not s["visible"]],
        "source_support": candidate.get("source_support") or [],
    }


def _result_view(result: engine.TokenResult, historical: bool = False) -> Dict[str, Any]:
    raw = engine.result_to_dict(result)
    view = {
        "found": bool(raw.get("best")),
        "decision": raw.get("decision"),
        "token_class": raw.get("token_class"),
        "best": _candidate_view(raw.get("best")),
        "alternatives": [_candidate_view(x) for x in (raw.get("alternatives") or [])[:4]],
        "explanation": _build_explanation(raw, historical=historical),
        "history_ref": raw.get("history_ref"),
        "history_class": raw.get("history_class"),
        "history_confidence": raw.get("history_confidence"),
        "history_context_required": raw.get("history_context_required"),
        "history_source_ids": raw.get("history_source_ids") or [],
        "requested_dialects": raw.get("requested_dialects") or [],
        "signature": _analysis_signature(raw),
    }
    return view


def _lhf_history(word: str) -> Optional[Dict[str, Any]]:
    meta = LHF_META.get(word)
    if not meta:
        return None
    pl = LHF_POLISH.get(meta.get("id"), {})
    return {
        "id": meta.get("id"),
        "class": meta.get("class"),
        "dialects": meta.get("dialects") or [],
        "context_required": meta.get("activation_mode") == "context_required",
        "current_explanation": pl.get("now") or "Forma jest traktowana synchronicznie jako zleksykalizowana jednostka i blokuje automatyczną produktywną segmentację.",
        "historical_explanation": pl.get("history") or "Katalog wskazuje historyczne pochodzenie lub reanalizę tej formy.",
        "historical_source_language": meta.get("historical_source_language"),
        "historical_etymon": meta.get("historical_etymon"),
        "source_ids": meta.get("source_ids") or [],
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    template = _jinja.get_template("index.html")
    return HTMLResponse(template.render(engine_version=engine.VERSION, legend=GROUP_LABELS))


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "engine_version": engine.VERSION, "lexicon_entries": len(LEXICON.entries)}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> Dict[str, Any]:
    word = payload.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="Wpisz karaimskie słowo.")
    if any(ch.isspace() for ch in word):
        raise HTTPException(status_code=400, detail="Ta wersja aplikacji analizuje jedno słowo naraz.")

    dialect = payload.dialect.upper().strip()
    if dialect not in {"UNKNOWN", "H", "T", "K"}:
        raise HTTPException(status_code=400, detail="Nieprawidłowy dialekt.")
    requested = [] if dialect == "UNKNOWN" else [dialect]

    current_r = engine.analyze_token(word, LEXICON, dialects=requested, historical_profile=None)
    historical_r = engine.analyze_token(word, LEXICON, dialects=requested, historical_profile="historical_western")

    current = _result_view(current_r, historical=False)
    historical = _result_view(historical_r, historical=True)
    lhf = _lhf_history(word)

    distinct_history = historical["found"] and historical["signature"] != current["signature"]
    if lhf:
        # LHF is historical information even when both parser modes return the same synchronic atom.
        distinct_history = True

    historical["distinct"] = distinct_history
    historical["available_but_same"] = historical["found"] and not distinct_history

    # Signatures are internal comparison helpers, not UI data.
    current.pop("signature", None)
    historical.pop("signature", None)

    return {
        "word": word,
        "dialect": dialect,
        "dialect_label": "automatyczne rozpoznanie / bez twardego ograniczenia" if dialect == "UNKNOWN" else DIALECT_LABELS[dialect],
        "engine_version": engine.VERSION,
        "current": current,
        "historical": historical,
        "lexicalized_history": lhf,
        "legend": [{"group": k, "label": v} for k, v in GROUP_LABELS.items()],
    }