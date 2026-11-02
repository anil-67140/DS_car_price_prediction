"""Deterministic parsing of free-text listing names.

Everything here is a pure function of the listing text, so it is available for future unseen
vehicles at prediction time (no target information is used). It is shared by cleaning, the
feature builder and the application dropdowns, so train/serve behaviour cannot diverge.
"""

from __future__ import annotations

import re

import pandas as pd

# canonical brand -> lowercase aliases found in the wild ----------------------------------------
_BRAND_ALIASES: dict[str, list[str]] = {
    "Maruti": ["maruti", "maruti suzuki", "marutisuzuki"],
    "Hyundai": ["hyundai"],
    "Mahindra": ["mahindra"],
    "Tata": ["tata"],
    "Honda": ["honda"],
    "Toyota": ["toyota"],
    "Chevrolet": ["chevrolet"],
    "Renault": ["renault"],
    "Ford": ["ford"],
    "Volkswagen": ["volkswagen", "vw"],
    "Skoda": ["skoda"],
    "Audi": ["audi"],
    "Datsun": ["datsun"],
    "Mini": ["mini"],
    "BMW": ["bmw"],
    "Mitsubishi": ["mitsubishi"],
    "Nissan": ["nissan"],
    "Mercedes-Benz": ["mercedes", "mercedes-benz", "mercedes benz"],
    "Fiat": ["fiat"],
    "Force Motors": ["force", "force motors"],
    "Hindustan Motors": ["hindustan", "hindustan motors"],
    "Jaguar": ["jaguar"],
    "Land Rover": ["land", "land rover"],
    "Jeep": ["jeep"],
    "Volvo": ["volvo"],
    "Kia": ["kia"],
    "MG": ["mg", "mg motor"],
    "Isuzu": ["isuzu"],
    "Lexus": ["lexus"],
    "Porsche": ["porsche"],
    "Ambassador": ["ambassador"],
}
ALIAS_TO_BRAND: dict[str, str] = {a: b for b, al in _BRAND_ALIASES.items() for a in al}
KNOWN_BRANDS: frozenset[str] = frozenset(_BRAND_ALIASES)

# leading tokens that belong to the brand, not to the model
_BRAND_TOKENS: dict[str, set[str]] = {
    "Maruti": {"maruti", "suzuki"},
    "Mercedes-Benz": {"mercedes", "benz", "mercedes-benz"},
    "Land Rover": {"land", "rover"},
    "Force Motors": {"force", "motors"},
    "Hindustan Motors": {"hindustan", "motors"},
    "MG": {"mg", "motor"},
}

# explicit two-token model names; avoids merging e.g. "Swift VDI" (variant) with "Swift Dzire"
_COMPOUND_PAIRS: dict[tuple[str, str], str] = {
    ("wagon", "r"): "Wagon R",
    ("grand", "i10"): "Grand i10",
    ("elite", "i20"): "Elite i20",
    ("alto", "k10"): "Alto K10",
    ("alto", "800"): "Alto 800",
    ("corolla", "altis"): "Corolla Altis",
    ("etios", "liva"): "Etios Liva",
    ("santro", "xing"): "Santro Xing",
    ("vitara", "brezza"): "Vitara Brezza",
    ("swift", "dzire"): "Dzire",  # sold both as "Swift Dzire" and "Dzire": same car
    ("indica", "v2"): "Indica V2",
    ("indica", "ev2"): "Indica V2",
    ("grand", "vitara"): "Grand Vitara",
    ("xuv", "500"): "XUV500",
}
_SERIES_WORDS = {"series", "class"}

_AUTO_TOKENS = {"amt", "at", "cvt", "dsg", "dct", "tiptronic", "automatic", "auto", "s-tronic"}
_MANUAL_TOKENS = {"mt", "manual"}
_ENGINE_RE = re.compile(r"(?<![\d.])([0-6]\.\d)(?:\s?l\b|\b)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9.\-+/]+")


def canonical_brand(company: str | None, name: str | None = None) -> str | None:
    """Map messy brand strings onto a canonical brand, else None (unknown / not a car brand)."""
    for text in (company, (name or "").split(" ")[0] if name else None):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        t = str(text).strip().lower()
        if t in ALIAS_TO_BRAND:
            return ALIAS_TO_BRAND[t]
        if t.rstrip(".") in ALIAS_TO_BRAND:
            return ALIAS_TO_BRAND[t.rstrip(".")]
    return None


def _tokens(name: str) -> list[str]:
    return _TOKEN_RE.findall(str(name))


def strip_brand(name: str, brand: str | None) -> list[str]:
    toks = _tokens(name)
    if not toks:
        return []
    drop = _BRAND_TOKENS.get(brand or "", set()) | ({brand.lower()} if brand else set())
    i = 0
    while i < len(toks) and toks[i].lower() in drop:
        i += 1
    return toks[i:]


def model_family(name: str, brand: str | None, n_max: int = 2) -> str:
    """Brand-aware model family, e.g. 'Maruti Suzuki Swift Dzire VDi' -> 'Dzire'."""
    toks = strip_brand(name, brand)
    if not toks:
        return "Unknown"
    t0 = toks[0]
    if len(toks) > 1:
        t1 = toks[1]
        pair = (t0.lower(), t1.lower())
        if n_max >= 2 and pair in _COMPOUND_PAIRS:
            return _COMPOUND_PAIRS[pair]
        if n_max >= 2 and t1.lower() in _SERIES_WORDS:
            return f"{t0.upper() if len(t0) <= 3 else t0} {t1.capitalize()}"
    return {"xuv500": "XUV500"}.get(t0.lower(), t0)


def variant_text(name: str, brand: str | None) -> str:
    return " ".join(strip_brand(name, brand))


def parse_engine_litres(name: str) -> float | None:
    """Engine displacement in litres if the listing text states it (e.g. '1.2', '1.5L')."""
    m = _ENGINE_RE.search(str(name))
    if not m:
        return None
    val = float(m.group(1))
    return val if 0.6 <= val <= 6.9 else None


def parse_transmission_hint(name: str) -> str:
    """'automatic' / 'manual' ONLY when the text says so; otherwise 'unknown'.

    Absence of a token is NOT evidence of a manual gearbox, so it is never mapped to 'manual'.
    """
    toks = {t.lower() for t in _tokens(name)}
    if toks & _AUTO_TOKENS:
        return "automatic"
    if toks & _MANUAL_TOKENS:
        return "manual"
    return "unknown"
