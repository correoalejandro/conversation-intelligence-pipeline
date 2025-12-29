"""Registry‑aware Spanish text preprocessing (verbose edition)
================================================================

Extiende **2.v7.preprocess_spanish_text.py** para procesar los `.json`
referenciados en el *Business Registry* y añade **salida de depuración
inteligible**.

Core workflow (sin cambios):
1. Encuentra registros `conversation_record` (o filtrados por `--batch`).
2. Carga el archivo JSON (`data_ref`) y toma **solo el campo `raw`**.
3. Limpia → corrige → lematiza.
4. Genera snapshots de vocabulario y registra el corpus limpio como
   `preprocess:cleaned_json`.

Novedades de depuración
----------------------
* `-v / --verbose` imprime:
  - Ruta del registry cargado y número total de registros.
  - Estadísticas de filtrado (por *stage*, `batch_id`, existencia de archivo).
  - Muestra las primeras rutas de archivo aceptadas/omitidas.
* Errores detallados si falta `raw` o el JSON no existe.
"""
from __future__ import annotations
from functools import lru_cache

from tqdm import tqdm
import json
import logging
import re
import sys
sys.path.append("c:/Projects/clasificador_mensajes")
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import click
import pandas as pd

from unidecode import unidecode

# ────────────────────────────────
# 🗂  Data‑layer helpers
# ────────────────────────────────
from _data_layer import api as dl_api
from _data_layer import registry as dl_registry

import spacy

# ────────────────────────────────
from spellchecker import SpellChecker


# Hunspell ≫ pyspellchecker (≈ 10‑50 × más rápido en español)
#
# ▸ Instalación rápida Linux / WSL
#     sudo apt-get install hunspell hunspell-es
#     pip install hunspell
# ▸ Instalación rápida Mac
#     brew install hunspell
#     brew install hunspell-es
#     pip install hunspell
# ▸ Windows
#     1. pip install hunspell
#     2. Descarga es_ES.dic y es_ES.aff desde
#        https://github.com/LibreOffice/dictionaries/tree/master/es
#     3. Colócalos en  ./dictionaries/es_ES.*  o en %HUNSPELL_PATH%
#
# El código busca automáticamente en:
#   /usr/share/hunspell  |  /usr/share/myspell  |  ./dictionaries
#import hunspell

#_hunspell: "hunspell.HunSpell | None" = None







# ────────────────────────────────
# 🔠 Regexes & stop‑words
# ────────────────────────────────
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

EXTRA_STOP = {
    "hola", "buenas", "gracias", "listo", "ok", "vale", "claro", "si", "sí",
    "sr", "sra", "usted", "uds", "señor", "señora","favor", "llamada", "deuda", "pago", "cuenta",
    "banco", "montos", "número", "contacto",
    "dígame", "porfa", "ajá", "eh", "pues","agente", "cliente", "javier", "laura","luis", "carlos","ana","yo", "plan", "saldo","app","realizar","lunes","martes","miércoles","jueves","viernes","sábado","domingo"
}

_nlp: "spacy.Language | None" = None
_spell: SpellChecker | None = None

# ────────────────────────────────
# 🔧 Load heavy models lazily
# ────────────────────────────────

'''

def _load_hunspell() -> "hunspell.HunSpell":
    """Lazy‑load Hunspell con diccionario español."""
    global _hunspell  # noqa: PLW0603
    if _hunspell is None:
        from pathlib import Path

        search = [
            Path("/usr/share/hunspell"),
            Path("/usr/share/myspell"),
            Path(__file__).with_suffix("").parent / "dictionaries",
        ]
        for root in search:
            dic = root / "es_ES.dic"
            aff = root / "es_ES.aff"
            if dic.exists() and aff.exists():
                _hunspell = hunspell.HunSpell(str(dic), str(aff))
                break
        if _hunspell is None:
            raise FileNotFoundError(
                "No se encontró es_ES.dic/.aff. Instala 'hunspell-es' o "
                "coloca los archivos en ./dictionaries"
            )
    return _hunspell


def spell_correct(tokens: List[str]) -> List[str]:
    """Corrige tokens usando Hunspell (≈ O(1) lookup + sugerencia)."""
    h = _load_hunspell()
    out: List[str] = []
    for w in tokens:
        if not h.spell(w):
            sugs = h.suggest(w)
            out.append(sugs[0] if sugs else w)
        else:
            out.append(w)
    return out


'''

def _load_spacy() -> "spacy.Language":
    global _nlp  # noqa: PLW0603
    if _nlp is None:
        _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    return _nlp


def _load_spell() -> SpellChecker:
    global _spell  # noqa: PLW0603
    if _spell is None:
        _spell = SpellChecker(language="es")
    return _spell

@lru_cache(maxsize=50000)
def _corr_cached(word: str) -> str:
    """Corrige una sola palabra con cache LRU."""
    spell = _load_spell()
    if word in spell:
        return word
    suggestion = spell.correction(word)
    return suggestion or word


def fast_correct_doc(doc: "spacy.tokens.Doc") -> str:
    """Corrige solo tokens OOV alfabéticos (>2 letras) aprovechando cache."""
    out = []
    for tok in doc:
        if tok.is_alpha and tok.is_oov and len(tok) > 2:
            out.append(_corr_cached(tok.text))
        else:
            out.append(tok.text)
    return " ".join(out)
# ────────────────────────────────
# 🔧 Text cleaning primitives
# ────────────────────────────────

def clean_text(text: str, *, keep_accents: bool = False, strip_emojis: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    if strip_emojis:
        text = EMOJI_RE.sub(" ", text)
    text = text.lower()
    if not keep_accents:
        text = unidecode(text)
    return re.sub(r"\s+", " ", text).strip()


def spell_correct(tokens: List[str]) -> List[str]:
    spell = _load_spell()
    out: List[str] = []
    for w in tokens:
        if w not in spell and spell.correction(w):
            out.append(spell.correction(w))
        else:
            out.append(w)
    return out


def lemmatise(text: str) -> List[str]:
    nlp = _load_spacy()
    doc = nlp(text)
    sw = nlp.Defaults.stop_words | EXTRA_STOP
    return [t.lemma_ for t in doc if t.is_alpha and len(t) > 1 and t.lemma_ not in sw]

# ────────────────────────────────
# 📜 Registry helpers
# ────────────────────────────────

def _iter_conversations(batch_id: str | None = None, *, verbose: bool = False) -> Iterable[Tuple[str, str, str]]:
    """Yield (conversation_id, data_ref_path, batch_id) from Business registry."""

    # 🚚 Carga todos los registros con el helper de alto nivel
    registry = dl_registry.find()          # ← antes _load(_BUSINESS_PATH)
    if verbose:
        logging.info("📖 Total registry records: %d", len(registry))

    kept, skipped = 0, 0
    for rec in registry:
        if rec.get("stage") != "conversation_record":
            skipped += 1
            continue
        if batch_id and rec.get("batch_id") != batch_id:
            skipped += 1
            continue
        data_ref = rec.get("data_ref") or ""
        if data_ref.endswith(".json") and Path(data_ref).exists():
            kept += 1
            if verbose and kept <= 5:
                logging.info("✔ Using %s", data_ref)
            yield rec["conversation_id"], data_ref, rec["batch_id"]
        else:
            skipped += 1
            if verbose and skipped <= 5:
                logging.warning("✖ Skipping: file not found or not JSON → %s", data_ref)

    if verbose:
        logging.info("🏷️  Filter stats → kept: %d | skipped: %d", kept, skipped)

# ────────────────────────────────
# 🏗  Data‑frame construction
# ────────────────────────────────

def _conversation_to_row(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "raw" not in obj or not obj["raw"]:
        raise ValueError(f"No 'raw' field in conversation file: {path}")
    return {
        "conversation_id": obj.get("conversation_id"),
        "batch_id": obj.get("batch_id"),
        "raw": obj["raw"],
        "messages": obj.get("messages", []),
        "text": obj["raw"],
    }

# ────────────────────────────────
# 💾 Persistence helpers
# ────────────────────────────────

def _save_vocab_snapshot(stage: str, token_lists: List[List[str]]):
    flat = [tok for lst in token_lists for tok in lst if isinstance(tok, str)]
    voc = Counter(flat)
    df = pd.DataFrame(voc.most_common(), columns=["term", "frequency"])
    dl_api.save_artifact(
        df,
        stage=f"vocab:{stage}",
        parameters={"vocab_size": len(voc)},
        backend="json",
        parents=[],
    )

# ────────────────────────────────
# 🚀 CLI entry‑point   (NUEVO: --replace)
# ────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug-single", is_flag=True,
              help="Procesa texto uno‑a‑uno con nlp(text) (lento, solo para debugging)")
@click.option("--batch", "batch_id", help="Process only this batch_id from the registry.")
@click.option("--limit", type=int, default=None, help="Max conversations to load (debug).")
@click.option("--keep-accents/--strip-accents", default=False, show_default=True)
@click.option("--keep-emojis/--strip-emojis", default=False, show_default=True)
@click.option("--skip-spellcheck", is_flag=True, help="Skip spell‑correction step.")
@click.option("--replace", is_flag=True,
              help="Delete existing vocab:* artifacts before saving new snapshots.")
@click.option("-v", "--verbose", is_flag=True, help="Enable detailed logging.")





def main(batch_id: str | None, limit: int | None,
         keep_accents: bool, keep_emojis: bool,
         skip_spellcheck: bool, replace: bool,
         debug_single: bool,                 # ← añade este parámetro
         verbose: bool):                     # ← el resto igual
  # noqa: PLR0913
    """Pull JSONs → clean text → register *4* vocab snapshots (with optional replacement)."""

    # ── logger ─────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(message)s")

    # ── helper para borrar vocab:* existentes ─────────────
    if replace:
        tech_reg = dl_registry.find()
        to_delete = [r for r in tech_reg if str(r.get("stage", "")).startswith("vocab:")]
        for rec in to_delete:
            Path(rec["data_ref"]).unlifnk(missing_ok=True)
            dl_registry.delete(rec["id"])  # ← debe existir delete() en tu registry
        if verbose:
            logging.info("🧹 Deleted %d previous vocab:* artifacts", len(to_delete))

    # ── carga de conversaciones ───────────────────────────
    rows: List[Dict[str, Any]] = []
    for idx, (cid, path, bid) in enumerate(tqdm(_iter_conversations(batch_id, verbose=verbose),
                                            desc="Load convs")):

        if limit and idx >= limit:
            break
        try:
            rows.append(_conversation_to_row(path))
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("💥 Error parsing %s → %s", path, exc)

    if not rows:
        logging.error("❌ No conversations found after filtering.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    logging.info("📦 Loaded %d conversations", len(df))

    # ── etapas de tokens y snapshots ───────────────────────
    raw_tokens = df.text.dropna().apply(str.split).tolist()
    _save_vocab_snapshot("raw", raw_tokens)

    df["cleaned"] = df.text.apply(
        lambda t: clean_text(t, keep_accents=keep_accents, strip_emojis=not keep_emojis)
    )
    cleaned_tokens = df.cleaned.dropna().apply(str.split).tolist()
    _save_vocab_snapshot("cleaned", cleaned_tokens)

    if skip_spellcheck:
        df["corrected"] = df.cleaned
        corrected_tokens = cleaned_tokens
    else:
        texts = df.cleaned.tolist()
        corrected = []
        logging.info("🔍 Spell‑correction (spaCy OOV + cache) on %d texts", len(texts))
        for doc in tqdm(
            _load_spacy().pipe(texts, batch_size=128, disable=["ner"]),
            total=len(texts),
            desc="spell‑correct",
        ):
            corrected.append(fast_correct_doc(doc))
        df["corrected"] = corrected
        corrected_tokens = [c.split() for c in corrected]
        _save_vocab_snapshot("corrected", corrected_tokens)

    # ───────── Lematización rápida vs debug ─────────
    if debug_single:
        # versión por documento (lenta, útil para depurar)
        df["lemmatised"] = df.corrected.apply(lambda txt: " ".join(lemmatise(txt)))
    else:
        # versión por lotes con nlp.pipe()
        texts = df.corrected.tolist()
        logging.info("🔠 Starting spaCy lemmatization on %d texts (batch_size=128)", len(texts))
        lemmatised = []

        for i, doc in enumerate(tqdm(_load_spacy().pipe(texts, batch_size=128, disable=["ner"]),
                                    total=len(texts), desc="spaCy lemmatise")):
            if verbose and i % 100 == 0 and i > 0:
                logging.info("📘 Processed %d/%d docs with spaCy", i, len(texts))

            sw = _nlp.Defaults.stop_words | EXTRA_STOP
            lemmas = [t.lemma_ for t in doc if t.is_alpha and len(t) > 1 and t.lemma_ not in sw]
            lemmatised.append(" ".join(lemmas))
        # 👉 guarda la lista como nueva columna antes de usarla
        df["lemmatised"] = lemmatised
        logging.info("✅ Finished spaCy lemmatization")
        logging.info("✅ Finished spaCy lemmatization")



    lemm_tokens = df.lemmatised.dropna().apply(str.split).tolist()
    _save_vocab_snapshot("lemmatised", lemm_tokens)

    art = dl_api.save_artifact(
        df,
        stage="preprocess:cleaned_json",
        parameters={
            "source": "conversation_record",
            "batch_filter": batch_id,
            "keep_accents": keep_accents,
            "keep_emojis": keep_emojis,
            "skip_spellcheck": skip_spellcheck,
        },
        backend="json",
        parents=[],
    )
    logging.info("🎉 Preprocessed conversations registered → %s", art.id)


if __name__ == "__main__":
    main()
