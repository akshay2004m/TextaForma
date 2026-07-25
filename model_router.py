"""
model_router.py — Intelligent Model Orchestrator
=================================================
Replaces Gemini API in app.py with locally trained models.

Architecture on g5.xlarge (A10G · 24 GB VRAM):
  ┌──────────────────────────────────────────────┐
  │  Request → RoutingEngine → Pipeline Select   │
  │                                              │
  │  PIPELINE A: text formalize (same language)  │
  │    FormalizationModel (mt5-base)             │
  │                                              │
  │  PIPELINE B: translate → formal English      │
  │    TranslationModel → FormalizationModel     │
  │                                              │
  │  PIPELINE C: speech → text                  │
  │    WhisperASR → (A or B)                    │
  └──────────────────────────────────────────────┘

Lazy loading: only one large model in VRAM at a time.
Models are loaded on first use and evicted on memory pressure.
"""

import os, gc, re, json, time, logging, threading
from typing import Optional, Generator, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ROUTER] %(levelname)s — %(message)s",
)
logger = logging.getLogger("model_router")

# ──────────────────────────────────────────────────────────────
# Enums & constants
# ──────────────────────────────────────────────────────────────

class Pipeline(Enum):
    FORMALIZE_SAME_LANG  = "formalize_same_lang"
    TRANSLATE_TO_ENGLISH = "translate_to_english"
    ASR_TRANSCRIBE       = "asr_transcribe"
    EXPLAIN_CHANGES      = "explain_changes"
    FALLBACK_GEMINI      = "fallback_gemini"   # graceful fallback

CONTEXT_TOKEN_MAP = {
    "email"         : "<email>",
    "report"        : "<report>",
    "meeting_notes" : "<meeting>",
    "presentation"  : "<slides>",
    "proposal"      : "<proposal>",
    "general"       : "<formal>",
    "legal"         : "<legal>",
    "academic"      : "<academic>",
}
LANG_TOKENS = {
    "English"   : "<en>", "Hindi"     : "<hi>", "Telugu"    : "<te>",
    "Tamil"     : "<ta>", "Kannada"   : "<kn>", "Malayalam" : "<ml>",
    "Marathi"   : "<mr>", "Bengali"   : "<bn>", "Gujarati"  : "<gu>",
    "Spanish"   : "<es>", "French"    : "<fr>", "German"    : "<de>",
    "Japanese"  : "<ja>",
}
MBART_LANG_CODES = {
    "English"   : "en_XX", "Hindi"     : "hi_IN", "Telugu"    : "te_IN",
    "Tamil"     : "ta_IN", "Kannada"   : "kn_IN", "Malayalam" : "ml_IN",
    "Marathi"   : "mr_IN", "Bengali"   : "bn_IN", "Gujarati"  : "gu_IN",
    "Spanish"   : "es_XX", "French"    : "fr_XX", "German"    : "de_DE",
    "Japanese"  : "ja_XX",
}
INDIC_CODES = {
    "Hindi"     : "hin_Deva", "Telugu"    : "tel_Telu", "Tamil"     : "tam_Taml",
    "Kannada"   : "kan_Knda", "Malayalam" : "mal_Mlym", "Marathi"   : "mar_Deva",
    "Bengali"   : "ben_Beng", "Gujarati"  : "guj_Gujr", "English"   : "eng_Latn",
}

VRAM_LIMIT_GB = 20.0   # leave 4 GB headroom on A10G

# ──────────────────────────────────────────────────────────────
# Model registry — edit paths after training
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict] = {
    "asr": {
        "path"    : os.getenv("ASR_MODEL_PATH",
                    "./saved_models/whisper_indic_asr_merged"),
        "type"    : "whisper",
        "vram_gb" : 6.0,
    },
    "formalization": {
        "path"    : os.getenv("FORM_MODEL_PATH",
                    "./saved_models/formalization_model_merged"),
        "type"    : "mt5",
        "vram_gb" : 2.5,
    },
    "translation": {
        "path"    : os.getenv("TRANS_MODEL_PATH",
                    "./saved_models/translation_model_merged"),
        "type"    : "mbart",     # or "indictrans2"
        "vram_gb" : 5.0,
    },
}

# ──────────────────────────────────────────────────────────────
# GPU memory helpers
# ──────────────────────────────────────────────────────────────

def free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 999.0
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    return free / 1e9

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ──────────────────────────────────────────────────────────────
# Loaded model state (singleton per process)
# ──────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    key         : str
    model       : Any
    tokenizer   : Any       = None
    processor   : Any       = None
    pipeline    : Any       = None   # HF pipeline object if used
    loaded_at   : float     = field(default_factory=time.time)
    last_used   : float     = field(default_factory=time.time)

_loaded: Dict[str, LoadedModel] = {}
_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────

def _load_asr() -> LoadedModel:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
    path = MODEL_REGISTRY["asr"]["path"]
    logger.info(f"Loading ASR model from {path}")
    processor = WhisperProcessor.from_pretrained(path)
    model     = WhisperForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto"
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model       = model,
        tokenizer   = processor.tokenizer,
        feature_extractor = processor.feature_extractor,
        chunk_length_s    = 30,
        device      = 0 if torch.cuda.is_available() else -1,
        torch_dtype = torch.float16,
    )
    return LoadedModel(key="asr", model=model, processor=processor, pipeline=pipe)

def _load_formalization() -> LoadedModel:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    path = MODEL_REGISTRY["formalization"]["path"]
    logger.info(f"Loading formalization model from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return LoadedModel(key="formalization", model=model, tokenizer=tokenizer)

def _load_translation() -> LoadedModel:
    m_type = MODEL_REGISTRY["translation"]["type"]
    path   = MODEL_REGISTRY["translation"]["path"]
    logger.info(f"Loading translation model ({m_type}) from {path}")
    if m_type == "indictrans2":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model     = AutoModelForSeq2SeqLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True
        )
    else:  # mbart
        from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
        tokenizer = MBart50TokenizerFast.from_pretrained(path)
        model     = MBartForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return LoadedModel(key="translation", model=model, tokenizer=tokenizer)

_LOADERS = {
    "asr"           : _load_asr,
    "formalization" : _load_formalization,
    "translation"   : _load_translation,
}

# ──────────────────────────────────────────────────────────────
# Smart load / evict
# ──────────────────────────────────────────────────────────────

def _evict_lru_if_needed(needed_gb: float):
    """Evict least-recently-used model if VRAM is tight."""
    if free_vram_gb() >= needed_gb:
        return
    with _lock:
        if not _loaded:
            return
        lru_key = min(_loaded, key=lambda k: _loaded[k].last_used)
        logger.info(f"Evicting '{lru_key}' to free VRAM …")
        lm = _loaded.pop(lru_key)
        del lm.model
        if lm.tokenizer:   del lm.tokenizer
        if lm.processor:   del lm.processor
        if lm.pipeline:    del lm.pipeline
        clear_vram()
        logger.info(f"After eviction: {free_vram_gb():.1f} GB free")

def get_model(key: str) -> LoadedModel:
    """Return (possibly cached) loaded model. Thread-safe."""
    with _lock:
        if key in _loaded:
            _loaded[key].last_used = time.time()
            return _loaded[key]

    needed = MODEL_REGISTRY[key]["vram_gb"]
    _evict_lru_if_needed(needed)

    # Check model path exists
    model_path = MODEL_REGISTRY[key]["path"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model '{key}' not found at '{model_path}'. "
            f"Run the training notebook first."
        )

    lm = _LOADERS[key]()
    with _lock:
        _loaded[key] = lm
    logger.info(f"'{key}' loaded | VRAM free: {free_vram_gb():.1f} GB")
    return lm

# ──────────────────────────────────────────────────────────────
# Routing logic
# ──────────────────────────────────────────────────────────────

ENGLISH_LANGS = {"english", "en"}
NATIVE_LANGS  = {
    "hindi", "telugu", "tamil", "kannada", "malayalam",
    "marathi", "bengali", "gujarati", "odia", "assamese",
    "punjabi", "hinglish",
    "spanish", "french", "german", "japanese", "mandarin",
}

def route_request(
    input_text  : str,
    language    : str,
    output_mode : str,   # "same_language" | "formal_english"
    context     : str,
    formality   : int,
) -> Pipeline:
    lang_lower = language.lower()
    if output_mode == "same_language":
        return Pipeline.FORMALIZE_SAME_LANG
    elif output_mode == "formal_english":
        if lang_lower in ENGLISH_LANGS:
            return Pipeline.FORMALIZE_SAME_LANG
        return Pipeline.TRANSLATE_TO_ENGLISH
    return Pipeline.FORMALIZE_SAME_LANG

# ──────────────────────────────────────────────────────────────
# Inference functions
# ──────────────────────────────────────────────────────────────

def _run_formalization(
    text: str,
    language: str,
    context: str,
    formality: int,
    max_new_tokens: int = 256,
) -> str:
    lm        = get_model("formalization")
    tokenizer = lm.tokenizer
    model     = lm.model

    # Build prompt with control tokens
    ctx_tok  = CONTEXT_TOKEN_MAP.get(context, "<formal>")
    f_round  = max(10, min(100, int(round(formality / 10) * 10)))
    f_tok    = f"<f{f_round}>"
    lang_tok = LANG_TOKENS.get(language.title(), "<en>")
    prompt   = f"{ctx_tok} {f_tok} {lang_tok} formalize: {text}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams      = 4,
            early_stopping = True,
            no_repeat_ngram_size = 3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def _run_translation(
    text    : str,
    src_lang: str,
    tgt_lang: str = "English",
    max_new_tokens: int = 256,
) -> str:
    lm        = get_model("translation")
    tokenizer = lm.tokenizer
    model     = lm.model
    m_type    = MODEL_REGISTRY["translation"]["type"]

    if m_type == "indictrans2":
        src_code = INDIC_CODES.get(src_lang, "eng_Latn")
        tgt_code = INDIC_CODES.get(tgt_lang, "eng_Latn")
        prompt   = f">>{tgt_code}<< {text}"
        inputs   = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        forced_bos = None
    else:  # mBART
        tokenizer.src_lang = MBART_LANG_CODES.get(src_lang, "en_XX")
        inputs   = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        forced_bos = tokenizer.lang_code_to_id[
            MBART_LANG_CODES.get(tgt_lang, "en_XX")
        ]

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens"       : max_new_tokens,
        "num_beams"            : 4,
        "early_stopping"       : True,
        "no_repeat_ngram_size" : 3,
    }
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    return lm.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def _run_asr(audio_path: str, language: str) -> str:
    lm   = get_model("asr")
    pipe = lm.pipeline
    lang_map = {
        "english": "en",  "hindi": "hi",   "telugu": "te",
        "tamil": "ta",    "kannada": "kn",  "malayalam": "ml",
        "marathi": "mr",  "bengali": "bn",  "gujarati": "gu",
        "spanish": "es",  "french": "fr",   "german": "de",
        "japanese": "ja",
    }
    lang_code = lang_map.get(language.lower(), "en")
    result = pipe(
        audio_path,
        generate_kwargs={"language": lang_code, "task": "transcribe"},
        return_timestamps=False,
    )
    return result["text"].strip()

def _run_explain(original: str, formalized: str) -> str:
    """
    Rule-based diff explanation when no dedicated model is trained.
    Upgrade: fine-tune mt5 on explanation pairs for better quality.
    """
    lines = []
    orig_words = set(original.lower().split())
    form_words = set(formalized.lower().split())
    removed    = orig_words - form_words
    added      = form_words - orig_words

    slang_map = {
        "gonna":"going to", "wanna":"want to", "kinda":"somewhat",
        "bro":"colleague", "ra":"please attend", "bhai":"",
        "yaar":"", "da":"", "anna":"", "tmrw":"tomorrow",
        "u":"you", "r":"are", "ur":"your", "ok":"acceptable",
    }
    for slang, formal in slang_map.items():
        if slang in removed and formal.split() and formal.split()[0] in added:
            lines.append(f"- '{slang}' → '{formal}' (informal colloquial replaced with formal equivalent)")

    if len(formalized.split(".")) > len(original.split(".")):
        lines.append("- Sentence structure expanded for clarity and professional tone")
    if len(formalized) > len(original) * 1.3:
        lines.append("- Sentence elaborated to meet formal communication standards")
    if not lines:
        lines.append("- Informal vocabulary and contractions replaced with formal alternatives")
        lines.append("- Sentence structure reorganised for professional context")
    return "\n".join(lines[:5])

# ──────────────────────────────────────────────────────────────
# Streaming helpers (mimic Gemini stream interface)
# ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 30) -> Generator[str, None, None]:
    """Yield text in word-chunks to simulate streaming."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size]) + (
            " " if i + chunk_size < len(words) else ""
        )

# ──────────────────────────────────────────────────────────────
# Public API  (drop-in replacement for formalize_with_gemini)
# ──────────────────────────────────────────────────────────────

def formalize_with_local_models(
    text            : str,
    language        : str,
    output_mode     : str,
    formality_level : str  = "professional",
    custom_formality: Optional[int] = None,
    context         : str  = "general",
    stream          : bool = False,
) -> Any:
    """
    Drop-in replacement for formalize_with_gemini().
    Returns either a string or a generator (when stream=True).
    """
    # Resolve formality score
    formality_presets = {
        "casual": 40, "professional": 75, "academic": 95,
    }
    formality_score = (
        custom_formality
        if custom_formality is not None
        else formality_presets.get(formality_level, 75)
    )

    pipeline = route_request(text, language, output_mode, context, formality_score)
    logger.info(f"Routed to pipeline: {pipeline.value} "
                f"| lang={language} | ctx={context} | f={formality_score}")

    try:
        if pipeline == Pipeline.FORMALIZE_SAME_LANG:
            result = _run_formalization(text, language, context, formality_score)

        elif pipeline == Pipeline.TRANSLATE_TO_ENGLISH:
            # Step 1: translate to English
            translated = _run_translation(text, language.title(), "English")
            # Step 2: formalize the English translation
            result = _run_formalization(
                translated, "English", context, formality_score
            )

        else:
            result = f"[Pipeline {pipeline.value} not yet implemented]"

    except FileNotFoundError as e:
        # Models not trained yet — return helpful error
        logger.warning(f"Model not found: {e}")
        result = (
            f"[LOCAL MODEL NOT FOUND]\n"
            f"Please run the training notebook for this model first.\n"
            f"Error: {e}"
        )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        result = f"[INFERENCE ERROR] {e}"

    # Clean output
    result = result.replace("**", "").strip()
    result = re.sub(r"^\s*\*\s+", "- ", result, flags=re.MULTILINE)

    if stream:
        return _chunk_text(result)
    return result


def transcribe_audio_local(audio_path: str, language: str) -> str:
    """Route audio through local Whisper ASR."""
    try:
        return _run_asr(audio_path, language)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ASR model not found. Run notebook 01 first. {e}"
        ) from e


def explain_changes_local(original: str, formalized: str) -> str:
    """Generate explanation of formalization changes."""
    return _run_explain(original, formalized)


# ──────────────────────────────────────────────────────────────
# Health check + status
# ──────────────────────────────────────────────────────────────

def router_status() -> Dict:
    """Return current router status for health endpoint."""
    status = {
        "loaded_models" : list(_loaded.keys()),
        "free_vram_gb"  : round(free_vram_gb(), 2),
        "device"        : "cuda" if torch.cuda.is_available() else "cpu",
    }
    for key, reg in MODEL_REGISTRY.items():
        status[f"{key}_path_exists"] = os.path.isdir(reg["path"])
    return status


# ──────────────────────────────────────────────────────────────
# CLI test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Model Router — Quick Test ===")
    print("Status:", json.dumps(router_status(), indent=2))

    test_cases = [
        dict(text="hey bro can u come tmrw for the meeting",
             language="English", output_mode="same_language",
             context="email", custom_formality=80),
        dict(text="bro tomorrow meeting ki ra late avvaku",
             language="Telugu", output_mode="formal_english",
             context="email", custom_formality=80),
        dict(text="bhai report kal tak bhej do",
             language="Hindi", output_mode="formal_english",
             context="report", custom_formality=85),
    ]

    for tc in test_cases:
        print(f"\n{'─'*60}")
        print(f"INPUT  : {tc['text']}")
        print(f"LANG   : {tc['language']} | MODE: {tc['output_mode']}")
        try:
            out = formalize_with_local_models(**tc)
            print(f"OUTPUT : {out}")
        except Exception as e:
            print(f"ERROR  : {e}")