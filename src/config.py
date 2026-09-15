"""Central configuration for the PRISM code-retrieval pipeline.

Every tunable lives here so that experiments are reproducible: to reproduce a
row in `experiments.md` you should only ever need this file plus a git SHA.

CONVENTION
----------
Values marked ``# PLACEHOLDER`` are deliberate guesses that nobody has
validated yet. Replace them with something measured, then log the before/after
numbers in ``experiments.md``.

OWNERSHIP
---------
The file is split into per-workstream sections so four people can edit it in
parallel without fighting over the same lines. Stay inside your own section.
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# SHARED - paths and dataset identity (change only with team agreement)
# =============================================================================

#: Repository root, resolved from this file's location (``src/config.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Scratch space for embeddings, indexes and other large build artifacts.
#: Git-ignored. Override with PRISM_DATA_DIR to point at a fast local disk.
DATA_DIR: Path = Path(os.environ.get("PRISM_DATA_DIR", PROJECT_ROOT / "data"))

#: Where MTEB writes its own result tree.
RESULTS_DIR: Path = Path(os.environ.get("PRISM_RESULTS_DIR", PROJECT_ROOT / "results"))

#: The single deliverable file the submission is scored from.
RESULTS_JSON: Path = PROJECT_ROOT / "appsretrieval_results.json"

#: Hugging Face cache. Pointing this inside the repo keeps the Docker layer
#: self-contained; leave unset to use the machine-wide ~/.cache/huggingface.
HF_CACHE_DIR: Path | None = (
    Path(os.environ["PRISM_HF_CACHE"]) if "PRISM_HF_CACHE" in os.environ else None
)

#: MTEB task under evaluation. Do not change - this is the graded task.
MTEB_TASK_NAME: str = "AppsRetrieval"

#: Underlying dataset, for reference when loading the corpus directly.
DATASET_ID: str = "CoIR-Retrieval/apps"

#: Metrics we report. NDCG@10 is the headline number.
PRIMARY_METRIC: str = "ndcg_at_10"
SECONDARY_METRIC: str = "mrr_at_10"

#: Fixed seed for anything stochastic, so reruns are comparable.
RANDOM_SEED: int = 42

#: CPU-only build. Nothing in this project may request a GPU.
DEVICE: str = "cpu"

#: Threads for torch / faiss. None = let the library decide.
#: PLACEHOLDER - set to physical core count if encode throughput is the bottleneck.
NUM_THREADS: int | None = None


# =============================================================================
# WORKSTREAM 1 - query preprocessing              (owner: query)
# =============================================================================

#: Toggle so the rest of the team can A/B your stage without editing code.
ENABLE_QUERY_PREPROCESSING: bool = False

#: Labels your classifier may emit. Consumers must tolerate an unknown label.
#: PLACEHOLDER - replace with the taxonomy you actually settle on.
QUERY_CATEGORIES: tuple[str, ...] = (
    "algorithm",
    "data_structure",
    "string_manipulation",
    "math",
    "io_parsing",
    "other",
)

#: Drop queries shorter than this after cleaning (characters).
#: PLACEHOLDER
MIN_QUERY_CHARS: int = 3

#: Hard cap on query length handed to the encoder (characters, pre-tokenization).
#: PLACEHOLDER
MAX_QUERY_CHARS: int = 2_000


# =============================================================================
# WORKSTREAM 2 - snippet preprocessing + indexing (owner: corpus)
# =============================================================================

ENABLE_SNIPPET_PREPROCESSING: bool = False

#: Truncate snippets to this many characters before encoding, as a safety net
#: ONLY - DENSE_MODEL_NAME (see workstream 3) was chosen specifically because
#: its 8k-token context makes truncation rare rather than routine. Raised from
#: the original 4,000 now that the model can actually see this much code;
#: still measure the truncation rate on the train split before trusting it.
MAX_SNIPPET_CHARS: int = 20_000

#: Strip comments/docstrings from code before encoding.
#: PLACEHOLDER - may help (less noise) or hurt (comments carry NL signal). Test it.
STRIP_COMMENTS: bool = False

#: Split long snippets into overlapping windows and pool the results. Kept
#: disabled: with an 8k-token bi-encoder + MAX_SNIPPET_CHARS above, chunking
#: is a rarely-needed escape hatch, not the default path. Flip it on only if
#: the truncation-rate measurement above says otherwise.
ENABLE_CHUNKING: bool = False
CHUNK_SIZE_CHARS: int = 1_500      # PLACEHOLDER - only matters if chunking is enabled
CHUNK_OVERLAP_CHARS: int = 200     # PLACEHOLDER - only matters if chunking is enabled

#: FAISS index factory string. FINALIZED as "Flat" (exact brute-force search).
#: At ~8.77k corpus docs, exact search over normalized vectors is a sub-5ms
#: numpy/FAISS matmul - there is no latency problem to solve, so an ANN index
#: (IVF/HNSW) would only trade accuracy away for a speed-up nobody needs. Only
#: revisit this if TOP_K_DENSE search is independently measured as a
#: bottleneck, and re-measure NDCG afterwards because ANN is lossy.
FAISS_INDEX_FACTORY: str = "Flat"

#: Cosine similarity via inner product requires L2-normalized vectors.
NORMALIZE_EMBEDDINGS: bool = True


# =============================================================================
# WORKSTREAM 3 - retrieval, fusion, reranking     (owner: retrieval)
# =============================================================================

# ---- Bi-encoder (dense) -----------------------------------------------------

#: FINALIZED (tech-stack decision, see TECH_STACK.md item 1).
#: jina-embeddings-v2-base-code: 161M params, code/NL bilingual retrieval
#: checkpoint (trained on CodeSearchNet-style text-to-code pairs across 30
#: languages, so it starts far closer to AppsRetrieval than a general-prose
#: model), and - the deciding factor for this dataset - an 8,192-token context
#: via ALiBi. That directly answers the "long APPS snippets get silently
#: truncated at 512 tokens" problem in the dataset brief, instead of papering
#: over it with chunking. It is still base-sized, so it fits the CPU-only
#: constraint for a corpus this small (~12.5k texts encoded once, then cached
#: - see ENABLE_EMBEDDING_CACHE).
#:
#: Fallback if the judging sandbox will not allow trust_remote_code (below):
#: "sentence-transformers/all-MiniLM-L6-v2" (the original placeholder, and the
#: model the day-one baseline number was measured with) - swap this one
#: constant back and MAX_SNIPPET_CHARS truncation starts mattering again.
DENSE_MODEL_NAME: str = "jinaai/jina-embeddings-v2-base-code"

#: jina-embeddings-v2-base-code ships custom modeling code (ALiBi attention)
#: rather than a stock HF architecture, so sentence-transformers needs this to
#: load it at all. Not needed (and ignored) for stock checkpoints like MiniLM.
DENSE_MODEL_TRUST_REMOTE_CODE: bool = True

#: Encoder batch size. Lower it if the CPU box starts swapping.
#: PLACEHOLDER
BATCH_SIZE: int = 32

#: FINALIZED at 1,024 tokens - a deliberate midpoint, not the model's max.
#: The checkpoint extrapolates to 8,192 tokens, but self-attention cost grows
#: roughly with the square of sequence length, and encoding the full corpus at
#: 8,192 tokens on CPU is not a one-time cost worth paying when most APPS
#: solutions are far shorter. 1,024 covers the long tail without that blowup;
#: re-measure the truncation rate on the train split and raise it only if
#: that rate is still uncomfortable.
MAX_SEQ_LENGTH: int | None = 1_024

#: jina-embeddings-v2-base-code is symmetric (no E5/BGE/GTE-style asymmetric
#: query/document instruction prefix) - confirmed against the model card, not
#: assumed. Leave both empty for this checkpoint; if the team swaps to an
#: asymmetric model later (BGE/E5/GTE), this is the first thing to fix, and
#: silently, since a missing prefix degrades quality without erroring.
QUERY_PROMPT_PREFIX: str = ""
DOCUMENT_PROMPT_PREFIX: str = ""

# ---- BM25 (sparse) ----------------------------------------------------------

ENABLE_BM25: bool = False

#: FINALIZED as the starting point: rank_bm25's Okapi defaults (also the
#: values from the original paper). Not tuned to this corpus yet - do that on
#: the train split's 5k queries, never on test. The bigger lever for code is
#: the TOKENIZER (src.corpus.index.tokenize_code, owner: corpus): split
#: snake_case/camelCase into subtokens (and keep the compound too) so "binary
#: search" matches `binary_search`, lowercase everything, and do NOT strip
#: underscores - they're part of the identifier BM25 exists to catch.
BM25_K1: float = 1.5
BM25_B: float = 0.75

# ---- Candidate depths -------------------------------------------------------
# Pipeline shape:  retrieve TOP_K_DENSE + TOP_K_BM25 -> fuse -> TOP_K_FUSED
#                  -> rerank -> TOP_K_FINAL
# TOP_K_FINAL must stay >= 10 or NDCG@10 is truncated and the score is invalid.

TOP_K_DENSE: int = 100   # PLACEHOLDER
TOP_K_BM25: int = 100    # PLACEHOLDER
TOP_K_FUSED: int = 100   # PLACEHOLDER - how many survive fusion into the reranker
TOP_K_FINAL: int = 10    # returned to MTEB; >= 10 required for NDCG@10

# ---- Fusion -----------------------------------------------------------------

#: "rrf" (rank-based, scale-free) or "weighted_sum" (needs score normalization).
#: PLACEHOLDER
FUSION_STRATEGY: str = "rrf"

#: RRF smoothing constant. 60 is the value from the original RRF paper.
RRF_K: int = 60  # PLACEHOLDER

#: Weights for (dense, bm25) when FUSION_STRATEGY == "weighted_sum".
#: PLACEHOLDER - must sum to 1.0 by convention.
FUSION_WEIGHTS: tuple[float, float] = (0.5, 0.5)

# ---- Reranking --------------------------------------------------------------

ENABLE_RERANK: bool = False

#: FINALIZED: cross-encoder/ms-marco-MiniLM-L-6-v2. There is no established,
#: widely-used code-specific cross-encoder to reach for here, so this stays a
#: general passage-relevance reranker - the "does this text answer this
#: query" signal it was trained on transfers reasonably to "does this snippet
#: answer this problem statement." What matters more is that it is small
#: (6-layer MiniLM, ~22M params): unlike the bi-encoder, this model is on the
#: PER-QUERY hot path (nothing here is cached), so it is the tightest CPU
#: latency budget in the whole pipeline. If the team has latency budget left
#: after measuring, cross-encoder/ms-marco-MiniLM-L-12-v2 is the next rung up
#: in quality, at roughly double the cost.
RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: FINALIZED starting point: 50. Rough CPU budget: 3.77k test queries x 50
#: candidates = ~188k (query, snippet) forward passes through a 6-layer
#: MiniLM. Measure actual wall-clock on the first full run (run_eval.py prints
#: it) and back this off toward 25 if the full evaluation doesn't comfortably
#: finish - a config that cannot finish scores nothing, per experiments.md.
RERANK_TOP_N: int = 50

RERANK_BATCH_SIZE: int = 32  # PLACEHOLDER - raise if CPU throughput measurement supports it


# =============================================================================
# WORKSTREAM 4 - evaluation + caching             (owner: eval)
# =============================================================================

#: Persist embeddings keyed by content hash so reruns skip re-encoding.
#: See src/versioning/cache.py.
ENABLE_EMBEDDING_CACHE: bool = True

CACHE_DIR: Path = DATA_DIR / "embedding_cache"

#: Bump to invalidate every cached embedding at once (e.g. after changing
#: preprocessing in a way the content hash cannot see). Bumped v1 -> v2 here
#: because DENSE_MODEL_NAME changed (MiniLM -> jina-code); make_cache_key
#: already folds in the model name too, so this bump is belt-and-suspenders,
#: not load-bearing - but it's the documented, deliberate way to say "nothing
#: written before this point should be trusted," which a silent model swap
#: alone doesn't communicate to someone reading the cache directory later.
CACHE_VERSION: str = "v2"

#: Cap evaluation to N queries for a fast smoke run. None = full evaluation.
#: MUST be None for any number recorded in experiments.md or submitted.
SMOKE_TEST_QUERY_LIMIT: int | None = None

#: Release tag for the final submission.
RELEASE_TAG: str = "PRISM_GENAI_HACKATHON_Y2026"


# =============================================================================
# BONUS / DEMO ONLY - AST inspection                (owner: whoever demos)
# =============================================================================
# Tech-stack decision: Python's stdlib `ast` module, not tree-sitter or any
# other parser. The corpus is 100% Python (APPS solutions), so there is no
# multi-language case to justify tree-sitter's extra dependency and grammar
# management. This flag must never be read by anything under src/pipeline or
# src/retrieval - AST inspection is presentation-layer only (e.g. showing a
# retrieved snippet's function signature + docstring instead of raw text, or
# an AST-diff view for the versioning demo) and must not affect the score.

ENABLE_AST_DEMO: bool = False


# =============================================================================
# OPTIONAL - LLM query expansion                    (owner: query, stretch)
# =============================================================================
# Tech-stack decision: SKIP for the scored submission. Expanding 3.77k test
# queries through any LLM - local or API - adds a per-query latency and
# failure-mode risk (a hung or rate-limited call mid-evaluation) for an
# unmeasured NDCG upside, on a CPU-only budget that is already spent on the
# reranker. If pursued at all, treat it as a demo-narrative feature, not a
# scored one: run it once, offline, on a handful of illustrative queries, with
# a model that needs neither a GPU nor network access at judging time (e.g.
# "google/flan-t5-small", ~80M params) - and cache the expansions through the
# same embedding cache so it is a one-time cost, never a per-eval-run one.

ENABLE_QUERY_EXPANSION: bool = False
QUERY_EXPANSION_MODEL: str = "google/flan-t5-small"  # demo-only if ever enabled


def ensure_dirs() -> None:
    """Create the scratch directories this config points at.

    Safe to call repeatedly; called by ``scripts/run_eval.py`` on startup.
    """
    for directory in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, object]:
    """Return the active configuration as a flat, JSON-serializable dict.

    Dumped alongside every evaluation run so an experiments.md row can always
    be traced back to the exact settings that produced it.
    """
    return {
        "task": MTEB_TASK_NAME,
        "dense_model": DENSE_MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME if ENABLE_RERANK else None,
        "batch_size": BATCH_SIZE,
        "query_preprocessing": ENABLE_QUERY_PREPROCESSING,
        "snippet_preprocessing": ENABLE_SNIPPET_PREPROCESSING,
        "bm25": ENABLE_BM25,
        "rerank": ENABLE_RERANK,
        "fusion_strategy": FUSION_STRATEGY if ENABLE_BM25 else None,
        "top_k_dense": TOP_K_DENSE,
        "top_k_bm25": TOP_K_BM25 if ENABLE_BM25 else None,
        "top_k_fused": TOP_K_FUSED,
        "top_k_final": TOP_K_FINAL,
        "rerank_top_n": RERANK_TOP_N if ENABLE_RERANK else None,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dense_model_trust_remote_code": DENSE_MODEL_TRUST_REMOTE_CODE,
        "faiss_index_factory": FAISS_INDEX_FACTORY,
        "cache_version": CACHE_VERSION if ENABLE_EMBEDDING_CACHE else None,
        "ast_demo": ENABLE_AST_DEMO,
        "query_expansion": ENABLE_QUERY_EXPANSION,
        "seed": RANDOM_SEED,
        "smoke_limit": SMOKE_TEST_QUERY_LIMIT,
    }
