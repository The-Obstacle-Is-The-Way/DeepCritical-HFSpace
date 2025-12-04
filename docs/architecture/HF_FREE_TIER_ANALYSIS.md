# Hugging Face Free Tier Reliability Analysis (December 2025)

## Executive Summary

**Root Cause:** The recurring 500/401 errors on the Free Tier (Advanced Mode without API keys) are caused by implicit routing of large models (70B+) to unstable third-party "Inference Providers" (Novita, Hyperbolic) instead of running natively on Hugging Face's infrastructure.

**Solution:** Switch the default Free Tier model from flagship-class models (72B) to high-performance mid-sized models (7B-32B) that are hosted natively by Hugging Face's Serverless Inference API.

---

## 1. The "Inference Providers" Trap

Hugging Face offers two distinct execution paths for its Inference API:

1.  **Serverless Inference API (Native):**
    *   **Host:** Hugging Face's own infrastructure.
    *   **Reliability:** High (Direct control).
    *   **Constraints:** Limited to models that fit on standard inference hardware (typically <10GB-30GB VRAM usage).
    *   **Typical Models:** `bert-base`, `gpt2`, `Mistral-7B`, `Qwen2.5-7B`.

2.  **Inference Providers (Third-Party Marketplace):**
    *   **Host:** Partners like Novita, Hyperbolic, Together AI, Sambanova.
    *   **Reliability:** Variable. "Staging mode" authentication issues, rate limits, and service outages (500 errors) are common on the free routing layer.
    *   **Purpose:** To serve massive models (Llama-3.1-405B, Qwen2.5-72B) that are too expensive for HF to host for free.

**The Problem:**
When we request `Qwen/Qwen2.5-72B-Instruct` (or `Llama-3.1-70B`) without an API key, HF transparently routes this request to a partner (Novita/Hyperbolic).
*   **Novita Status:** Currently returning 500 Internal Server Errors.
*   **Hyperbolic Status:** Previously returned 401 Unauthorized (Staging Mode auth bug).

We are effectively relying on a "best effort" chain of third-party providers for our core application stability.

## 2. The "Golden Path" for Free Tier

To ensure stability, the Free Tier must target models that reside on the **Native** path.

**Criteria for Native Stability:**
*   **Size:** < 30B parameters (ideal: 7B - 12B).
*   **Popularity:** "Warm" models (high traffic keeps them loaded in memory).
*   **Architecture:** Standard transformers (easy for HF to serve).

**Candidate Models (Dec 2025):**

| Model | Size | Provider Risk | Native Capability |
|-------|------|---------------|-------------------|
| **Qwen/Qwen2.5-7B-Instruct** | 7B | **Low** | **Excellent** (Math: 75.5, Code: 84.8) |
| **mistralai/Mistral-Nemo-Instruct-2407** | 12B | Low | Very Good |
| **Qwen/Qwen2.5-72B-Instruct** | 72B | **High** (Novita) | Excellent (but unreliable) |
| **meta-llama/Llama-3.1-70B-Instruct** | 70B | **High** (Hyperbolic) | Excellent (but unreliable) |

## 3. Recommendation

**Immediate Fix:**
Change the default `HUGGINGFACE_MODEL` in `src/utils/config.py` from `Qwen/Qwen2.5-72B-Instruct` to **`Qwen/Qwen2.5-7B-Instruct`**.

**Why Qwen2.5-7B?**
*   **Performance:** Outperforms Llama-3.1-8B and matches GPT-3.5 levels in many benchmarks.
*   **Reliability:** Small enough to be hosted natively.
*   **Context:** 128k context window (perfect for RAG).

## 4. Future Architecture (Unified Client)

For the Unified Chat Client architecture:
1.  **Tier 0 (Free):** Hardcoded to Native Models (Qwen 7B, Mistral Nemo).
2.  **Tier 1 (BYO Key):** Allow user to select any model (70B+), assuming they provide a key that grants access to premium providers or PRO tier.

---

## 5. Known Content Quality Limitations (7B Models)

**Status**: As of December 2025, the Free Tier (Qwen 2.5 7B) produces **working multi-agent orchestration** but with notable content quality limitations.

### What Works Well
- Multi-agent coordination (Manager → Search → Hypothesis → Report)
- Clean streaming output (no garbage tokens, no raw JSON)
- Proper agent handoffs and progress tracking
- Coherent narrative structure

### Known Limitations

| Issue | Description | Severity |
|-------|-------------|----------|
| **Hallucinated Citations** | Model generates plausible-sounding but fake paper titles/authors instead of using actual search results | Medium |
| **Anatomical Confusion** | May apply male anatomy (e.g., "penile rigidity") to female health queries | High |
| **Nonsensical Medical Claims** | May generate claims like "prostate cancer risk" in context of female patients | High |
| **Duplicate Content** | Final reports sometimes contain repeated sections | Low |

### Why This Happens

7B parameter models have limited:
- **World knowledge**: Can't reliably recall specific paper titles/authors
- **Context grounding**: May ignore search results and hallucinate instead
- **Domain reasoning**: Complex medical topics exceed reasoning capacity

### User Guidance

**Free Tier is best for:**
- Understanding the research workflow
- Getting general topic overviews
- Testing the system before committing to paid tier

**For accurate medical research:**
- Use Paid Tier (GPT-5) for citation accuracy
- Always verify citations against actual databases
- Treat Free Tier output as "draft quality"

### Not a Stack Bug

These are **model capability limitations**, not bugs in the DeepBoner architecture. The orchestration, streaming, and agent coordination are working correctly.

---
*Analysis performed by Gemini CLI Agent, Dec 2, 2025*
*Content quality section added Dec 3, 2025*
