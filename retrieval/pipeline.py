"""
Full RAG pipeline: retrieve -> prompt -> Gemma 4 E2B (local) -> structured response.

Architecture:
  - Retrieval   : Hybrid BM25 + ChromaDB dense + cross-encoder re-ranking
  - Generation  : google/gemma-4-E2B-it loaded locally via transformers (4-bit NF4)

First run downloads model weights to ~/.cache/huggingface/
Requires HUGGINGFACE_API_TOKEN in .env to download the gated model.

Usage:
    python pipeline.py "What are the key symptoms of major depression?"
    python pipeline.py "my patient hasn't eaten in 3 days" --debug
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from retrieve import retrieve
from prompt import build_prompt

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_ID        = "google/gemma-4-E2B-it"
# The cross-encoder reranker is now the quality gate — the threshold here
# only catches genuinely zero-relevance queries before loading the LLM.
RELEVANCE_THRESHOLD = -12.0  # cross-encoder logit floor — only blocks truly random noise.
                             # ms-marco scores clinical docs lower than web QA by design;
                             # the LLM's found_in_documents is the real quality gate.
NO_MATCH_ANSWER = (
    "No relevant guidance found in the available documents. "
    "Please consult your supervisor."
)

_model     = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig

    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "HUGGINGFACE_API_TOKEN is not set. "
            "Add it to your .env file — required to download the gated model."
        )

    print(f"Loading {MODEL_ID} onto GPU (4-bit quantization)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    _model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=token,
    )
    _processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)

    print("Model ready.\n")
    return _model, _processor


def _generate(prompt: str) -> str:
    import torch

    model, processor = _load_model()
    device = next(p.device for p in model.parameters())

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def _parse_answer(raw: str) -> dict:
    text = raw.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {
            "answer":             text,
            "sources_used":       [],
            "confidence":         "low",
            "found_in_documents": True,
        }

    if "answer" in parsed and isinstance(parsed["answer"], str):
        parsed["answer"] = parsed["answer"].replace("NOT_FOUND", "").strip(" .,\n")

    return parsed


def run_rag_pipeline(scenario: str, k: int = 4, debug: bool = False) -> dict:
    """
    Run the full RAG pipeline for a clinical scenario.

    Steps:
      1. Hybrid retrieval  — BM25 + dense + cross-encoder reranker
      2. Relevance gate    — drop truly irrelevant results
      3. Build prompt      — grounded context window
      4. LLM generation    — Gemma 4 E2B
      5. Parse JSON output — structured answer

    Args:
        scenario: The trainee's question or clinical scenario.
        k:        Number of chunks to retrieve (default 4).
        debug:    Print retrieved chunks and raw LLM output.
    """
    start = time.time()

    # Step 1 — hybrid retrieval (BM25 + dense + cross-encoder reranker)
    chunks = retrieve(scenario, k=k)

    if debug:
        print("\n" + "─" * 60)
        print("DEBUG — Chunks after hybrid retrieval + reranking:")
        for i, c in enumerate(chunks):
            print(f"  [{i}] rerank_score={c['relevance_score']}  {c['source']} p.{c['page']}")
            print(f"       {c['text'][:200].strip()!r}")
        print("─" * 60 + "\n")

    # Step 2 — relevance gate (reranker scores are logits; below -5 means no match at all)
    if not chunks or chunks[0]["relevance_score"] < RELEVANCE_THRESHOLD:
        return {
            "answer":             NO_MATCH_ANSWER,
            "sources":            [],
            "sources_used":       [],
            "confidence":         "low",
            "found_in_documents": False,
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    # Step 3 — build grounded prompt
    prompt = build_prompt(scenario, chunks)

    # Step 4 — generate with local Gemma 4 E2B
    raw_answer = _generate(prompt)

    if debug:
        print("─" * 60)
        print("DEBUG — Raw LLM output (before JSON parsing):")
        print(raw_answer)
        print("─" * 60 + "\n")

    # Step 5 — handle NOT_FOUND sentinel
    if raw_answer.strip() == "NOT_FOUND":
        return {
            "answer":             NO_MATCH_ANSWER,
            "sources":            [],
            "sources_used":       [],
            "confidence":         "low",
            "found_in_documents": False,
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    # Step 6 — parse structured JSON output
    parsed = _parse_answer(raw_answer)

    # Step 7 — respect the LLM's own grounding judgement
    if not parsed.get("found_in_documents", True):
        return {
            "answer":             NO_MATCH_ANSWER,
            "sources":            [],
            "sources_used":       [],
            "confidence":         "low",
            "found_in_documents": False,
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    result = {
        "answer":             parsed.get("answer", raw_answer),
        "sources":            chunks,
        "sources_used":       parsed.get("sources_used", []),
        "confidence":         parsed.get("confidence", "medium"),
        "found_in_documents": parsed.get("found_in_documents", True),
        "processing_time_ms": int((time.time() - start) * 1000),
    }
    if debug:
        result["raw_llm_output"] = raw_answer
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args  = sys.argv[1:]
    debug = "--debug" in args
    args  = [a for a in args if a != "--debug"]

    scenario = args[0] if args else "What are the key symptoms of major depression?"

    print(f"Scenario: {scenario}\n{'='*60}")
    result = run_rag_pipeline(scenario, debug=debug)

    print(f"Found in documents : {result['found_in_documents']}")
    print(f"Confidence         : {result['confidence']}")
    print(f"Processing time    : {result['processing_time_ms']} ms\n")
    print(f"Answer:\n{result['answer']}\n")

    if result["sources_used"]:
        print(f"Sources cited: {result['sources_used']}")

    if result["sources"]:
        print("\nRetrieved chunks:")
        for s in result["sources"]:
            print(f"  [{s['relevance_score']}] {s['source']} - {s['section']} (p.{s['page']})")
