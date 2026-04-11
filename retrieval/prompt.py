"""
Prompt engineering for the clinical RAG assistant.

Design principles:
- Gemma 4 answers ONLY from the provided documents (grounded generation).
- Structured JSON output makes the FastAPI layer and frontend trivial to parse.
- NOT_FOUND sentinel lets the pipeline detect when the guardrail fires.
- Explicit citation requirement keeps answers traceable to source documents.
"""

import json


def build_prompt(scenario: str, chunks: list[dict]) -> str:
    """
    Build the prompt sent to Gemma 4.

    Args:
        scenario: The trainee's clinical question or scenario.
        chunks:   Retrieved chunks from retrieve(), each with
                  text / source / section / page keys.

    Returns:
        A fully formatted prompt string requesting structured JSON output.
    """
    context = "\n\n".join([
        f"[Source: {c['source']}, Section: {c['section']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    ])

    # List unique sources so the model knows what to cite
    sources = list({c["source"] for c in chunks})

    return f"""You are a clinical guidance assistant for psychology trainees.
Answer ONLY based on the documents provided below.
Never invent information. If the answer cannot be found in the documents, respond with exactly: NOT_FOUND

Available sources: {json.dumps(sources)}

Return your response as valid JSON with this exact structure:
{{
  "answer": "Your detailed guidance here, referring to the source by name.",
  "sources_used": ["Exact source title 1", "Exact source title 2"],
  "confidence": "high | medium | low",
  "found_in_documents": true
}}

CLINICAL DOCUMENTS:
{context}

TRAINEE SCENARIO:
{scenario}

JSON RESPONSE:"""
