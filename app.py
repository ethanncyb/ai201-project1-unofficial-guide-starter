"""Gradio web interface for The Unofficial Guide.

Run:
    python app.py
Then open http://localhost:7860 in a browser.

Layout:
- Left sidebar: sliders for `top_k`, `temperature`, `max_tokens` — passed
  through to `generate.answer()` per-query so a viewer can see how each knob
  changes retrieval / generation behavior.
- Main panel: question textbox → Ask button → two output boxes:
  - `Answer` shows the LLM prose with inline [N] citations and the
    programmatic `References:` block.
  - `Debug — Retrieved chunks` is a verbose audit panel: for each hit it
    shows rank, citation number, source, parent_title, cosine distance, and
    a text preview, so the viewer can verify the answer is grounded in the
    cited chunks (not invented).
"""

from __future__ import annotations

import gradio as gr

from embed import DEFAULT_TOP_K
from generate import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, answer

PREVIEW_CHARS = 300


def _format_debug(hits: list[dict], sources: list[dict], top_k: int, temperature: float, max_tokens: int) -> str:
    # Build a source → citation number lookup so each chunk shows which [N]
    # it contributes to (chunks from the same source share a number).
    number_for = {r["source"]: r["number"] for r in sources}

    header = (
        f"params: top_k={top_k}  temperature={temperature:.2f}  max_tokens={max_tokens}\n"
        f"retrieved {len(hits)} chunk(s)\n"
        f"{'-' * 72}"
    )
    if not hits:
        return f"{header}\n(no hits — collection empty or query off-topic)"

    blocks: list[str] = [header]
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        cite = number_for.get(m["source"], "—")
        title = m.get("parent_title") or ""
        title_part = f' — "{title}"' if title else ""
        blocks.append(
            f"\n#{i}  [cite {cite}]  {m['source']}#{m['chunk_index']}  "
            f"distance={h['distance']:.3f}{title_part}"
        )
        text = h["text"]
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
        for line in preview.splitlines():
            blocks.append(f"   {line}")
    return "\n".join(blocks)


def handle_query(
    question: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    result = answer(
        question,
        k=int(top_k),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    debug = _format_debug(result["hits"], result["sources"], int(top_k), float(temperature), int(max_tokens))
    return result["answer"], debug


with gr.Blocks(title="The Unofficial Guide — USC off-campus housing") as demo:
    with gr.Sidebar():
        gr.Markdown("### Parameters")
        top_k_slider = gr.Slider(
            label="top_k (chunks retrieved)",
            minimum=1,
            maximum=10,
            value=DEFAULT_TOP_K,
            step=1,
        )
        temperature_slider = gr.Slider(
            label="temperature (Groq sampling)",
            minimum=0.0,
            maximum=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
        )
        max_tokens_slider = gr.Slider(
            label="max_tokens (response cap)",
            minimum=128,
            maximum=2048,
            value=DEFAULT_MAX_TOKENS,
            step=64,
        )
        gr.Markdown(
            "_Lower temperature → more faithful, less creative. "
            "Higher top_k → more context but more noise._"
        )

    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask about off-campus housing and daily living near USC / Downtown LA — "
        "buildings, shuttles, neighborhood safety, nearby supermarkets, transit, amenities. "
        "Answers are grounded in student-generated reviews, Reddit threads, and the r/USC wiki, "
        "with paper-style `[N]` citations and a references list. "
        "If the documents don't cover your question, the system will say so rather than guess."
    )
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. Which streets near USC are considered safest at night?",
        lines=2,
    )
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=12)
    debug_box = gr.Textbox(label="Debug — Retrieved chunks", lines=14)

    inputs = [question, top_k_slider, temperature_slider, max_tokens_slider]
    outputs = [answer_box, debug_box]
    ask_btn.click(handle_query, inputs=inputs, outputs=outputs)
    question.submit(handle_query, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
