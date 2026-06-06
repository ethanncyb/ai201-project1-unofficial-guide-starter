"""Gradio web interface for The Unofficial Guide.

Run:
    python app.py
Then open http://localhost:7860 in a browser.

The handler calls `generate.answer()` (retrieval + grounded generation) and
routes the model's answer into the `Answer` box and the programmatic list of
unique source filenames into the `Retrieved from` box. Sources are surfaced
from chunk metadata, not parsed out of the LLM output — so attribution is
verifiable even if the model forgets to cite inline.
"""

from __future__ import annotations

import gradio as gr

from generate import answer


def handle_query(question: str) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    result = answer(question)
    sources_text = "\n".join(f"- {s}" for s in result["sources"])
    return result["answer"], sources_text


with gr.Blocks(title="The Unofficial Guide — USC off-campus housing") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask about off-campus housing near USC / Downtown LA. "
        "Answers are grounded in student-generated reviews, Reddit threads, and the r/USC wiki. "
        "If the documents don't cover your question, the system will say so rather than guess."
    )
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. Which streets near USC are considered safest at night?",
        lines=2,
    )
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=10)
    sources_box = gr.Textbox(label="Retrieved from", lines=5)

    ask_btn.click(handle_query, inputs=question, outputs=[answer_box, sources_box])
    question.submit(handle_query, inputs=question, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
