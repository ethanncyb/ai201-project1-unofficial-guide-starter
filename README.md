# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

This project focuses on off-campus housing advice for the USC and other Downtown LA campuses. It's aimed at out-of-area and international students who often can't find practical information from official channels. Things like neighborhood safety, typical commutes, local amenities, and the day-to-day realities of living nearby. I'll collect firsthand perspectives from forums, subreddits, local groups, and student reports to build a useful guide.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/USC wiki (Off-campus housing) | Reddit wiki/FAQ | https://www.reddit.com/r/USC/wiki/offcampushousing/ |
| 2 | r/USC thread: "Does anyone live in an off campus housing that they actually like" | Reddit thread | https://www.reddit.com/r/USC/comments/1pnrslg/does_anyone_live_in_an_off_campus_housing_that/ |
| 3 | r/USC thread: "Off Campus Housing" | Reddit thread | https://www.reddit.com/r/USC/comments/1tukar5/off_campus_housing/ |
| 4 | r/USC thread: "Best off-campus housing options for International/Transfer students!" | Reddit thread | https://www.reddit.com/r/USC/comments/1tpprrr/best_offcampus_housing_options_for/ |
| 5 | r/USC thread: "best off campus housing options" | Reddit thread | https://www.reddit.com/r/USC/comments/1pdhe0p/best_off_campus_housing_options/ |
| 6 | r/USC thread: "OFF CAMPUS HOUSING" | Reddit thread | https://www.reddit.com/r/USC/comments/1sz911f/off_campus_housing/ |
| 7 | RateMyDorm — McCarthy Honors reviews | Dorm review page | https://www.ratemydorm.com/reviews/university-of-southern-california/university-of-southern-california-mccarthy-honors-residential-college |
| 8 | Yelp — The Residences at Lorenzo | Yelp business reviews | https://www.yelp.com/biz/the-residences-at-lorenzo-los-angeles |
| 9 | Yelp — University Gateway | Yelp business reviews | https://www.yelp.com/biz/university-gateway-los-angeles-2 |
| 10 | r/AskLosAngeles thread: "Honestly, what is the area around USC like for a single female?" | Reddit thread | https://www.reddit.com/r/AskLosAngeles/comments/1s205xq/honestly_what_is_the_area_around_usc_like_for_a/ |

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** ~200 tokens (measured with the MiniLM tokenizer used downstream)

This should be the sweet spot to capture a full Reddit comment or a detailed paragraph from a Yelp review without diluting the context.

**Overlap:** ~40 tokens

If a student's review spills over a hard paragraph break, the overlap ensures the connection between their thoughts isn't lost in the split.

**Preprocessing before chunking:**

`preprocess.py` reads the raw HTML in `documents/raw_data/source_*_raw_*.html`, strips scripts/styles/navigation/share-button noise, and writes per-source plain-text files (`source_N_<type>_clean.txt`). Each source type uses a dedicated handler that emits a consistent flat format — Reddit threads become `Thread Title:` / `Original Post by …` / `Comment by …` / `Reply by X to Y …` lines, Yelp business pages become `Business:` / `Location:` headers plus `Review by …` and `Reply by … (Business Owner) …` blocks, RateMyDorm and the r/USC wiki get analogous treatments. `chunk.py` then loads those clean files and runs LangChain's `RecursiveCharacterTextSplitter` against them.

**Why these choices fit your documents:**

The corpus relies heavily on Reddit threads and Yelp reviews, so the information is highly conversational and opinionated. Review-style text naturally occurs in short bursts, so the chunks should be smaller than what would suit a long-form textbook. A recursive character splitter is used because it tries paragraphs first, then sentences, then words — which lines up with the blank-line-separated comment/review blocks the preprocess step emits.

The chunker is configured with the `all-MiniLM-L6-v2` tokenizer (the same one the embedding step uses), so the "~200 tokens" target reflects the model's actual view of the text rather than a character-based approximation. Each chunk also carries a `parent_title` metadata field (thread title for Reddit, dorm name for RateMyDorm, business for Yelp, wiki title for wiki) so a chunk complaining about "the terrible elevators" still tells the generation stage which building it's about, even after the chunker drops the header.

**Final chunk count:**

165 chunks total across 10 sources (min 83 chars, avg 647 chars, max 1008 chars). 
Per-type split: 101 Reddit, 46 Yelp, 13 RateMyDorm, 5 wiki. The full set is persisted to `documents/chunks.jsonl`.

**Sample chunks:**

*1. From `source_10_reddit_clean.txt` (chunk #3, 527 chars) — parent thread: "Honestly, what is the area around USC like for a single female?"*

> Reply by LAD17Decoy to Independent_Owl_4292: Haha, I had 6 bikes stolen from me during Undergrad at USC. I once went to a bike store in the area and found my bike that had just been stolen. It was hilarious. Biking in the area is safe and nobody is going to actually steal your bike while you're riding it. The USC area is a safe place. Just don't go west of Vermont or south of campus. Also don't go east of the Galen Center. USC was the best experience of my life. You will feel safe if you just follow the rules I mentioned.

*2. From `source_1_wiki_clean.txt` (chunk #0, 383 chars) — parent: "Off-Campus Housing Options"*

> \# Guide: Off-Campus Housing Options
>
> Are you in need to research off-campus housing options? Here are some suggestions..
>
> \## University Park
> Located close to USC campus, these apartment buildings or rental homes are generally located close to campus on the north, east, or west sides. It is recommended to live within the DPS patrol area and the north side of campus is most popular.

*3. From `source_7_ratemydorm_clean.txt` (chunk #0, 673 chars) — parent: "McCarthy Honors Residential College"*

> Dorm: McCarthy Honors Residential College
> Location: University of Southern California
> Overall Rating: 4.8/5 (Based on 23 reviews)
>
> --- REVIEW ---
> User: [Anonymous]
> Rating: 5/5
> Time: 3 years ago
> Content: McCarthy is great! New, clean facilities, room common spaces cleaned 2x per week. My favorite part is the study lounges on every floor which have a variety of different setups and table sizes :) Only possible downsides: bedroom size can widely vary depending on which room type you have, and the windows are kind of small. I've seen bedrooms that are nearly double the size of mine. But overall a really great place to live, location is amazing (especially for SCA ppl)!

*4. From `source_8_yelp_clean.txt` (chunk #0, 257 chars) — parent: "The Residences at Lorenzo"*

> Business: The Residences at Lorenzo
> Location: 325 W Adams Blvd, Los Angeles, CA 90007
>
> --- Q&A SECTION ---
>
> Question: Why are there only 559 "visible" reviews yet 711 "removed" reviews here on Yelp?
> Answer by Diana M.: Yelp, removes reviews
>
> --- REVIEWS ---

*5. From `source_8_yelp_clean.txt` (chunk #12, 733 chars) — parent: "The Residences at Lorenzo"*

> Reply by J T. (Business Owner) to Butian X. (Jan 24, 2026): Hi Butian, Thank you again for taking the time to share your detailed feedback. While we have already responded to your review on Google, we also wanted to acknowledge your comments here as well. We're sorry to hear that your experience at The Lorenzo has not met expectations, and we understand how frustrating ongoing concerns such as elevator disruptions, noise, lighting, and building alarms can be. We take resident feedback seriously and would appreciate the opportunity to review your concerns further. Please reach out to our team directly at info@ghpmgmt.com with your unit information and any additional details you'd like to share so we can follow up and assist.

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**

- `all-MiniLM-L6-v2` via `sentence-transformers` chosen because it runs locally, produces strong semantic embeddings for conversational and review-style text, and requires no API key or external service.

**Production tradeoff reflection:**

- **Domain-specific accuracy:** `all-MiniLM-L6-v2` is a general-purpose model. For better performance on Reddit/Yelp slang and domain-specific phrasing, a model fine-tuned on conversational or review data would likely improve retrieval precision.
- **Context length:** This project chunks documents (~200 tokens) before embedding. If we needed to embed very large documents without chunking, prefer embedding models that support larger context windows or use hierarchical retrieval.
- **Latency & scalability:** Local models give low-latency, cost-free inference for development and grading, but require more operational effort to scale. Hosted API models simplify scaling at monetary cost.
- **Multilingual & robustness:** If supporting other languages or noisy text, consider multilingual or larger transformer models that handle slang/typos more robustly.
- **Operational note:** `all-MiniLM-L6-v2` was selected to keep the pipeline reproducible on a local machine. Changing the embedding model requires rebuilding the ChromaDB index (`embed.py --rebuild`).
---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

The system prompt defined in `generate.py` (`SYSTEM_PROMPT`) enforces grounding through five strict rules that get sent on every call.

* The prompt opens by naming the assistant "The Unofficial Guide" and scoping it to off-campus housing and day-to-day living near USC and Downtown LA. The scope explicitly covers building reviews, shuttle reliability, neighborhood safety, supermarkets, transit, and other everyday realities of the area.
* Rule 1 tells the model to answer ONLY using the retrieved documents and to ignore prior training or outside knowledge.
* Rule 2 tells the model to cite inline using bracketed numbers like [1] or [1][2]. The numbers correspond to the [N] source_filename headers in the documents block, and the model is told not to invent new numbers or write filenames inline.
* Rule 3 forces the model to reply with the exact sentence "I don't have enough information on that" when the documents do not cover the question. The exact wording is important because the UI, the test harness, and the evaluation table all rely on matching that string to detect the refusal path.
* Rule 4 forbids inventing facts, building names, prices, quotes, or details.
* Rule 5 asks for a concise paragraph or short bullets, and tells the model NOT to write its own References section because one is appended programmatically.

**How source attribution is surfaced in the response:**

Attribution is handled both inline by the model and as a programmatic References block, so it is verifiable even if the model misbehaves.

* Before the prompt is sent, `_assign_citation_numbers` in `generate.py` walks the retrieved chunks in retrieval order and assigns a number to each unique source filename. Chunks that share a source share a number, matching academic paper citation style.
* `_build_context` then renders each chunk under a header that includes the citation number, the source filename, and the parent_title metadata (thread title for Reddit, dorm name for RateMyDorm, business for Yelp, wiki title for wiki). This is how the model knows which number to use when it cites a chunk.
* After the model returns its answer, the code parses the inline citations with a regex (`CITATION_PATTERN`), keeps only the sources the model actually cited, and appends a programmatic `References:` block listing them in citation order.
* The References block is skipped when the model returns the exact refusal sentence, so refusals never carry fake attributions.
* In the Gradio UI, the same citation numbers also drive the "Retrieved chunks" debug panel, which shows the rank, citation number, source, parent_title, cosine distance, and a short preview for every chunk that was sent to the model. This lets a reviewer confirm exactly what the model was given before it wrote its answer.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How are the reliability and wait times of the shuttles at The Residences at Lorenzo? | The shuttles are rarely on time and are often crowded to maximum capacity. | Mentions that Lorenzo has a free shuttle, but primarily retrieves complaints about broken elevators, package delivery, and other amenities. It does not address shuttle reliability or wait times. | Partially relevant | Inaccurate |
| 2 | Which specific streets or patrol boundaries are considered the safest for walking around the USC campus at night? | 30th St, 29th St, Ellendale Place, Orchard Ave, and USC Village. | Recommends staying on 36th or 37th streets or within the "USC bubble" bounded by Expo, Fig, Adams, and Vermont. | Partially relevant | Inaccurate |
| 3 | When comparing Tuscany and Icon, which apartment complex is more expensive? | Icon Plaza is generally more expensive because it offers true private single-bedroom units. | States that Tuscany is not that expensive, while Icon is considered really expensive, but lacks the specific reasoning about single-bedroom units. | Relevant | Partially accurate |
| 4 | Which off-campus housing companies or apartment buildings offer furnished rooms or lenient guarantor requirements for international students? | International students frequently choose housing companies such as Tripalink, Stuho, and Orion Housing, as well as large apartment complexes like The Lorenzo and University Gateway. | Retrieves discussions from international/transfer students mentioning Gateway, Hub, and Mosaic student housing, but does not mention lenient guarantor requirements or the specific companies expected. | Partially relevant | Inaccurate |
| 5 | What are the most common daily annoyances regarding the elevators and street noise at University Gateway? | Tenants frequently complain that the eight available elevators take too long to arrive during peak morning hours and other busy times. Units facing the main roads also experience high levels of city and traffic noise. | Mentions that elevators have been broken for nearly a month and are out of service, but misses the specific details about "eight available elevators" and "peak morning hours". | Relevant | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**
> How are the reliability and wait times of the shuttles at The Residences at Lorenzo?

**What the system returned:**
>I don't have enough information on that.

**Root cause (tied to a specific pipeline stage):**

* My first guess was that this was a chunking problem. I thought the relevant shuttle details might have been split across chunk boundaries, so retrieval was only returning partial context.
* To test that theory, I rephrased the question to ask about general commute issues, but the system returned the same kind of incomplete answer.
* The real root cause sits upstream of chunking. The source data itself does not contain enough detail on shuttle reliability or wait times.
* The Lorenzo Yelp reviews acknowledge that a shuttle exists, but reviewers focused their complaints on elevators, package delivery, and noise. Wait times are not discussed.
* A secondary issue lived in the generation stage. My original system prompt was scoped too narrowly to "housing questions only", so transit adjacent questions were sometimes refused even when partial information was available.

**What you would change to fix it:**

* I widened the generation prompt from "only housing questions" to "housing and any living related question, including buses, area, and safety". This lets the model surface adjacent information instead of refusing outright.
* I confirmed the data gap by asking "How are the reliability and wait times of the shuttles" again without the Lorenzo qualifier. The system correctly returned that Lorenzo has a shuttle but that no document describes its reliability.
* On the data side, the fix would be to add a source that specifically covers shuttle experiences, such as a Reddit thread on USC area transportation or rider side reviews of the Lorenzo shuttle.


---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

The five evaluation questions I wrote in `planning.md` ended up being the most useful part of the spec for me.

* Having those questions locked in before any pipeline code was written gave me a fixed target to test against at every stage.
* When an answer came back wrong, the fixed questions let me trace the failure to a specific stage. If retrieval was pulling the wrong chunks, the problem was likely chunk size or the embedding model. If retrieval looked correct but the answer was still off, the problem was almost always the generation prompt or a gap in the source data.
* The shuttle question is a clear example. Retrieval surfaced Lorenzo content but the final answer was still wrong, which told me the issue was not in chunking but further down the pipeline in generation and data coverage.
* Without that fixed list of questions, I would have spent a lot more time changing settings at random without knowing which stage I was actually improving.

**One way your implementation diverged from the spec, and why:**

For the UI design, I did not write much detail in `planning.md` up front.

* The first Gradio implementation Claude produced was just a query box and a single answer box, with no debug or verbose information.
* After seeing that output, I went back and updated my prompt to ask for a verbose audit panel showing the retrieved chunks, plus a sidebar with temperature and top_k sliders so the result could be fine tuned at query time.
* The final `app.py` reflects that revised design, not the minimal one the spec originally implied.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

* *What I gave the AI:* I gave Claude my UI plan from `planning.md` and asked it to scaffold the Gradio app in `app.py`. The plan only said "a Gradio interface that takes a question and shows the answer".
* *What it produced:* A minimal Gradio app with one input textbox, an Ask button, and a single answer output. There was no way to see which chunks were retrieved and no way to adjust top_k or temperature without editing the code.
* *What I changed or overrode:* I rewrote the prompt to specify a `Blocks` layout with a left sidebar for top_k, temperature, and max_tokens sliders, plus a main panel containing both an Answer box and a separate debug panel for the retrieved chunks. That panel shows rank, source, parent_title, cosine distance, and a preview for each hit. The final `app.py` matches that revised design.

**Instance 2**

* *What I gave the AI:* I gave Claude the grounded generation requirements from the rubric and asked it to write the system prompt for `generate.py`. My initial instruction said the assistant should only answer "off campus housing questions" and refuse anything else.
* *What it produced:* A system prompt that refused any question not directly about apartments or dorms. During evaluation, the shuttle and safety questions started returning "I don't have enough information on that" even when relevant context was retrieved.
* *What I changed or overrode:* I overrode the scope rule and asked Claude to broaden it to "housing and any living related question, including buses, area, and safety". I also kept the strict citation and refusal rules from the original prompt, since those parts were working correctly and align with the rubric requirement for visible source attribution.
