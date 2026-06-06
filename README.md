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

**How source attribution is surfaced in the response:**

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

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

**What the system returned:**

**Root cause (tied to a specific pipeline stage):**

**What you would change to fix it:**

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

**One way your implementation diverged from the spec, and why:**

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

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*

**Instance 2**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*
