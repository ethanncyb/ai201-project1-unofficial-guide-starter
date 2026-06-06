"""Flatten raw HTML sources in documents/ into plain-text files for chunking.

Usage:
    python preprocess.py                          # process every source_*_raw_*.html
    python preprocess.py documents/source_2_raw_reddit.html   # one file
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Tag

DOCUMENTS_DIR = Path(__file__).parent / "documents"
SOURCE_RE = re.compile(r"^source_(\d+)_raw_(reddit|wiki|ratemydorm|yelp)\.html$")
DATE_RE = re.compile(r"^[A-Z][a-z]{2,9} \d{1,2}, \d{4}$")


def _clean_text(s: str) -> str:
    """Collapse whitespace, normalize non-breaking spaces, strip."""
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------- Reddit ----------

def _reddit_body_text(scope: Tag) -> str:
    """Pull the rendered comment/post body out of a shreddit element."""
    body = scope.find("div", id=lambda v: v and v.endswith("-post-rtjson-content"))
    if not body:
        return ""
    return _clean_text(body.get_text(" ", strip=True))


def flatten_reddit(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    post = soup.find("shreddit-post")
    if not post:
        return ""

    title = post.get("post-title", "").strip()
    op_author = post.get("author", "[unknown]").strip() or "[unknown]"
    op_body = _reddit_body_text(post)

    lines = [f"Thread Title: {title}", ""]
    if op_body:
        lines.append(f"Original Post by {op_author}: {op_body}")
        lines.append("")

    comments = soup.find_all("shreddit-comment")
    # thingid → author lookup for resolving "Reply by X to Y"
    author_by_id = {
        c.get("thingid"): (c.get("author") or "[unknown]").strip() or "[unknown]"
        for c in comments
        if c.get("thingid")
    }

    for c in comments:
        author = (c.get("author") or "[unknown]").strip() or "[unknown]"
        depth = int(c.get("depth") or 0)
        body = _reddit_body_text(c)
        if not body:
            continue
        if depth == 0:
            lines.append(f"Comment by {author}: {body}")
        else:
            parent_id = c.get("parentid")
            parent_author = author_by_id.get(parent_id, "[unknown]")
            lines.append(f"Reply by {author} to {parent_author}: {body}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------- Wiki ----------

def flatten_wiki(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_=lambda c: c and "wiki" in c)
    if not container:
        return ""

    lines: list[str] = []
    for el in container.children:
        if not isinstance(el, Tag):
            continue
        name = el.name
        if name == "h1":
            text = _clean_text(el.get_text(" ", strip=True))
            # Strip a leading "FAQ:" so the wiki reads as "# Guide: <topic>"
            text = re.sub(r"^FAQ:\s*", "", text, flags=re.IGNORECASE)
            lines.append(f"# Guide: {text}")
            lines.append("")
        elif name == "h2":
            text = _clean_text(el.get_text(" ", strip=True))
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"## {text}")
        elif name == "p":
            text = _clean_text(el.get_text(" ", strip=True))
            if text:
                lines.append(text)
                lines.append("")
        elif name == "h3":
            # h3 inside the wiki is a list item, usually wrapping a link to an apartment
            link = el.find("a")
            label = _clean_text(el.get_text(" ", strip=True))
            if link and link.get("href"):
                href = link["href"].strip()
                lines.append(f"- {label}: {href}")
            elif label:
                lines.append(f"- {label}")
        elif name == "ul":
            for li in el.find_all("li", recursive=False):
                link = li.find("a")
                label = _clean_text(li.get_text(" ", strip=True))
                if link and link.get("href"):
                    lines.append(f"- {label}: {link['href'].strip()}")
                elif label:
                    lines.append(f"- {label}")
            lines.append("")
        elif name == "hr":
            continue

    return _collapse_blank_runs("\n".join(lines)).rstrip() + "\n"


# ---------- RateMyDorm ----------

def _ratemydorm_aggregate(soup: BeautifulSoup) -> dict:
    """Pull dorm name, rating, count from the JSON-LD AggregateRating block."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("@type") == "AggregateRating":
            item = data.get("itemReviewed") or {}
            return {
                "name": item.get("name", "").strip(),
                "rating": data.get("ratingValue"),
                "count": data.get("ratingCount"),
            }
    return {}


def flatten_ratemydorm(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        # keep ld+json — we already harvested it above, but re-parse here
        if s.get("type") != "application/ld+json":
            s.decompose()

    agg = _ratemydorm_aggregate(BeautifulSoup(html, "html.parser"))

    # Location is the school name; lives in an <h3> that is NOT a link
    location = ""
    for h3 in soup.find_all("h3"):
        text = _clean_text(h3.get_text(" ", strip=True))
        if text and "Southern California" in text and not h3.find("a"):
            location = text
            break

    name = agg.get("name") or ""
    rating = agg.get("rating")
    count = agg.get("count")

    header = [f"Dorm: {name}"]
    if location:
        header.append(f"Location: {location}")
    if rating is not None and count is not None:
        header.append(f"Overall Rating: {rating}/5 (Based on {count} reviews)")

    review_sections = soup.find_all(
        "section",
        class_=lambda c: c and "space-y-6" in c and "py-4" in c,
    )

    blocks: list[str] = []
    for sec in review_sections:
        stars = sec.find_all("svg", class_=lambda c: c and "text-yellowRating" in c)
        rating_n = len(stars)

        # Time-ago lives in a <p> with "ago" text
        time_text = ""
        for node in sec.find_all(string=lambda s: s and "ago" in s):
            stripped = node.strip()
            if stripped and len(stripped) < 40 and "ago" in stripped.lower():
                time_text = stripped
                break

        # Setup: room type label rendered as "Lived in a <type>" — sometimes split across nodes
        setup = ""
        sec_text = _clean_text(sec.get_text(" ", strip=True))
        m = re.search(r"Lived in (?:an?|the)\s+([A-Za-z][A-Za-z\- ]{2,30})", sec_text)
        if m:
            setup = m.group(1).strip().rstrip(".").title()

        # Review body is the longest <p> in the section
        body = ""
        for p in sec.find_all("p"):
            txt = _clean_text(p.get_text(" ", strip=True))
            # Skip the "X years ago" timestamp paragraph
            if "ago" in txt.lower() and len(txt) < 40:
                continue
            if txt.lower() == "verified student":
                continue
            if len(txt) > len(body):
                body = txt

        block = ["--- REVIEW ---", "User: [Anonymous]"]
        if rating_n:
            block.append(f"Rating: {rating_n}/5")
        if time_text:
            block.append(f"Time: {time_text}")
        if setup:
            block.append(f"Setup: {setup}")
        if body:
            block.append(f"Content: {body}")
        blocks.append("\n".join(block))

    out = "\n".join(header) + "\n\n" + "\n\n".join(blocks)
    return out.rstrip() + "\n"


# ---------- Yelp ----------

def _yelp_business_name(soup: BeautifulSoup) -> str:
    """Yelp dumps don't include <head>, so derive the name from 'Start your review of <name>'."""
    for a in soup.find_all("a"):
        txt = a.get_text(" ", strip=True)
        if txt.startswith("Start your review of"):
            return txt.replace("Start your review of", "", 1).strip()
    return ""


def _yelp_address(soup: BeautifulSoup) -> str:
    tag = soup.find("address")
    if not tag:
        return ""
    parts = [_clean_text(line) for line in tag.stripped_strings]
    return ", ".join(p for p in parts if p)


def _yelp_qa_block(soup: BeautifulSoup) -> str:
    """Build the Q&A section by parsing each Q/A <li> under 'Ask the Community'."""
    h = soup.find("h2", string="Ask the Community")
    if not h:
        return ""
    section = h
    for anc in h.parents:
        if anc.name == "section":
            section = anc
            break

    # Each Q/A pair is rendered as <li> containing a "Q:" marker and an "A:" marker
    qa_lis = []
    for q_marker in section.find_all(string=lambda s: s and s.strip() == "Q:"):
        for anc in q_marker.parents:
            if anc.name == "li":
                if anc not in qa_lis:
                    qa_lis.append(anc)
                break

    if not qa_lis:
        return ""

    pieces = ["--- Q&A SECTION ---", ""]
    for li in qa_lis:
        # Question text is the <p> that immediately follows the Q: marker
        question = ""
        q_marker = li.find(string=lambda s: s and s.strip() == "Q:")
        if q_marker:
            # Find the nearest following <p> in DOM order
            for el in q_marker.parent.find_next("p") if q_marker.parent else []:
                pass
            p = q_marker.parent.find_next("p") if q_marker.parent else None
            if p:
                question = _clean_text(p.get_text(" ", strip=True))

        # Answer text is the <p> that follows the A: marker
        answer = ""
        a_marker = li.find(string=lambda s: s and s.strip() == "A:")
        if a_marker:
            p = a_marker.parent.find_next("p") if a_marker.parent else None
            if p:
                answer = _clean_text(p.get_text(" ", strip=True))

        # Answerer name: first try /user_details anchor; older Q&A renders the name as plain text
        answerer = ""
        for a in li.find_all("a", href=lambda h: h and "/user_details" in str(h)):
            txt = _clean_text(a.get_text(" ", strip=True))
            if txt:
                answerer = txt
                break
        if not answerer and a_marker:
            # Walk siblings after the answer paragraph, accepting the first plausible name/role span
            ans_p = a_marker.parent.find_next("p") if a_marker.parent else None
            if ans_p:
                for span in ans_p.find_all_next("span"):
                    candidate = _clean_text(span.get_text(" ", strip=True))
                    if not candidate:
                        continue
                    lower = candidate.lower()
                    if candidate in {"more", "less"} or lower.startswith(("read more", "see ", "show ")):
                        continue
                    if "ago" in lower:
                        break
                    if len(candidate) > 60:
                        continue
                    answerer = candidate
                    break

        if question:
            pieces.append(f"Question: {question}")
            if answer:
                label = f"Answer by {answerer}" if answerer else "Answer"
                pieces.append(f"{label}: {answer}")
            pieces.append("")

    return "\n".join(pieces).rstrip() + "\n"


def _yelp_review_lis(soup: BeautifulSoup) -> list[Tag]:
    """Each Yelp review wraps in an <li>. Find lis that contain a comment paragraph."""
    seen: list[Tag] = []
    for p in soup.select('p[class*="comment__"]'):
        for anc in p.parents:
            if anc.name == "li":
                if anc not in seen:
                    seen.append(anc)
                break
    return seen


def _yelp_extract_review(li: Tag) -> dict:
    """Pull current review body, owner reply body, both dates, reviewer, owner name out of a single Yelp <li>."""
    review_body = ""
    reply_body = ""
    review_date = ""
    reply_date = ""

    # Walk in DOM order — last seen date is the date that belongs to the next comment paragraph
    last_date = ""
    last_date_seen_count = 0
    for el in li.descendants:
        if isinstance(el, str):
            text = el.strip()
            if text and DATE_RE.match(text):
                # Dedupe immediately-repeated dates (some reviews render the date twice)
                if text != last_date:
                    last_date = text
                    last_date_seen_count = 1
                else:
                    last_date_seen_count += 1
            continue
        if not isinstance(el, Tag):
            continue
        if el.name != "p":
            continue
        cls = el.get("class") or []
        cls_str = " ".join(cls)
        if "comment__" not in cls_str:
            continue
        # Classify by inner span: lang="en" → user-written, else → owner reply
        inner_lang_span = el.find("span", attrs={"lang": "en"})
        body = _clean_text(el.get_text(" ", strip=True))
        if not body:
            continue
        is_truncated = "truncated__" in cls_str
        if inner_lang_span and not is_truncated:
            # Current review body
            if not review_body:
                review_body = body
                review_date = last_date
        elif not inner_lang_span:
            # Owner reply body
            if not reply_body:
                reply_body = body
                reply_date = last_date
        # else: truncated user review (previous version) — skip

    reviewer = ""
    for a in li.find_all("a", href=lambda h: h and "/user_details" in str(h)):
        txt = _clean_text(a.get_text(" ", strip=True))
        if txt:
            reviewer = txt
            break

    rating = ""
    for el in li.find_all(attrs={"aria-label": True}):
        label = str(el.get("aria-label", ""))
        m = re.match(r"^(\d+(?:\.\d+)?) star rating$", label)
        if m:
            rating = m.group(1)
            break

    owner_name = ""
    # Yelp tags the responder with either "Business Owner" or "Business Customer Service"
    owner_label = li.find(string=lambda s: s and s.strip() in ("Business Owner", "Business Customer Service"))
    if owner_label:
        label_text = owner_label.strip()
        for anc in owner_label.parents:
            txt = _clean_text(anc.get_text(" | ", strip=True))
            marker = f" | {label_text}"
            if marker in txt and len(txt) < 120:
                candidate = txt.split(marker)[0].strip()
                candidate = re.sub(r"^Business owner information\s*\|?\s*", "", candidate).strip()
                if candidate:
                    owner_name = candidate
                    break

    return {
        "reviewer": reviewer or "[unknown]",
        "rating": rating,
        "review_date": review_date,
        "review_body": review_body,
        "reply_body": reply_body,
        "reply_date": reply_date,
        "owner_name": owner_name or "[unknown]",
    }


def _yelp_reviews(soup: BeautifulSoup) -> str:
    lis = _yelp_review_lis(soup)
    if not lis:
        return ""

    pieces = ["--- REVIEWS ---", ""]
    for li in lis:
        r = _yelp_extract_review(li)
        if not r["review_body"]:
            continue
        rating_part = f"{r['rating']} stars" if r["rating"] else "no rating"
        date_part = f", {r['review_date']}" if r["review_date"] else ""
        pieces.append(f"Review by {r['reviewer']} ({rating_part}{date_part}): {r['review_body']}")
        pieces.append("")
        if r["reply_body"]:
            reply_date_part = f" ({r['reply_date']})" if r["reply_date"] else ""
            pieces.append(
                f"Reply by {r['owner_name']} (Business Owner) to {r['reviewer']}{reply_date_part}: {r['reply_body']}"
            )
            pieces.append("")

    return "\n".join(pieces).rstrip() + "\n"


def flatten_yelp(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()

    name = _yelp_business_name(soup)
    address = _yelp_address(soup)

    header_lines: list[str] = []
    if name:
        header_lines.append(f"Business: {name}")
    if address:
        header_lines.append(f"Location: {address}")
    header_lines.append("")

    qa = _yelp_qa_block(soup)
    reviews = _yelp_reviews(soup)

    parts: list[str] = ["\n".join(header_lines)]
    if qa:
        parts.append(qa)
    if reviews:
        parts.append(reviews)

    return _collapse_blank_runs("\n".join(parts)).rstrip() + "\n"


# ---------- shared helpers ----------

def _collapse_blank_runs(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


HANDLERS = {
    "reddit": flatten_reddit,
    "wiki": flatten_wiki,
    "ratemydorm": flatten_ratemydorm,
    "yelp": flatten_yelp,
}


# ---------- CLI ----------

def process_file(path: Path) -> bool:
    m = SOURCE_RE.match(path.name)
    if not m:
        print(f"skip {path.name}: filename does not match source_N_raw_<type>.html")
        return False

    n, source_type = m.group(1), m.group(2)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        print(f"skip {path.name}: file is empty")
        return False

    handler = HANDLERS[source_type]
    output = handler(raw)
    if not output.strip():
        print(f"skip {path.name}: handler produced empty output")
        return False

    out_path = path.parent / f"source_{n}_{source_type}_clean.txt"
    out_path.write_text(output, encoding="utf-8")
    # entry count = blank-line-separated blocks, rough but useful signal
    entries = sum(1 for block in output.split("\n\n") if block.strip())
    print(f"{path} -> {out_path} ({entries} blocks)")
    return True


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        targets = [Path(p) for p in argv[1:]]
    else:
        targets = sorted(DOCUMENTS_DIR.glob("source_*_raw_*.html"))
    if not targets:
        print(f"no source HTML files found in {DOCUMENTS_DIR}")
        return 1
    for p in targets:
        process_file(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
