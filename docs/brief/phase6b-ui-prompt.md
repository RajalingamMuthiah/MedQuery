Phase 6b: redesign the frontend UI. Plan first; show me the plan before coding. Frontend only: no backend, API, evaluation or report changes. Keep all existing behaviour and tests passing (update tests where markup changes).

Design direction: calm, clinical, trustworthy — a cross between Perplexity (numbered citations) and a medical reference site. Generous whitespace, clear typography, one accent colour, no clutter.

1. Layout
- Header: "MedQuery" wordmark with a small icon, tagline "Answers from public health guidelines, with sources", nav (Ask · Sources · About).
- Empty state: a centred, large ask box (like a search engine home page), topic chips (Diabetes · Hypertension · Tuberculosis), and sample questions as tidy cards below.
- After asking: question at the top, answer card below, sources list under the answer, ask box pinned at the bottom (or top) for the next question.
- Footer: the not-medical-advice notice and "Built on 10 public-domain CDC/NIH/VA documents". Keep a slim, dismissible-for-the-session banner at the top as well, not a heavy sticky bar.

2. Answer card
- Replace inline citation chips with superscript numbered citations [1] [2] in the text; a "Sources" list under the answer shows each number with document title, publisher, year, page, and a "Superseded" badge where relevant.
- Clicking a number or source opens the side panel (sheet on mobile) with the passage and highlighted text, as now.
- Abstention: a gentle, well-designed card with an icon ("Not covered by the documents"), a one-line explanation and a link to Sources — never styled like an error.
- Show latency subtly (small grey text), not as a prominent line.

3. Verification ("Check this answer")
- A clear secondary button with a short note "can take up to a minute".
- While checking: a progress state on the button and a subtle shimmer on sentences.
- Results: a small coloured dot/marker beside each sentence plus a legend (Supported · Unsupported · Unverified · Note) and a one-line summary ("3 of 4 sentences supported"). Text labels for accessibility; colour never the only signal.

4. States and polish
- Loading skeleton for the answer (not just button text).
- Friendly error messages for rate limits ("Daily demo limit reached — try again tomorrow") and network errors, with a retry button.
- Smooth, subtle transitions; consistent spacing scale; rounded cards with light borders/shadows.
- Dark mode (follows system preference).
- Fully responsive (360 px phones to desktop), keyboard-accessible, visible focus rings, WCAG AA contrast.

5. About page
- Short explanation of how it works (hybrid search -> grounded answer -> citations -> optional check), the headline test results from the README (56% -> 83% correctness, false refusals 39% -> 11%, 100% of unanswerable questions declined), and limitations in plain language.

Rules: Tailwind only (no new heavy UI library unless you justify it); keep bundle small; update the frontend tests for the new markup and add tests for numbered citations, the sources list, the abstention card and the verification summary. CI green, commit, push, summary, STOP.
