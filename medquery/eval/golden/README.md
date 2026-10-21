# Golden set: how to review `review_v1.csv`

`review_v1.csv` holds the candidate questions written by `qwen/qwen3.8-27b` from source pages. It opens in Excel, which displays the UTF-8 file correctly because it starts with a byte-order mark. **Save it as CSV (UTF-8)** when you're done.

## What to do in each row
| Column | What to do |
|---|---|
| `decision` | Leave `keep`, or change it to `fix` (you edited the row) or `drop` (remove it from the set) |
| `reviewed` | Put `y` on every row you checked against the source PDF. **At least 25% of rows must be marked** before the set can be frozen |
| `question`, `reference_answer` | Edit freely when `decision` is `fix` |
| `evidence` | One line per quote: `doc_id \| page \| "exact quote"`. A fixed quote must appear verbatim on that page; `freeze` re-checks this |
| `answerable` | `yes` or `no`. For `no` rows, check the question is a plausible near-miss, not off-topic |
| `notes` | Anything worth recording; it's kept in the frozen set |

`source` gives the document and page each question came from, so you can open `data/raw/<doc_id>.pdf` at that page. `generator_note` says why an unanswerable question shouldn't be answerable.

## What to look for
- The question can be answered from the cited page alone, and the reference answer is correct and complete. (For `no` rows, it can't be answered from any document.)
- The question reads naturally and doesn't give away its answer.
- Table questions really need the table; multi-chunk questions really need both pages.
- Superseded questions (JNC 7) have a reference answer that notes the 2003 guidance and that it has been superseded.

## Files
- `candidates_v1.jsonl`: every candidate, including rejected ones with `reject_reason`, rewrites with `original_question`, and the unanswerability check's verdict.
- `review_v1.csv`: accepted candidates, for your review.
- After freezing: `golden_v1.jsonl` and `golden_v1.meta.json` (sha256, review stats). The frozen set is never edited by hand. Any change is a new version.
