"""Throttle bookings for the live app: the output tokens a request is expected to use.

The client throttle used to book each request's full max_tokens against Groq's 8,000 tokens per
minute, so one verify (max_tokens 4,000) blocked half the budget and a check after a few asks
waited about 60 s. These numbers come from recorded completion tokens (no API calls):
- verifier, single-call verifications in the Phase 5 dev and test runs (n = 44):
  p50 192, p95 383, max 492 -> book 600 (p95 + ~55%, above the max seen);
- answer_v2 generations in the Phase 4 dev and Phase 5 test runs (n = 67):
  p50 77, p95 215, max 264 -> book 400.
max_tokens itself is unchanged (4,000 and 600), so a long reply is never cut off. If a booking
is too low, Groq's 429 and rate-limit headers still apply: the client retries 429s with backoff
and waits out an exhausted token budget.
"""

VERIFY_EXPECTED_TOKENS = 600
ANSWER_EXPECTED_TOKENS = 400
