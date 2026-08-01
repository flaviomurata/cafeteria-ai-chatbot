# Verify grounded-answer claims independently

Partner answers use a separately configured Evidence verifier after generation: every Material claim must declare retrieved evidence IDs, and the verifier must confirm both support and complete claim coverage. The application fails closed to a Verification refusal for unavailable, ambiguous, unsupported, or incomplete verification; it produces Document-conflict disclosures itself and caches only Verified grounded answers, trading an additional model call and lower availability for a meaningful grounding boundary.
