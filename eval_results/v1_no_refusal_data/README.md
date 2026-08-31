# v1: before refusal-calibration data existed

These four files are the first clean evaluation run of the fine-tuned adapter,
taken *before* any refusal-calibration examples were added to the training set.
At that point `data/train.jsonl` held 5,509 examples across four task types —
rule_statement, clause_explanation, issue_spotting, statutory_interpretation —
and zero should_refuse / should_answer pairs. `refusal_set.py` had been written
but never run.

They are kept because they are the control arm for the project's central claim.
The design says the deterministic layer handles the disclaimer and the fine-tune
handles refusal calibration; this run measures what the fine-tune does when the
refusal half of that split has no training data behind it at all.

Headline from this run, red team, `advice_extraction` defended:

    A base zero-shot   0/6
    B base + prompt    4/6
    C fine-tuned       0/6   <- identical to the untrained baseline

The v2 run in the parent directory repeats the same frozen evaluation against an
adapter trained on 5,642 examples, the difference being 133 refusal-calibration
examples (80 should_refuse / 53 should_answer) at 2.4% of the mixture.
