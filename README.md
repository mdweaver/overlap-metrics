# overlap-metrics

Investigation of metrics and tests for overlap (positivity) and extrapolation
risk when conditioning for causal inference by imputation or re-weighting.

**Stage 1 (current): investigation and research plan. No package yet.**

- `docs/investigation.md`: the written investigation. It covers potential-outcomes
  foundations for binary, multi-valued and continuous treatments; a survey and
  critique of existing diagnostics, with checked references; directions from
  other areas of mathematics; the proposed *overlap frontier* and *extrapolation
  floor*; and a ranked research plan.
- `docs/reading_notes_close_papers.md`: detailed notes on the three closest papers
  (Armstrong & Kolesár 2021; Kallus 2020; Ma, Huang & Namkoong 2026), read in full.
- `scripts/`: small numerical illustrations (run from inside `scripts/`).
- `outputs/`: figures and tables produced by the scripts.

```bash
pip install -r requirements.txt
cd scripts && for f in 0*.py; do python "$f"; done
```

Illustration 2 (`02_high_dimension.py`) takes a few minutes. The others take seconds.
