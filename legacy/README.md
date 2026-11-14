# Legacy: the rule-based version (V5)

Not part of the submission. Kept for transparency: this is the version we built first and
compared against before switching to the ML pipeline (see `SOLUTION.md`, section 2.1).

V5 labels possession with hand-written rules on the ball: the team of the player nearest to
the ball, with hysteresis, plus a conservative dead-ball detector based on ball visibility,
ball position and restarts from the smart tagging. On degraded DFL matches it scored spellF1
0.18–0.26, against ≈0.42 for the submitted pipeline.

To run it (same environment, from the repository root):

```bash
python src/features.py
python legacy/dead_ball.py
python legacy/possession.py
python legacy/predict.py        # writes submission_v5.csv
```

The `check_*.py` scripts are the diagnostics we used while developing V5. Their comments and
messages are in Romanian.
