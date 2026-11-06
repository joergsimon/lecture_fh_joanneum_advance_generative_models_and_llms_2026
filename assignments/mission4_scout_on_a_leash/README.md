# Scout Mission 4: Scout on a Leash

Core effort: 60–90 minutes across two stages. Course contribution: 10 percentage points inside the official 40% assignments component. No separate pass requirement.

## Stage A after Lecture 6

Inspect or modify one deterministic control. Produce one benign trace and one rejected trace. Define a success oracle and a safety oracle. State the measurement still missing.

```bash
python harness.py --variant controlled --task benign
python harness.py --variant controlled --task sensitive
```

## Stage B during or after Lecture 7

Compare permissive and controlled variants on identical held-out cases. Use at least five trials per variant, either locally or through the supplied replay bundle.

```bash
python evaluate_replays.py
```

Report raw counts, valid success, policy violations, latency, cost, `pass^5`, one earliest divergence, and a deploy, constrain, or reject decision. Submit the final system card plus the common Moodle-and-Git record.
