# Lecture 3 transformer laboratory

This 30-minute lab extends Scout Mission 1 rather than creating another assignment. Work in pairs and submit one compact table plus one uncertainty in Moodle.

Tasks:

1. inspect token counts for difficult strings;
2. calculate one masked attention row;
3. calculate KV-cache memory for MHA, GQA, and MQA;
4. compare supplied time-to-first-token and decode-throughput measurements;
5. state one conclusion that the measurements support and one they do not.

```bash
python transformer_lab.py
```

The fallback data are synthetic teaching fixtures, clearly labelled as such. Students with a compatible local model may replace them with measured timings.
