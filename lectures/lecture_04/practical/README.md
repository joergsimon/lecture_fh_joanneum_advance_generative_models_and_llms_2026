# Lecture 4 practical: Scout retrieval laboratory

This is the compact 12-15 minute in-class path. It uses only the Python standard library and runs without an LLM, GPU, API key, or network connection. All documents are fictional course fixtures.

## Baseline

From this folder, run:

```bash
python scout_retrieval_lab.py
```

Read the ranked document IDs, scores, dates, status values, and authority fields.

## Change one variable

Choose exactly one intervention:

```bash
python scout_retrieval_lab.py --top-k 5
python scout_retrieval_lab.py --current-only
python scout_retrieval_lab.py --query "Which authorised provider documents EU processing and deletion?"
```

Then test this deliberately unanswerable query:

```bash
python scout_retrieval_lab.py --query "Which approved provider offers H100 capacity and at what price?"
```

Retrieval always returns a ranked list. Students must decide whether the evidence supports an answer or requires abstention.

## Submission

Submit through the temporary Moodle activity opened during class:

1. the exact command or configuration;
2. the top-k document IDs and scores;
3. one sentence explaining what improved or failed;
4. the Git commit URL if the student changed the starter code.

The compact submission counts as participation evidence. Scout Mission 2 extends the same work with LSA, dense retrieval, hybrid ranking, or a small local LLM.
