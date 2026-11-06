# Lecture 2 live check: Autoregressive models

Ungraded live interaction for students in the lab and online. Suggested Moodle activity type: **Feedback**. Show the analysis after submissions close; do not reveal the answer key in the activity description.

## Description shown to students

Work from the four-pixel blackboard example. Submit individually after the short pair discussion. This is ungraded; it helps us test the distinction between exact likelihood and sequential generation.

## Questions

1. **Participation mode** — Multiple choice, required
   - In the lab
   - Online

2. **For x = [1, 0, 1, 1], what is its joint probability when the four observed-value factors are 0.80, 0.75, 0.70, and 0.90?** — Multiple choice, required
   - 0.2205
   - **0.378**
   - 0.4725
   - 0.973

3. **What does “exact likelihood” mean here?** — Multiple choice, required
   - Sampling always returns the most likely image.
   - Greedy decoding and random sampling are equivalent.
   - **For a fixed complete image, every normalized conditional factor can be evaluated and multiplied without approximation.**
   - Every sample receives the same probability.

4. **If the observed value x2 changes while the raster order stays fixed, which factors may change because their context changed?** — Multiple choice, required
   - Only p(x1)
   - Only p(x2 | x1)
   - **p(x3 | x1,x2) and p(x4 | x1,x2,x3)**
   - No factor changes

5. **What would you measure to make the practical cost of autoregressive image generation visible as image resolution grows?** — Longer text, required

## Lecturer key

- Q2: 0.80 × 0.75 × 0.70 × 0.90 = 0.378.
- Q3: Exact evaluation applies to a fixed observed image; generation remains stochastic unless a deterministic decoding rule is deliberately chosen.
- Q4: A changed earlier observation changes the context of later conditionals, not earlier factors retroactively.
- Q5: Look for time or model evaluations per generated pixel/token, total sampling latency versus resolution, and optionally memory/throughput. Strong answers connect a H×W image to H·W sequential decisions in a basic raster PixelCNN.
