# Scout Mission 1: Sampling Laboratory

Core effort: 60–90 minutes. Course contribution: 10 percentage points inside the official 40% assignments component. No separate pass requirement.

Run the same prompt with three decoding configurations and five samples per configuration. Record the exact model and runtime when using Ollama. The supplied replay data supports students without a compatible local model.

```bash
python run_sampling.py --replay --out results.json
python run_sampling.py --ollama-model qwen2.5:1.5b --out results.json
```

Submit code, environment information, `results.json`, a compact comparison, one useful difference, one failure, and the common AI-use declaration through Git plus Moodle.
