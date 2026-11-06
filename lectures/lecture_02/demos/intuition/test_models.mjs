import {
  mulberry32,
  randn,
  makeTwoMoons,
  makeTwoBlobs,
  sampleBatch,
  RealNVP,
  VAE2D,
  EnergyModel,
  DiffusionModel,
  diffusionSchedule,
  factorsFor,
  OBSERVED,
} from "./models.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

const factors = factorsFor(OBSERVED);
const joint = factors.reduce((a, b) => a * b, 1);
console.log("AR factors", factors.map((v) => v.toFixed(2)).join(" × "), "=", joint.toFixed(3));
assert(Math.abs(joint - 0.378) < 1e-9, "joint should be 0.378");

function finiteDifferenceFlow() {
  const rng = mulberry32(1);
  const flow = new RealNVP(3, 8, rng);
  const data = makeTwoMoons(16, 0.08, mulberry32(2)).points;
  const loss0 = flow.nll(data, 16, true);
  const layer = flow.layers[1].net.layers[1];
  const idx = 5;
  const analytic = layer.dW[idx] / 16;
  const w0 = layer.W[idx];
  const eps = 1e-3;
  layer.W[idx] = w0 + eps;
  const lp = flow.nll(data, 16, false);
  layer.W[idx] = w0 - eps;
  const lm = flow.nll(data, 16, false);
  layer.W[idx] = w0;
  const fd = (lp - lm) / (2 * eps);
  console.log("flow grad", { loss0, analytic, fd, err: Math.abs(analytic - fd) });
  assert(Math.abs(analytic - fd) < 2e-2, "flow gradient mismatch");
}

function roundtrip() {
  const flow = new RealNVP(4, 16, mulberry32(3));
  const z = new Float32Array(20);
  const rng = mulberry32(4);
  for (let i = 0; i < z.length; i++) z[i] = randn(rng);
  const { x } = flow.forward(z, 10);
  const { z: back } = flow.inverse(x, 10);
  let max = 0;
  for (let i = 0; i < z.length; i++) max = Math.max(max, Math.abs(z[i] - back[i]));
  console.log("flow inverse error", max);
  assert(max < 1e-5, "inverse should recover z");
}

function trainFlow() {
  const rng = mulberry32(12);
  const data = makeTwoMoons(800, 0.08, mulberry32(12));
  const flow = new RealNVP(4, 24, rng);
  const buf = new Float32Array(128 * 2);
  let first = 0;
  let last = 0;
  for (let step = 0; step < 250; step++) {
    const { batch } = sampleBatch(data.points, 128, rng, buf);
    const loss = flow.nll(batch, 128, true);
    flow.adam(2e-3);
    if (step === 0) first = loss;
    last = loss;
  }
  const fixed = new Float32Array(200 * 2);
  for (let i = 0; i < fixed.length; i++) fixed[i] = randn(rng);
  const { x } = flow.forward(fixed, 200);
  let near = 0;
  for (let i = 0; i < 200; i++) {
    let best = Infinity;
    for (let j = 0; j < 400; j++) {
      const dx = x[i * 2] - data.points[j * 2];
      const dy = x[i * 2 + 1] - data.points[j * 2 + 1];
      best = Math.min(best, dx * dx + dy * dy);
    }
    if (best < 0.35 ** 2) near += 1;
  }
  console.log("flow nll", first.toFixed(3), "→", last.toFixed(3), "coverage", near / 200);
  assert(last < first - 0.15, "flow should reduce NLL");
}

function trainVAE() {
  const rng = mulberry32(5);
  const data = makeTwoMoons(600, 0.08, mulberry32(5));
  const vae = new VAE2D(rng, 24);
  const buf = new Float32Array(128 * 2);
  let first = 0;
  let last = 0;
  for (let step = 0; step < 300; step++) {
    const { batch } = sampleBatch(data.points, 128, rng, buf);
    const stats = vae.step(batch, 128, rng, { lr: 3e-3, beta: 0.4 });
    if (step === 0) first = stats.recon;
    last = stats.recon;
  }
  console.log("vae recon", first.toFixed(3), "→", last.toFixed(3));
  assert(last < first * 0.55, "vae reconstruction should drop");
}

function trainEBM() {
  const rng = mulberry32(7);
  const data = makeTwoBlobs(400, mulberry32(7));
  const model = new EnergyModel(rng, 24);
  const buf = new Float32Array(64 * 2);
  const particles = new Float32Array(64 * 2);
  for (let i = 0; i < particles.length; i++) particles[i] = randn(rng) * 1.4;
  let gap0 = 0;
  let gap1 = 0;
  for (let step = 0; step < 200; step++) {
    const grad = model.inputGrad(particles, 64);
    const stepSize = 0.04;
    for (let i = 0; i < particles.length; i++) {
      const g = Math.max(-8, Math.min(8, grad[i]));
      particles[i] += -0.5 * stepSize * g + Math.sqrt(stepSize) * 0.8 * randn(rng);
      particles[i] = Math.max(-3, Math.min(3, particles[i]));
    }
    const { batch } = sampleBatch(data.points, 64, rng, buf);
    const stats = model.step(batch, particles, 64, { lr: 2e-3 });
    if (step === 10) gap0 = stats.gap;
    gap1 = stats.gap;
  }
  const probe = new Float32Array([-1.15, 0, 1.15, 0, 0, 0, 0, 2.2]);
  const e = model.energy(probe, 4);
  console.log("ebm gap", gap0.toFixed(3), "→", gap1.toFixed(3), "energies", [...e].map((v) => v.toFixed(2)).join(", "));
  assert(e[0] < e[2] + 0.4 && e[1] < e[3], "wells should be lower than the outside probe");
  assert(e[0] < e[2] || e[1] < e[2], "a mode should sit below the barrier");
}

function trainDiffusion() {
  const rng = mulberry32(9);
  const data = makeTwoMoons(500, 0.08, mulberry32(9));
  const schedule = diffusionSchedule(24, 1e-3, 0.12);
  console.log("alpha_bar final", schedule.alphaBars[schedule.steps - 1].toFixed(4));
  const model = new DiffusionModel(rng, schedule, 32);
  const buf = new Float32Array(96 * 2);
  const labels = new Float32Array(96 * 2);
  let first = 0;
  let last = 0;
  for (let step = 0; step < 280; step++) {
    const { batch, index } = sampleBatch(data.points, 96, rng, buf);
    for (let i = 0; i < 96; i++) {
      const lab = data.labels[index[i]];
      labels[i * 2] = lab === 0 ? 1 : 0;
      labels[i * 2 + 1] = lab === 1 ? 1 : 0;
    }
    const stats = model.step(batch, labels, 96, rng, { lr: 2.5e-3 });
    if (step === 0) first = stats.mse;
    last = stats.mse;
  }
  const x = new Float32Array(80 * 2);
  const noiseRng = mulberry32(11);
  for (let i = 0; i < x.length; i++) x[i] = randn(noiseRng);
  const label = [1, 0];
  for (let t = schedule.steps - 1; t >= 0; t--) {
    const z = new Float32Array(x.length);
    if (t > 0) for (let i = 0; i < z.length; i++) z[i] = randn(noiseRng);
    const next = model.reverseStep(x, t, z, label, 1.4);
    x.set(next);
  }
  let near = 0;
  for (let i = 0; i < 80; i++) {
    let best = Infinity;
    for (let j = 0; j < 250; j++) {
      if (data.labels[j] !== 0) continue;
      const dx = x[i * 2] - data.points[j * 2];
      const dy = x[i * 2 + 1] - data.points[j * 2 + 1];
      best = Math.min(best, dx * dx + dy * dy);
    }
    if (best < 0.55 ** 2) near += 1;
  }
  console.log("diffusion mse", first.toFixed(3), "→", last.toFixed(3), "class coverage", near / 80);
  assert(last < first * 0.7, "diffusion mse should drop");
}

finiteDifferenceFlow();
roundtrip();
trainFlow();
trainVAE();
trainEBM();
trainDiffusion();
console.log("ok");
