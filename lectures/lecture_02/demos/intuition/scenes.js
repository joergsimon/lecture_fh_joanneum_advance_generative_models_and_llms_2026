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
  forwardDiffuse,
  pOne,
  tempered,
  factorsFor,
  OBSERVED,
} from "./models.js";
import { fitBox, hsl, mix, clamp } from "./render.js";

const CAD = [0.89, 0.61, 0.29];
const GLA = [0.62, 0.82, 0.86];
const CHALK = [0.96, 0.94, 0.91];
const CORAL = [0.88, 0.48, 0.41];
const MIST = [0.62, 0.68, 0.74];
const VOID = [0.07, 0.1, 0.14, 1];
const INK = [0.12, 0.16, 0.21, 1];

function ticking(ctx) {
  return !ctx.paused || ctx.stepOnce;
}

function budget(state, dt, ctx, perSecond) {
  if (ctx.paused) return ctx.stepOnce ? 1 : 0;
  state._acc = (state._acc || 0) + dt;
  const n = Math.min(4, Math.floor(state._acc * perSecond));
  if (n > 0) state._acc -= n / perSecond;
  return n;
}

function queueTrain(state, steps = 40) {
  state.burst = (state.burst || 0) + steps;
}

function consumeTrain(state, dt, ctx, perSecond) {
  let n = budget(state, dt, ctx, perSecond);
  if (state.burst > 0) {
    const extra = Math.min(8, state.burst);
    state.burst -= extra;
    n += extra;
  }
  return n;
}

const TRAIN_ACTION = { id: "burst", label: "Train 40 steps", button: true };

function fmt(v, digits = 2) {
  return Number.isFinite(v) ? v.toFixed(digits) : "—";
}

function percentile(vals, p) {
  const copy = Array.from(vals).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!copy.length) return 0;
  return copy[Math.min(copy.length - 1, Math.floor((copy.length - 1) * p))];
}

function paintDensity(vals, pix, lo, hi) {
  const span = hi - lo || 1;
  for (let i = 0; i < vals.length; i++) {
    const t = clamp((vals[i] - lo) / span, 0, 1);
    const e = t * t * (3 - 2 * t);
    const col = e < 0.5
      ? mix([0.06, 0.09, 0.13], [0.2, 0.4, 0.5], e / 0.5)
      : mix([0.2, 0.4, 0.5], [0.95, 0.94, 0.9], (e - 0.5) / 0.5);
    const a = 0.18 + 0.82 * e;
    pix[i * 4] = col[0] * 255;
    pix[i * 4 + 1] = col[1] * 255;
    pix[i * 4 + 2] = col[2] * 255;
    pix[i * 4 + 3] = a * 255;
  }
}

function paintEnergy(vals, pix, lo, hi) {
  const span = hi - lo || 1;
  for (let i = 0; i < vals.length; i++) {
    const t = clamp((vals[i] - lo) / span, 0, 1);
    const col = t < 0.42
      ? mix([0.9, 0.58, 0.24], [0.15, 0.31, 0.42], t / 0.42)
      : mix([0.15, 0.31, 0.42], [0.05, 0.07, 0.1], (t - 0.42) / 0.58);
    pix[i * 4] = col[0] * 255;
    pix[i * 4 + 1] = col[1] * 255;
    pix[i * 4 + 2] = col[2] * 255;
    pix[i * 4 + 3] = 235;
  }
}

function drawAxes(stage, cam, step = 1) {
  stage.grid(step, [0.62, 0.68, 0.74, 0.13]);
  if (cam.xmin < 0 && cam.xmax > 0) stage.line(0, cam.ymin, 0, cam.ymax, [0.93, 0.91, 0.86, 0.22]);
  if (cam.ymin < 0 && cam.ymax > 0) stage.line(cam.xmin, 0, cam.xmax, 0, [0.93, 0.91, 0.86, 0.22]);
}

function labelTicks(stage, name, cam) {
  const x0 = Math.ceil(cam.xmin);
  const x1 = Math.floor(cam.xmax);
  for (let x = x0; x <= x1; x++) {
    if (x < cam.xmin + 0.25 || x > cam.xmax - 0.25) continue;
    stage.worldText(name, x, cam.ymin, String(x), {
      size: 11,
      color: "#8b97a3",
      align: "center",
      baseline: "bottom",
    });
  }
}

function plateBox(w, h) {
  const pad = 16;
  const top = 34;
  return { x: pad, y: top, w: Math.max(10, w - pad * 2), h: Math.max(10, h - top - pad) };
}

function full(stage, top = 16) {
  const pad = 16;
  return { x: pad, y: top, w: Math.max(10, stage.cssW - pad * 2), h: Math.max(10, stage.cssH - top - pad) };
}

function cloud(stage, pts, n, colorOf, size, alpha) {
  for (let i = 0; i < n; i++) {
    const rgb = colorOf(i);
    stage.glow(pts[i * 2], pts[i * 2 + 1], rgb, size, alpha);
  }
}

function cellRect(i) {
  const col = i % 2;
  const row = (i / 2) | 0;
  const size = 0.92;
  const gap = 0.12;
  return { x: col * (size + gap), y: (1 - row) * (size + gap), w: size, h: size };
}

function strokeRect(stage, rect, weight, color) {
  stage.quad(rect.x, rect.y, rect.w, weight, color);
  stage.quad(rect.x, rect.y + rect.h - weight, rect.w, weight, color);
  stage.quad(rect.x, rect.y, weight, rect.h, color);
  stage.quad(rect.x + rect.w - weight, rect.y, weight, rect.h, color);
}

const SUB = ["₁", "₂", "₃", "₄"];

function createAR() {
  const template = new Uint8Array(16 * 16);
  const c = 7.5;
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const dx = x - c;
      const dy = y - c;
      const r = Math.hypot(dx, dy);
      const ring = r > 3.1 && r < 5.5;
      const bite = dx > 0.6 && Math.abs(dy) < 2.4;
      template[y * 16 + x] = ring && !bite ? 1 : 0;
    }
  }
  const state = {
    mode: "likelihood",
    image: OBSERVED.slice(),
    cursor: 0,
    clock: 0,
    prefix: [],
    chosen: [],
    wait: 0,
    done: false,
    sampling: false,
    seed: 4,
    drawIndex: 1,
    history: [],
    temperature: 1,
    raster: new Uint8Array(16 * 16).fill(255),
    rasterIndex: 0,
    prefixRows: 0,
    logp: 0,
    decisions: 0,
    lastP: 0.8,
  };
  state.rng = mulberry32(state.seed);

  function resetSample() {
    state.prefix = [];
    state.chosen = [];
    state.wait = 0;
    state.done = false;
    state.sampling = false;
    state.rng = mulberry32(state.seed + state.drawIndex * 17);
  }

  function commitPixel() {
    if (state.done || state.prefix.length >= 4) return;
    if (!state.prefix.length) state.drawTemperature = state.temperature;
    const p1 = tempered(pOne(state.prefix), state.drawTemperature);
    state.lastP = p1;
    const bit = state.rng() < p1 ? 1 : 0;
    state.prefix.push(bit);
    state.chosen.push(bit ? p1 : 1 - p1);
    if (state.prefix.length === 4) {
      state.done = true;
      state.sampling = false;
      state.history.unshift({
        bits: state.prefix.slice(),
        temperature: state.drawTemperature,
        chosen: state.chosen.slice(),
      });
      if (state.history.length > 6) state.history.pop();
    }
  }

  function resetRaster() {
    state.raster.fill(255);
    state.logp = 0;
    state.decisions = 0;
    state.rasterIndex = 0;
    for (let y = 0; y < state.prefixRows; y++) {
      for (let x = 0; x < 16; x++) state.raster[y * 16 + x] = template[y * 16 + x];
    }
    while (state.rasterIndex < 256 && state.raster[state.rasterIndex] !== 255) state.rasterIndex += 1;
  }

  function rasterProb(index) {
    const x = index % 16;
    const y = (index / 16) | 0;
    const prior = template[index];
    let logit = (prior - 0.5) * 5.4;
    const pull = (nx, ny) => {
      if (nx < 0 || ny < 0) return;
      const v = state.raster[ny * 16 + nx];
      if (v === 255) return;
      logit += (v ? 0.7 : -0.7) * (prior ? 0.65 : -0.2);
    };
    pull(x - 1, y);
    pull(x, y - 1);
    const p = 1 / (1 + Math.exp(-logit / state.temperature));
    return p;
  }

  function commitRaster() {
    if (state.rasterIndex >= 256) return;
    const p = rasterProb(state.rasterIndex);
    state.lastP = p;
    const bit = state.rng() < p ? 1 : 0;
    state.raster[state.rasterIndex] = bit;
    const prob = bit ? p : 1 - p;
    state.logp += Math.log(Math.max(prob, 1e-8));
    state.decisions += 1;
    state.rasterIndex += 1;
    while (state.rasterIndex < 256 && state.raster[state.rasterIndex] !== 255) state.rasterIndex += 1;
  }

  function drawSampleHistory(stage, cam) {
    const earlier = state.history.slice(state.done ? 1 : 0);
    if (!earlier.length) return;
    const cell = 0.22;
    const gap = 0.045;
    const grid = cell * 2 + gap;
    const xRight = 2.2;
    const xLeft = cam.xmin + 0.15;
    const x0 = cam.xmax > xRight + grid + 0.12 ? xRight : xLeft;
    if (x0 + grid > cam.xmax - 0.08) return;
    stage.worldText("main", x0, Math.min(cam.ymax - 0.06, 2.15), "earlier draws", {
      size: 13,
      color: "#c5ced6",
      baseline: "bottom",
    });
    let y = 1.96 - grid;
    for (const item of earlier) {
      if (y < cam.ymin + 0.2) break;
      for (let i = 0; i < 4; i++) {
        const col = i % 2;
        const row = (i / 2) | 0;
        const on = item.bits[i] === 1;
        stage.quad(x0 + col * (cell + gap), y + (1 - row) * (cell + gap), cell, cell, on ? [0.93, 0.9, 0.84, 1] : [0.16, 0.2, 0.26, 1]);
      }
      stage.worldText("main", x0 + grid / 2, y - 0.08, `T ${item.temperature.toFixed(2)}`, {
        size: 13,
        align: "center",
        baseline: "top",
        color: "#e39b4a",
      });
      y -= grid + 0.36;
    }
  }

  resetRaster();

  return {
    id: "ar",
    nav: "Autoregression",
    title: "Four pixels",
    state,
    setMode(id) {
      state.mode = id;
      state.clock = 0;
      if (id === "sample") resetSample();
      if (id === "raster") resetRaster();
    },
    setControl(id, value) {
      state[id] = value;
      if (id === "prefixRows" && state.mode === "raster") resetRaster();
      if (id === "temperature" && state.mode === "raster") resetRaster();
      if (id === "temperature" && state.mode === "sample" && !state.sampling) {
        if (state.prefix.length) {
          state.drawIndex += 1;
          resetSample();
        }
      }
    },
    action(id) {
      if (id === "sample") {
        if (state.prefix.length) state.drawIndex += 1;
        resetSample();
        state.drawTemperature = state.temperature;
        state.sampling = true;
      }
      if (id === "redraw") {
        state.drawIndex = 1;
        state.history = [];
        resetSample();
      }
      if (id === "reraster") resetRaster();
    },
    pointer(hit) {
      if (!hit || state.mode !== "likelihood") return;
      for (let i = 0; i < 4; i++) {
        const rect = cellRect(i);
        if (hit.x >= rect.x && hit.x <= rect.x + rect.w && hit.y >= rect.y && hit.y <= rect.y + rect.h) {
          state.image[i] = state.image[i] ? 0 : 1;
        }
      }
    },
    update(dt, ctx) {
      if (state.mode === "sample") {
        if (ctx.stepOnce) {
          state.sampling = false;
          commitPixel();
          return;
        }
        if (!state.sampling || state.done) return;
        state.wait += Math.min(dt, 0.05);
        if (state.wait > 0.48) {
          state.wait = 0;
          commitPixel();
        }
        return;
      }
      if (!ticking(ctx)) return;
      const step = ctx.paused ? 0.75 : dt;
      if (state.mode === "likelihood") {
        state.clock += step;
        if (state.clock > 0.85) {
          state.clock = 0;
          state.cursor = (state.cursor + 1) % 4;
        }
      } else if (state.rasterIndex < 256) {
        state.wait += step;
        const pace = ctx.paused ? 1 : 0.045;
        if (state.wait > pace) {
          state.wait = 0;
          const burst = ctx.paused ? 1 : 2;
          for (let k = 0; k < burst && state.rasterIndex < 256; k++) commitRaster();
        }
      }
    },
    spec() {
      const image = state.mode === "sample" ? state.prefix : state.image;
      const base = factorsFor(OBSERVED, 1);
      const shown = state.mode === "sample"
        ? state.chosen
        : factorsFor(state.image, 1);
      const pending = state.mode === "sample" && !state.done
        ? tempered(pOne(state.prefix), state.prefix.length ? state.drawTemperature : state.temperature)
        : null;
      let sentence;
      let tryThis;
      let limit;
      if (state.mode === "likelihood") {
        const changed = state.image.some((v, i) => v !== OBSERVED[i]);
        sentence = changed
          ? "Flipping a pixel leaves earlier factors alone. Later factors change only because their context now includes the new value."
          : "For this fixed image, every factor is already a probability. Multiply them and you have the exact probability of the four pixels.";
        tryThis = changed
          ? "Plain numbers are unchanged. Underlined: that pixel’s own value changed. Coral: a later factor, changed by context."
          : "Click a pixel to flip it, then see which of the four factors move.";
        limit = "Exact likelihood scores one finished image. Sampling is still a sequence of random choices.";
      } else if (state.mode === "sample") {
        sentence = state.prefix.length
          ? "Sampling commits to one pixel, then asks the next conditional with that pixel inside the context."
          : "The bar is the chance the next pixel is on, at the temperature you set. Nothing is drawn until you sample.";
        tryThis = "Set the temperature, then press Sample. Each finished draw stays on the right, so a colder run and a hotter run can sit together.";
        limit = "One completed row is a single trajectory. Restart from the seed to draw that same sequence again.";
      } else {
        sentence = "A masked model can score every observed pixel in one pass. Writing the image still takes one decision per pixel, in order.";
        tryThis = "Raise the observed prefix. Only the unknown rows are sampled, each one conditioned on everything above it.";
        limit = "The crescent is a hand-specified conditional, so the raster is easy to see. It is not the Fashion-MNIST PixelCNN.";
      }
      const product = [];
      if (state.mode !== "raster") {
        for (let i = 0; i < 4; i++) {
          if (i) product.push("×");
          const known = state.mode === "likelihood" || i < state.prefix.length;
          let tone = "dim";
          let text = "·";
          if (known && state.mode === "likelihood") {
            text = shown[i].toFixed(2);
            if (Math.abs(shown[i] - base[i]) < 1e-9) tone = i === state.cursor ? "hot" : "same";
            else tone = state.image[i] !== OBSERVED[i] ? "value" : "context";
          } else if (known) {
            text = shown[i].toFixed(2);
            tone = "same";
          } else if (pending != null && i === state.prefix.length) {
            text = pending.toFixed(2);
            tone = "hot";
          }
          product.push({ text, tone });
        }
      }
      const readouts = [];
      if (state.mode === "likelihood") {
        const joint = shown.reduce((a, b) => a * b, 1);
        readouts.push({ label: "Joint probability", value: fmt(joint, 3) });
        readouts.push({
          label: "Negative log-likelihood",
          value: fmt(-Math.log(joint), 3),
        });
      } else if (state.mode === "sample") {
        const joint = shown.reduce((a, b) => a * b, 1);
        readouts.push({ label: "Draw", value: state.prefix.length ? String(state.drawIndex) : "—" });
        readouts.push({
          label: state.done ? "Probability of this draw" : "Probability so far",
          value: state.prefix.length ? fmt(joint, 3) : "—",
        });
        readouts.push({
          label: state.done ? "Temperature of this draw" : "P(next pixel = 1)",
          value: state.done ? fmt(state.drawTemperature, 2) : fmt(pending, 2),
        });
      } else {
        readouts.push({ label: "Decision", value: `${Math.min(state.decisions, 256)} of ${256 - state.prefixRows * 16}` });
        readouts.push({ label: "Sum of log factors", value: fmt(state.logp, 2) });
        readouts.push({ label: "P(on | past)", value: state.rasterIndex < 256 ? fmt(rasterProb(state.rasterIndex), 2) : "done" });
      }
      return {
        sentence,
        tryThis,
        limit,
        product: state.mode === "raster" ? null : product,
        modes: [
          { id: "likelihood", label: "Exact likelihood" },
          { id: "sample", label: "Sequential sampling" },
          { id: "raster", label: "Raster order" },
        ],
        mode: state.mode,
        controls: state.mode === "likelihood" ? [] : [
          { id: "temperature", label: "Temperature", min: 0.35, max: 1.8, step: 0.05, value: state.temperature, digits: 2 },
          ...(state.mode === "raster"
            ? [{ id: "prefixRows", label: "Observed rows", min: 0, max: 12, step: 1, value: state.prefixRows, digits: 0 }]
            : []),
        ],
        choices: [],
        actions: state.mode === "sample"
          ? [{ id: "sample", label: "Sample", button: true }, { id: "redraw", label: "Restart from seed 4" }]
          : state.mode === "raster"
            ? [{ id: "reraster", label: "Rewrite the image" }]
            : [],
        readouts,
      };
    },
    draw(stage) {
      const area = full(stage, 36);
      if (state.mode !== "raster") {
        const cam = { x: area.x, y: area.y, w: area.w, h: area.h, xmin: -0.85, xmax: 2.55, ymin: -0.55, ymax: 2.35 };
        const aspect = area.w / area.h;
        const worldAspect = (cam.xmax - cam.xmin) / (cam.ymax - cam.ymin);
        if (aspect > worldAspect) {
          const mid = (cam.xmin + cam.xmax) / 2;
          const half = ((cam.ymax - cam.ymin) * aspect) / 2;
          cam.xmin = mid - half;
          cam.xmax = mid + half;
        } else {
          const mid = (cam.ymin + cam.ymax) / 2;
          const half = ((cam.xmax - cam.xmin) / aspect) / 2;
          cam.ymin = mid - half;
          cam.ymax = mid + half;
        }
        stage.plot("main", cam);
        const image = state.mode === "sample" ? state.prefix : state.image;
        const focus = state.mode === "sample" ? image.length : state.cursor;
        for (let i = 0; i < 4; i++) {
          const rect = cellRect(i);
          const known = i < image.length || state.mode === "likelihood";
          if (!known && i !== focus) {
            stage.quad(rect.x, rect.y, rect.w, rect.h, [0.1, 0.13, 0.17, 1]);
            strokeRect(stage, rect, 0.025, [0.45, 0.52, 0.58, 0.7]);
          } else if (state.mode === "sample" && i === image.length && !state.done) {
            const temp = image.length ? state.drawTemperature : state.temperature;
            const p = tempered(pOne(image), temp);
            state.lastP = p;
            stage.quad(rect.x, rect.y, rect.w, rect.h, INK);
            stage.quad(rect.x, rect.y, rect.w * p, rect.h, [0.89, 0.61, 0.29, 0.95]);
            strokeRect(stage, rect, 0.03, [0.96, 0.94, 0.91, 0.9]);
            stage.worldText("main", rect.x + rect.w / 2, rect.y + rect.h + 0.06, `${Math.round(p * 100)}% on`, {
              size: 14,
              align: "center",
              baseline: "bottom",
              color: "#e39b4a",
            });
          } else if (known) {
            const on = state.mode === "likelihood" ? state.image[i] : image[i];
            stage.quad(rect.x, rect.y, rect.w, rect.h, on ? [0.93, 0.9, 0.84, 1] : [0.16, 0.2, 0.26, 1]);
            if (state.mode === "likelihood" && i === state.cursor) {
              strokeRect(stage, rect, 0.035, [0.89, 0.61, 0.29, 1]);
            }
          }
          stage.worldText("main", rect.x + rect.w / 2, rect.y + rect.h / 2, `x${SUB[i]}`, {
            size: 22,
            align: "center",
            baseline: "middle",
            color: known && (state.mode === "likelihood" ? state.image[i] : image[i]) ? "#1b1916" : "#d5dbe2",
          });
        }
        const caption = state.mode === "likelihood"
          ? "click a tile to change the observed image"
          : state.prefix.length
            ? `seed ${state.seed}, draw ${state.drawIndex}, temperature ${state.drawTemperature.toFixed(2)}`
            : "bar width is the chance the next pixel is on";
        stage.screenText(area.x, 26, caption, { size: 13, color: "#c5ced6" });
        if (state.mode === "sample") drawSampleHistory(stage, cam);
      } else {
        const cam = fitBox(area.w, area.h, 7.5, 7.5, 9.2);
        stage.plot("main", { ...area, ...cam });
        const cell = 0.86;
        for (let i = 0; i < 256; i++) {
          const x = i % 16;
          const y = (i / 16) | 0;
          const wy = 15 - y;
          const v = state.raster[i];
          const observed = y < state.prefixRows;
          if (v === 255 && i !== state.rasterIndex) {
            stage.quad(x, wy, cell, cell, [0.09, 0.12, 0.16, 1]);
          } else if (i === state.rasterIndex && v === 255) {
            const p = rasterProb(i);
            stage.quad(x, wy, cell, cell, [0.12, 0.15, 0.2, 1]);
            stage.quad(x, wy, cell * p, cell, [0.89, 0.61, 0.29, 0.95]);
          } else {
            const on = v === 1;
            const rgb = observed
              ? (on ? [0.55, 0.76, 0.82, 1] : [0.14, 0.22, 0.28, 1])
              : (on ? [0.93, 0.9, 0.84, 1] : [0.17, 0.21, 0.27, 1]);
            stage.quad(x, wy, cell, cell, rgb);
          }
        }
        stage.screenText(area.x, 26, "glacier rows are observed; chalk rows are sampled left to right, top to bottom", {
          size: 13,
          color: "#c5ced6",
        });
      }
    },
  };
}

function createVAE() {
  const data = makeTwoMoons(700, 0.08, mulberry32(12));
  const state = {
    mode: "train",
    beta: 0.6,
    walk: 0.5,
    steps: 0,
    recon: 0,
    kl: 0,
    clock: 0,
  };

  function boot() {
    state.rng = mulberry32(21);
    state.model = new VAE2D(state.rng, 32);
    state.buf = new Float32Array(128 * 2);
    state.view = new Float32Array(160 * 2);
    state.viewLabels = new Uint8Array(160);
    for (let i = 0; i < 80; i++) {
      state.view[i * 2] = data.points[i * 2];
      state.view[i * 2 + 1] = data.points[i * 2 + 1];
      state.viewLabels[i] = 0;
      const j = 350 + i;
      state.view[(80 + i) * 2] = data.points[j * 2];
      state.view[(80 + i) * 2 + 1] = data.points[j * 2 + 1];
      state.viewLabels[80 + i] = 1;
    }
    state.samples = new Float32Array(0);
    state.eps = new Float32Array(16 * 2);
    state.steps = 0;
    let a = -1;
    let b = -1;
    for (let i = 0; i < 160; i++) {
      if (a < 0 && state.viewLabels[i] === 0) a = i;
      if (b < 0 && state.viewLabels[i] === 1) b = i;
    }
    state.endA = a;
    state.endB = b;
  }
  boot();

  function endpoints() {
    const pair = new Float32Array(4);
    pair[0] = state.view[state.endA * 2];
    pair[1] = state.view[state.endA * 2 + 1];
    pair[2] = state.view[state.endB * 2];
    pair[3] = state.view[state.endB * 2 + 1];
    const { mu } = state.model.encode(pair, 2);
    return mu;
  }

  return {
    id: "vae",
    nav: "Latent codes",
    title: "Latent codes",
    state,
    setMode(id) {
      state.mode = id;
      if (id === "train") queueTrain(state);
    },
    setControl(id, value) {
      state[id] = value;
    },
    action(id) {
      if (id === "burst") queueTrain(state);
      if (id === "reset") boot();
      if (id === "drawprior") {
        const n = 20;
        const z = new Float32Array(n * 2);
        for (let i = 0; i < z.length; i++) z[i] = randn(state.rng);
        const dec = state.model.decode(z, n);
        const merged = new Float32Array(state.samples.length + dec.length);
        merged.set(state.samples);
        merged.set(dec, state.samples.length);
        state.samples = merged.length > 400 * 2 ? merged.slice(merged.length - 400 * 2) : merged;
      }
    },
    pointer() {},
    update(dt, ctx) {
      const n = state.mode === "train" ? consumeTrain(state, dt, ctx, 36) : 0;
      if (state.mode === "train") {
        for (let s = 0; s < n; s++) {
          const { batch } = sampleBatch(data.points, 128, state.rng, state.buf);
          const stats = state.model.step(batch, 128, state.rng, { lr: 3e-3, beta: state.beta });
          state.recon = stats.recon;
          state.kl = stats.kl;
          state.steps += 1;
          state.samples = new Float32Array(0);
        }
      }
      if (state.mode === "train" && ticking(ctx)) {
        for (let i = 0; i < state.eps.length; i++) state.eps[i] = randn(state.rng);
      }
      if (ticking(ctx)) state.clock += ctx.paused ? 0 : dt;
    },
    spec() {
      const early = state.steps < 40;
      const sentence = state.mode === "walk"
        ? "The line between the two codes is straight. The decoded point can still travel through the empty gap between the moons."
        : state.mode === "sample"
          ? "A new point is drawn from the unit Gaussian, then decoded. The draw is a sample from the model, not a copy of one training moon."
          : "The encoder offers a mean and a spread. Training pulls the decode onto the moons and the codes toward the unit Gaussian.";
      const tryThis = state.mode === "walk"
        ? "Park the walk in the middle and compare that decode with the two endpoints."
        : state.mode === "sample"
          ? "Draw a handful, then go back and raise the KL weight before you draw again."
          : "Train 40 steps runs a batch whether or not Play is on. Play keeps training, and Step is a single update.";
      const limit = early && state.mode !== "train"
        ? "The weights are only a few steps old. Let training run, then come back to this view."
        : "The moons are already two-dimensional, so the warp stays on the page. A VAE is allowed to compress; a flow is not.";
      const controls = [];
      if (state.mode === "train") {
        controls.push({ id: "beta", label: "KL weight", min: 0.05, max: 4, step: 0.05, value: state.beta, digits: 2 });
      }
      if (state.mode === "walk") {
        controls.push({ id: "walk", label: "Walk", min: 0, max: 1, step: 0.01, value: state.walk, digits: 2 });
      }
      return {
        sentence,
        tryThis,
        limit,
        product: null,
        modes: [
          { id: "train", label: "Training" },
          { id: "walk", label: "Latent walk" },
          { id: "sample", label: "Sample the prior" },
        ],
        mode: state.mode,
        controls,
        choices: [],
        actions: [
          ...(state.mode === "train" ? [TRAIN_ACTION] : []),
          { id: "reset", label: "Reset weights" },
          ...(state.mode === "sample" ? [{ id: "drawprior", label: "Draw from the prior" }] : []),
        ],
        readouts: [
          ...(state.mode === "train" ? [{ label: "Steps you queued", value: String(state.burst || 0) }] : []),
          { label: "Training steps", value: String(state.steps) },
          { label: "Reconstruction", value: fmt(state.recon, 3) },
          { label: "KL", value: fmt(state.kl, 3) },
        ],
      };
    },
    draw(stage) {
      const top = 34;
      const pad = 16;
      const gap = 16;
      const w = Math.max(10, (stage.cssW - pad * 2 - gap) / 2);
      const h = Math.max(10, stage.cssH - top - pad);
      const left = { x: pad, y: top, w, h };
      const right = { x: pad + w + gap, y: top, w, h };
      const camL = fitBox(w, h, 0, 0.15, 2.45);
      const camR = fitBox(w, h, 0, 0, 2.45);
      stage.plot("data", { ...left, ...camL });
      drawAxes(stage, camL);
      const nView = 160;
      const enc = state.model.encode(state.view, nView);
      const recon = state.model.decode(enc.mu, nView);
      const gridN = 9;
      const grid = new Float32Array(gridN * gridN * 2);
      for (let j = 0; j < gridN; j++) {
        for (let i = 0; i < gridN; i++) {
          grid[(j * gridN + i) * 2] = -2 + (4 * i) / (gridN - 1);
          grid[(j * gridN + i) * 2 + 1] = -2 + (4 * j) / (gridN - 1);
        }
      }
      const decodedGrid = state.model.decode(grid, gridN * gridN);
      for (let j = 0; j < gridN; j++) {
        const row = [];
        const col = [];
        for (let i = 0; i < gridN; i++) {
          row.push([decodedGrid[(j * gridN + i) * 2], decodedGrid[(j * gridN + i) * 2 + 1]]);
          col.push([decodedGrid[(i * gridN + j) * 2], decodedGrid[(i * gridN + j) * 2 + 1]]);
        }
        stage.ribbon(row, 0.016, [0.62, 0.82, 0.86, 0.16]);
        stage.ribbon(col, 0.016, [0.62, 0.82, 0.86, 0.16]);
      }
      for (let i = 0; i < nView; i++) {
        const rgb = state.viewLabels[i] ? GLA : CAD;
        stage.glow(state.view[i * 2], state.view[i * 2 + 1], rgb, 5.5, 0.55);
      }
      for (let i = 0; i < nView; i += 2) {
        stage.glow(recon[i * 2], recon[i * 2 + 1], CHALK, 6.5, 0.9);
      }
      const muEnds = endpoints();
      const zWalk = [
        (1 - state.walk) * muEnds[0] + state.walk * muEnds[2],
        (1 - state.walk) * muEnds[1] + state.walk * muEnds[3],
      ];
      if (state.mode === "walk") {
        const dec = state.model.decode(new Float32Array(zWalk), 1);
        stage.glow(dec[0], dec[1], CORAL, 11, 1);
      }
      if (state.samples.length) {
        for (let i = 0; i < state.samples.length / 2; i++) {
          stage.glow(state.samples[i * 2], state.samples[i * 2 + 1], GLA, 6, 0.8);
        }
      }
      stage.plot("latent", { ...right, ...camR });
      drawAxes(stage, camR);
      const circle = [];
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        circle.push([Math.cos(a), Math.sin(a)]);
      }
      stage.ribbon(circle, 0.028, [0.93, 0.91, 0.86, 0.7]);
      for (let i = 0; i < nView; i += 2) {
        const rgb = state.viewLabels[i] ? GLA : CAD;
        const s0 = Math.exp(0.5 * enc.logvar[i * 2]);
        const s1 = Math.exp(0.5 * enc.logvar[i * 2 + 1]);
        if (i % 40 === 0 && s0 < 1.35 && s1 < 1.35) {
          const ell = [];
          for (let k = 0; k <= 28; k++) {
            const a = (k / 28) * Math.PI * 2;
            ell.push([enc.mu[i * 2] + s0 * Math.cos(a), enc.mu[i * 2 + 1] + s1 * Math.sin(a)]);
          }
          stage.ribbon(ell, 0.018, [...rgb, 0.55]);
        }
        stage.glow(enc.mu[i * 2], enc.mu[i * 2 + 1], rgb, 5.5, 0.9);
      }
      if (state.mode === "train") {
        const i0 = 0;
        const s0 = Math.exp(0.5 * enc.logvar[i0 * 2]);
        const s1 = Math.exp(0.5 * enc.logvar[i0 * 2 + 1]);
        for (let k = 0; k < 16; k++) {
          const zx = enc.mu[0] + s0 * state.eps[k * 2];
          const zy = enc.mu[1] + s1 * state.eps[k * 2 + 1];
          stage.glow(zx, zy, CHALK, 4, 0.7);
        }
      }
      if (state.mode === "walk") {
        stage.line(muEnds[0], muEnds[1], muEnds[2], muEnds[3], [0.88, 0.48, 0.41, 0.9]);
        stage.glow(zWalk[0], zWalk[1], CORAL, 10, 1);
        stage.glow(muEnds[0], muEnds[1], CAD, 8, 1);
        stage.glow(muEnds[2], muEnds[3], GLA, 8, 1);
      }
      stage.screenText(pad, 24, "Data space. Cadmium and glacier are the moons; chalk is the decode.", { size: 13, color: "#c5ced6" });
      stage.screenText(right.x, 24, "Latent space. The ring is the unit Gaussian.", { size: 13, color: "#c5ced6" });
    },
  };
}

function createFlow() {
  const data = makeTwoMoons(900, 0.08, mulberry32(12));
  const state = {
    mode: "maps",
    scale: 1,
    shear: 0,
    clock: 0,
    phase: 0,
    holdMaps: false,
    layers: 4,
    showLayer: 4,
    follow: true,
    steps: 0,
    loss: 0,
    round: 0,
  };

  function boot() {
    state.rng = mulberry32(12);
    state.model = new RealNVP(state.layers, 32, state.rng);
    state.buf = new Float32Array(128 * 2);
    state.count = 640;
    state.z = new Float32Array(state.count * 2);
    state.hue = new Float32Array(state.count);
    for (let i = 0; i < state.count; i++) {
      state.z[i * 2] = randn(state.rng);
      state.z[i * 2 + 1] = randn(state.rng);
      state.hue[i] = (Math.atan2(state.z[i * 2 + 1], state.z[i * 2]) / (Math.PI * 2) + 1) % 1;
    }
    state.steps = 0;
    state.loss = 0;
    state.showLayer = state.layers;
    state.follow = true;
    state.gridW = 28;
    state.gridH = 28;
    state.grid = new Float32Array(state.gridW * state.gridH);
    state.pix = new Uint8Array(state.gridW * state.gridH * 4);
    state.row = 0;
    state.lo = -6;
    state.hi = -1;
    state.painted = false;
  }
  boot();

  function square(scale, shear) {
    const pts = [];
    const s = 0.72;
    const edge = 10;
    const map = (x, y) => {
      const xs = x * scale;
      return [xs, y + shear * Math.tanh(xs / 2)];
    };
    for (let i = 0; i <= edge; i++) pts.push(map(-s + (2 * s * i) / edge, -s));
    for (let i = 1; i <= edge; i++) pts.push(map(s, -s + (2 * s * i) / edge));
    for (let i = 1; i <= edge; i++) pts.push(map(s - (2 * s * i) / edge, s));
    for (let i = 1; i < edge; i++) pts.push(map(-s, s - (2 * s * i) / edge));
    return pts;
  }

  function areaOf(pts) {
    let a = 0;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[(i + 1) % pts.length];
      a += x1 * y2 - x2 * y1;
    }
    return Math.abs(a) / 2;
  }

  return {
    id: "flow",
    nav: "Flow",
    title: "Normalising flow",
    state,
    setMode(id) {
      state.mode = id;
      state.clock = 0;
      state.phase = 0;
      state.round = 0;
      if (id === "train") queueTrain(state);
    },
    setControl(id, value) {
      if (id === "layers" && Number(value) !== state.layers) {
        state.layers = Number(value);
        boot();
        return;
      }
      if (id === "scale" || id === "shear") state.holdMaps = true;
      if (id === "showLayer") {
        state.showLayer = value;
        state.follow = false;
      } else state[id] = value;
    },
    action(id) {
      if (id === "burst") queueTrain(state);
      if (id === "reset") boot();
      if (id === "follow") {
        state.follow = true;
        state.showLayer = state.layers;
      }
      if (id === "playmaps") state.holdMaps = false;
    },
    pointer() {},
    update(dt, ctx) {
      if (state.mode === "maps" && ticking(ctx) && !state.holdMaps && !ctx.dragging) {
        state.clock += ctx.paused ? 0.4 : dt;
        const span = 9;
        const u = (state.clock % span) / span;
        if (u < 0.28) {
          state.phase = 0;
          state.scale = 1 + (u / 0.28);
          state.shear = 0;
        } else if (u < 0.4) {
          state.phase = 1;
          state.scale = 2;
          state.shear = 0;
        } else if (u < 0.68) {
          state.phase = 2;
          state.scale = 2;
          state.shear = ((u - 0.4) / 0.28) * 0.9;
        } else if (u < 0.8) {
          state.phase = 3;
          state.scale = 2;
          state.shear = 0.9 * (1 - (u - 0.68) / 0.12);
        } else {
          state.phase = 4;
          state.scale = 2 - (u - 0.8) / 0.2;
          state.shear = 0;
        }
      }
      if (state.mode === "train") {
        const n = consumeTrain(state, dt, ctx, 30);
        for (let s = 0; s < n; s++) {
          const { batch } = sampleBatch(data.points, 128, state.rng, state.buf);
          state.loss = state.model.nll(batch, 128, true);
          state.model.adam(2e-3);
          state.steps += 1;
        }
        if (state.follow) state.showLayer = state.layers;
        if (ticking(ctx) && ctx.width) {
          const area = plateBox(ctx.width, ctx.height);
          const cam = fitBox(area.w, area.h, 0, 0, 3.15);
          state.cam = cam;
          const bounds = { xmin: cam.xmin, xmax: cam.xmax, ymin: cam.ymin, ymax: cam.ymax };
          const rows = 4;
          const row1 = Math.min(state.gridH, state.row + rows);
          state.model.fillLogProb(state.grid, state.gridW, state.gridH, bounds, state.row, row1);
          state.row = row1;
          if (state.row >= state.gridH) {
            state.row = 0;
            const lo = percentile(state.grid, 0.08);
            const hi = percentile(state.grid, 0.98);
            state.lo = state.painted ? state.lo * 0.75 + lo * 0.25 : lo;
            state.hi = state.painted ? state.hi * 0.75 + hi * 0.25 : hi;
            paintDensity(state.grid, state.pix, state.lo, state.hi);
            state.painted = true;
          }
        }
      }
      if (state.mode === "round") {
        state.errClock = (state.errClock || 0) + dt;
        if (state.errClock > 0.6) {
          state.errClock = 0;
          const probeN = 48;
          const { x } = state.model.forward(state.z, probeN);
          const { z } = state.model.inverse(x, probeN);
          let err = 0;
          for (let i = 0; i < probeN * 2; i++) err = Math.max(err, Math.abs(z[i] - state.z[i]));
          state.inverseError = err;
        }
      }
      if (state.mode === "round" && ticking(ctx)) {
        state.clock += ctx.paused ? 0.7 : dt;
        const frames = state.layers * 2;
        if (state.clock > 0.7) {
          state.clock = 0;
          state.round = (state.round + 1) % (frames + 1);
        }
      }
    },
    spec() {
      const det = state.scale;
      const measured = state.mode === "maps" ? areaOf(square(state.scale, state.shear)) : 0;
      const baseArea = state.mode === "maps" ? areaOf(square(1, 0)) : 0;
      let sentence;
      let tryThis;
      let limit;
      if (state.mode === "maps") {
        sentence = state.phase < 2
          ? "Stretching the horizontal axis multiplies area by the scale and divides density by the same factor. The map is still invertible."
          : "The shear bends the cloud. Its Jacobian determinant is 1, so area and density stay where the scale left them.";
        tryThis = "Let it play, or drag the sliders. Every position is an exact map, including the way back.";
        limit = "These two maps are written by hand. They are the picture of a determinant, before any fitting.";
      } else if (state.mode === "train") {
        sentence = "One fixed Gaussian draw is pushed through the couplings. Training bends that map until the draw sits on the moons.";
        tryThis = "Train 40 steps runs a batch of your own. Scrub the coupling index to see the output of that many real layers.";
        limit = "The flow keeps both dimensions. It has an inverse at every step of training, even while the fit is still bad.";
      } else {
        sentence = "Forward couplings carry the base noise toward the moons. The inverse couplings run backwards and recover that same noise.";
        tryThis = "Pause on a coupling and read which layer is on the plate before you continue.";
        limit = "The return error should sit near zero. That is the definition being checked, not a quality score.";
      }
      const controls = [];
      if (state.mode === "maps") {
        controls.push(
          { id: "scale", label: "Horizontal scale", min: 1, max: 2.5, step: 0.01, value: state.scale, digits: 2 },
          { id: "shear", label: "Shear", min: 0, max: 1.2, step: 0.01, value: state.shear, digits: 2 },
        );
      } else if (state.mode === "train") {
        controls.push({ id: "showLayer", label: "Coupling shown", min: 0, max: state.layers, step: 1, value: state.showLayer, digits: 0 });
      }
      const phaseName = ["scaling horizontally", "scale held at 2", "adding shear", "removing shear", "scaling back to 1"][state.phase] || "";
      const readouts = state.mode === "maps"
        ? [
            { label: "Determinant", value: fmt(det, 2) },
            { label: "Density factor", value: fmt(1 / det, 2) },
            { label: "Square area", value: `${fmt(measured, 2)}  (base ${fmt(baseArea, 2)})` },
            { label: "Map", value: phaseName },
          ]
        : [
            ...(state.mode === "train" ? [{ label: "Steps you queued", value: String(state.burst || 0) }] : []),
            { label: "Training steps", value: String(state.steps) },
            { label: "Negative log-likelihood", value: fmt(state.loss, 3) },
            { label: "Layer", value: state.mode === "round" ? roundLabel(state) : `${state.showLayer} of ${state.layers}` },
            ...(state.mode === "round" ? [{ label: "Inverse error", value: state.inverseError == null ? "—" : state.inverseError.toExponential(1) }] : []),
          ];
      return {
        sentence,
        tryThis,
        limit,
        product: null,
        modes: [
          { id: "maps", label: "Scale and shear" },
          { id: "train", label: "Train RealNVP" },
          { id: "round", label: "Round trip" },
        ],
        mode: state.mode,
        controls,
        choices: state.mode === "maps" ? [] : [
          { id: "layers", label: "Coupling layers", value: state.layers, options: [
            { id: 2, label: "2" }, { id: 4, label: "4" }, { id: 6, label: "6" },
          ] },
        ],
        actions: state.mode === "maps"
          ? (state.holdMaps ? [{ id: "playmaps", label: "Play the maps" }] : [])
          : [
              ...(state.mode === "train" ? [TRAIN_ACTION] : []),
              { id: "reset", label: "Reset weights" },
              { id: "follow", label: "Show the full map" },
            ],
        readouts,
      };
    },
    draw(stage) {
      const area = plateBox(stage.cssW, stage.cssH);
      const cam = fitBox(area.w, area.h, 0, 0, 3.15);
      state.cam = cam;
      stage.plot("main", { ...area, ...cam });
      if (state.mode === "maps") {
        const W = 48;
        const H = 48;
        if (!state.mapPix || state.mapPix.length !== W * H * 4) {
          state.mapPix = new Uint8Array(W * H * 4);
          state.mapVals = new Float32Array(W * H);
        }
        const vals = state.mapVals;
        const dest = state.mapPix;
        for (let j = 0; j < H; j++) {
          const y = cam.ymin + ((j + 0.5) / H) * (cam.ymax - cam.ymin);
          for (let i = 0; i < W; i++) {
            const x = cam.xmin + ((i + 0.5) / W) * (cam.xmax - cam.xmin);
            const z0 = x / state.scale;
            const z1 = y - state.shear * Math.tanh(x / 2);
            vals[j * W + i] = -0.5 * (z0 * z0 + z1 * z1 + 2 * Math.log(2 * Math.PI)) - Math.log(state.scale);
          }
        }
        paintDensity(vals, dest, percentile(vals, 0.05), percentile(vals, 0.995));
        stage.field(dest, W, H);
      } else if (state.painted) {
        stage.field(state.pix, state.gridW, state.gridH);
      }
      drawAxes(stage, cam);
      for (let i = 0; i < data.points.length / 2; i += 2) {
        const rgb = data.labels[i] ? GLA : CAD;
        stage.point(data.points[i * 2], data.points[i * 2 + 1], [...rgb, 0.22], 4.5);
      }
      if (state.mode === "maps") {
        const loop = square(state.scale, state.shear);
        stage.ribbon(loop.concat([loop[0]]), 0.035, [0.96, 0.94, 0.91, 0.9]);
        for (let i = 0; i < state.count; i += 1) {
          const x = state.z[i * 2] * state.scale;
          const y = state.z[i * 2 + 1] + state.shear * Math.tanh(x / 2);
          stage.glow(x, y, hsl(state.hue[i], 0.55, 0.66), 5.2, 0.75);
        }
        stage.screenText(area.x, 24, "colour is the base angle, so you can see where each point was carried", { size: 13, color: "#c5ced6" });
      } else {
        const layer = state.mode === "round" ? forwardCount(state) : state.showLayer;
        const { mids } = state.model.forward(state.z, state.count);
        let shown;
        let caption;
        if (state.mode === "round" && state.round > state.layers) {
          const back = state.model.inverseFrames(mids[state.layers], state.count);
          const k = state.round - state.layers;
          shown = back[Math.min(k, back.length - 1)];
          caption = k >= state.layers ? "inverse finished — recovered base noise" : `inverse coupling ${k} of ${state.layers}`;
        } else {
          const idx = Math.max(0, Math.min(layer, mids.length - 1));
          shown = mids[idx];
          caption = idx === 0 ? "base Gaussian, before any coupling" : `after coupling ${idx} of ${state.layers}`;
        }
        for (let i = 0; i < state.count; i++) {
          stage.glow(shown[i * 2], shown[i * 2 + 1], hsl(state.hue[i], 0.55, 0.66), 5, 0.72);
        }
        stage.screenText(area.x, 24, caption, { size: 13, color: "#c5ced6" });
      }
    },
  };
}

function forwardCount(state) {
  if (state.round <= state.layers) return state.round;
  return state.layers;
}

function roundLabel(state) {
  if (state.round === 0) return "base";
  if (state.round <= state.layers) return `forward ${state.round}`;
  if (state.round < state.layers * 2) return `inverse ${state.round - state.layers}`;
  return "recovered base";
}

function energy1(x) {
  const u = x * x - 1;
  return u * u;
}
function grad1(x) {
  return 4 * x * (x * x - 1);
}

function createEBM() {
  const blobs = makeTwoBlobs(500, mulberry32(7));
  const state = {
    mode: "langevin",
    noise: 1,
    eta: 0.02,
    learnNoise: 0.8,
    learnEta: 0.05,
    steps: 0,
    eData: 0,
    eModel: 0,
  };

  function bootParticles() {
    state.rng = mulberry32(20);
    state.n = 28;
    state.x = new Float32Array(state.n);
    state.hue = new Float32Array(state.n);
    for (let i = 0; i < state.n; i++) {
      state.x[i] = -1.7 + (3.4 * i) / (state.n - 1);
      state.hue[i] = i / (state.n - 1);
    }
    state.hist = new Float32Array(48);
    state.visits = [];
  }

  function bootModel() {
    state.modelRng = mulberry32(8);
    state.model = new EnergyModel(state.modelRng, 32);
    state.trainBuf = new Float32Array(64 * 2);
    state.negN = 64;
    state.neg = new Float32Array(state.negN * 2);
    for (let i = 0; i < state.neg.length; i++) state.neg[i] = randn(state.modelRng) * 1.3;
    state.trail = Array.from({ length: state.negN }, () => []);
    state.steps = 0;
    state.gridW = 36;
    state.gridH = 36;
    state.grid = new Float32Array(state.gridW * state.gridH);
    state.pix = new Uint8Array(state.gridW * state.gridH * 4);
    state.row = 0;
    state.painted = false;
    state.bounds = { xmin: -2.7, xmax: 2.7, ymin: -2.1, ymax: 2.1 };
  }
  bootParticles();
  bootModel();

  return {
    id: "ebm",
    nav: "Energy",
    title: "Energy and noise",
    state,
    setMode(id) {
      state.mode = id;
      if (id === "train") queueTrain(state);
    },
    setControl(id, value) { state[id] = value; },
    action(id) {
      if (id === "burst") queueTrain(state);
      if (id === "reseed") bootParticles();
      if (id === "reset") bootModel();
    },
    pointer() {},
    update(dt, ctx) {
      if (!ticking(ctx) && state.mode === "langevin") return;
      if (state.mode === "langevin") {
        const sub = budget(state, dt, ctx, 70);
        for (let s = 0; s < sub; s++) {
          for (let i = 0; i < state.n; i++) {
            const drift = -0.5 * state.eta * grad1(state.x[i]);
            const noise = state.noise * Math.sqrt(state.eta) * randn(state.rng);
            state.x[i] = Math.max(-2.15, Math.min(2.15, state.x[i] + drift + noise));
            state.visits.push(state.x[i]);
          }
        }
        if (state.visits.length > 4000) state.visits.splice(0, state.visits.length - 4000);
        state.hist.fill(0);
        const lo = -2.2;
        const hi = 2.2;
        for (const v of state.visits) {
          const b = Math.floor(((v - lo) / (hi - lo)) * state.hist.length);
          if (b >= 0 && b < state.hist.length) state.hist[b] += 1;
        }
      } else {
        if (ctx.width > 2) {
          const area = plateBox(ctx.width, ctx.height);
          const cam = fitBox(area.w, area.h, 0, 0, 2.55);
          const next = { xmin: cam.xmin, xmax: cam.xmax, ymin: cam.ymin, ymax: cam.ymax };
          if (!state.bounds || Math.abs(next.xmin - state.bounds.xmin) > 0.03 || Math.abs(next.ymin - state.bounds.ymin) > 0.03) {
            state.row = 0;
          }
          state.bounds = next;
        }
        const sub = consumeTrain(state, dt, ctx, 14);
        for (let s = 0; s < sub; s++) {
          for (let k = 0; k < 4; k++) {
            const g = state.model.inputGrad(state.neg, state.negN);
            for (let i = 0; i < state.negN; i++) {
              const gx = clamp(g[i * 2], -8, 8);
              const gy = clamp(g[i * 2 + 1], -8, 8);
              state.neg[i * 2] += -0.5 * state.learnEta * gx + Math.sqrt(state.learnEta) * state.learnNoise * randn(state.modelRng);
              state.neg[i * 2 + 1] += -0.5 * state.learnEta * gy + Math.sqrt(state.learnEta) * state.learnNoise * randn(state.modelRng);
              state.neg[i * 2] = clamp(state.neg[i * 2], -2.6, 2.6);
              state.neg[i * 2 + 1] = clamp(state.neg[i * 2 + 1], -2.0, 2.0);
            }
          }
          for (let i = 0; i < state.negN; i++) {
            const path = state.trail[i];
            path.push([state.neg[i * 2], state.neg[i * 2 + 1]]);
            if (path.length > 16) path.shift();
          }
          const { batch } = sampleBatch(blobs.points, 64, state.modelRng, state.trainBuf);
          const stats = state.model.step(batch, state.neg, state.negN, { lr: 2e-3 });
          state.eData = stats.data;
          state.eModel = stats.model;
          state.steps += 1;
        }
        if (ticking(ctx)) {
          const { xmin, xmax, ymin, ymax } = state.bounds;
          const row1 = Math.min(state.gridH, state.row + 6);
          const batch = (row1 - state.row) * state.gridW;
          const buf = new Float32Array(batch * 2);
          let p = 0;
          for (let j = state.row; j < row1; j++) {
            const y = ymin + ((j + 0.5) / state.gridH) * (ymax - ymin);
            for (let i = 0; i < state.gridW; i++) {
              buf[p++] = xmin + ((i + 0.5) / state.gridW) * (xmax - xmin);
              buf[p++] = y;
            }
          }
          const e = state.model.energy(buf, batch);
          for (let i = 0; i < batch; i++) state.grid[state.row * state.gridW + i] = e[i];
          state.row = row1;
          if (state.row >= state.gridH) {
            state.row = 0;
            paintEnergy(state.grid, state.pix, percentile(state.grid, 0.04), percentile(state.grid, 0.9));
            state.painted = true;
          }
        }
      }
    },
    spec() {
      const quiet = state.noise < 0.05;
      const low = state.noise < 0.55;
      const sentence = state.mode === "train"
        ? "Each step lowers energy on the two data clusters and raises it where the model’s own particles are sitting."
        : quiet
          ? "With the noise off, this is descent on E(x) = (x² − 1)². Every chain stops in the well it can reach downhill."
          : low
            ? "A little noise shakes each chain inside its well. The ridge at zero is still hard to cross."
            : "Drift walks downhill on E(x) = (x² − 1)². Enough noise lets a chain cross the ridge instead of settling forever.";
      const tryThis = state.mode === "train"
        ? "Train 40 steps runs a batch of your own. Turn the particle noise down and the energy rises only along paths the particles still visit."
        : "Move the noise from 0 to 1 and watch which colours ever change wells.";
      const limit = state.mode === "train"
        ? "If the particles never visit a region, training does not raise the energy there. Sampling and the loss are the same story."
        : "Writing down E(x) was immediate. The chains are there because the normaliser is not.";
      const controls = state.mode === "langevin"
        ? [
            { id: "noise", label: "Noise scale", min: 0, max: 1.3, step: 0.01, value: state.noise, digits: 2 },
            { id: "eta", label: "Step size", min: 0.005, max: 0.08, step: 0.001, value: state.eta, digits: 3 },
          ]
        : [
            { id: "learnNoise", label: "Particle noise", min: 0, max: 1.5, step: 0.01, value: state.learnNoise, digits: 2 },
            { id: "learnEta", label: "Langevin step", min: 0.01, max: 0.12, step: 0.005, value: state.learnEta, digits: 3 },
          ];
      const readouts = state.mode === "langevin"
        ? [
            { label: "Chains", value: String(state.n) },
            { label: "Energy at the wells", value: "0" },
            { label: "Energy on the ridge", value: "1" },
          ]
        : [
            { label: "Steps you queued", value: String(state.burst || 0) },
            { label: "Training steps", value: String(state.steps) },
            { label: "Energy on data", value: fmt(state.eData, 2) },
            { label: "Energy on particles", value: fmt(state.eModel, 2) },
          ];
      return {
        sentence,
        tryThis,
        limit,
        product: null,
        modes: [
          { id: "langevin", label: "Two wells" },
          { id: "train", label: "Train an energy" },
        ],
        mode: state.mode,
        controls,
        choices: [],
        actions: state.mode === "langevin"
          ? [{ id: "reseed", label: "Reset chains" }]
          : [TRAIN_ACTION, { id: "reset", label: "Reset energy" }],
        readouts,
      };
    },
    draw(stage) {
      if (state.mode === "langevin") {
        const pad = 16;
        const histH = Math.max(96, stage.cssH * 0.3);
        const curve = { x: pad, y: 32, w: stage.cssW - pad * 2, h: Math.max(40, stage.cssH - histH - 48) };
        const hist = { x: pad, y: curve.y + curve.h + 8, w: curve.w, h: Math.max(40, stage.cssH - (curve.y + curve.h + 8) - 12) };
        const cam = { xmin: -2.35, xmax: 2.35, ymin: -0.25, ymax: 4.15 };
        stage.plot("curve", { ...curve, ...cam });
        const curvePts = [];
        for (let i = 0; i <= 140; i++) {
          const x = cam.xmin + ((cam.xmax - cam.xmin) * i) / 140;
          curvePts.push([x, energy1(x)]);
        }
        stage.ribbon(curvePts, 0.045, [0.93, 0.91, 0.86, 0.9]);
        stage.glow(-1, 0, CAD, 8, 1);
        stage.glow(1, 0, GLA, 8, 1);
        stage.glow(0, 1, CORAL, 7, 0.9);
        for (let i = 0; i < state.n; i++) {
          const x = state.x[i];
          stage.glow(x, energy1(x), hsl(state.hue[i], 0.6, 0.64), 7, 0.9);
        }
        stage.worldText("curve", -1, 0.22, "well", { size: 12, color: "#e39b4a", align: "center", baseline: "bottom" });
        stage.worldText("curve", 1, 0.22, "well", { size: 12, color: "#9fd0dc", align: "center", baseline: "bottom" });
        stage.worldText("curve", 0, 1.08, "ridge", { size: 12, color: "#e07a68", align: "center", baseline: "bottom" });
        let maxBin = 1;
        for (const v of state.hist) maxBin = Math.max(maxBin, v);
        stage.plot("hist", { ...hist, xmin: -2.35, xmax: 2.35, ymin: 0, ymax: maxBin * 1.15 });
        const width = (4.7 / state.hist.length) * 0.84;
        for (let i = 0; i < state.hist.length; i++) {
          const x = -2.2 + (4.4 * i) / state.hist.length;
          const h = state.hist[i];
          stage.quad(x, 0, width, h, [0.62, 0.82, 0.86, 0.85]);
        }
        stage.screenText(pad, 22, "particles sit on E(x). The histogram is where the chains have actually been.", { size: 13, color: "#c5ced6" });
      } else {
        const area = plateBox(stage.cssW, stage.cssH);
        const cam = state.bounds || fitBox(area.w, area.h, 0, 0, 2.55);
        stage.plot("main", { ...area, ...cam });
        if (state.painted) stage.field(state.pix, state.gridW, state.gridH);
        drawAxes(stage, cam);
        for (let i = 0; i < state.negN; i++) {
          const path = state.trail[i];
          for (let k = 1; k < path.length; k++) {
            const a = k / path.length;
            stage.line(path[k - 1][0], path[k - 1][1], path[k][0], path[k][1], [0.9, 0.93, 0.95, 0.15 * a]);
          }
        }
        for (let i = 0; i < blobs.points.length / 2; i += 2) {
          const rgb = blobs.labels[i] ? GLA : CAD;
          stage.point(blobs.points[i * 2], blobs.points[i * 2 + 1], [...rgb, 0.55], 5);
        }
        for (let i = 0; i < state.negN; i++) stage.glow(state.neg[i * 2], state.neg[i * 2 + 1], CHALK, 5.5, 0.85);
        stage.screenText(area.x, 24, "warm valleys are low energy. Chalk points are the model’s particles.", { size: 13, color: "#c5ced6" });
      }
    },
  };
}

function createDiffusion() {
  const data = makeTwoMoons(600, 0.08, mulberry32(9));
  const schedule = diffusionSchedule(40, 1e-3, 0.2);
  const state = {
    mode: "forward",
    t: 0,
    clock: 0,
    steps: 0,
    mse: 1,
    w: 1.5,
    label: 0,
    hold: 0,
  };

  function boot() {
    state.rng = mulberry32(9);
    state.model = new DiffusionModel(state.rng, schedule, 40);
    state.buf = new Float32Array(80 * 2);
    state.lab = new Float32Array(80 * 2);
    state.steps = 0;
    state.mse = 1;
    state.showN = 14;
    state.showX0 = new Float32Array(14 * 2);
    for (let i = 0; i < 7; i++) {
      state.showX0[i * 2] = data.points[i * 2];
      state.showX0[i * 2 + 1] = data.points[i * 2 + 1];
      const j = 320 + i;
      state.showX0[(7 + i) * 2] = data.points[j * 2];
      state.showX0[(7 + i) * 2 + 1] = data.points[j * 2 + 1];
    }
    state.showEps = new Float32Array(14 * 2);
    const epsRng = mulberry32(44);
    for (let i = 0; i < state.showEps.length; i++) state.showEps[i] = randn(epsRng);
    state.showT = 6;
    const n = 160;
    state.n = n;
    state.x0 = new Float32Array(n * 2);
    state.eps = new Float32Array(n * 2);
    state.labels = new Uint8Array(n);
    const src = mulberry32(3);
    for (let i = 0; i < n; i++) {
      const j = (src() * (data.points.length / 2)) | 0;
      state.x0[i * 2] = data.points[j * 2];
      state.x0[i * 2 + 1] = data.points[j * 2 + 1];
      state.labels[i] = data.labels[j];
      state.eps[i * 2] = randn(src);
      state.eps[i * 2 + 1] = randn(src);
    }
    resetChain();
    resetGuidance();
  }

  function resetChain() {
    state.rev = new Float32Array(state.n * 2);
    state.revHue = new Float32Array(state.n);
    for (let i = 0; i < state.n; i++) {
      state.rev[i * 2] = randn(state.rng);
      state.rev[i * 2 + 1] = randn(state.rng);
      state.revHue[i] = (Math.atan2(state.rev[i * 2 + 1], state.rev[i * 2]) / (Math.PI * 2) + 1) % 1;
    }
    state.revT = schedule.steps - 1;
    state.hold = 0;
  }

  function resetGuidance() {
    const n = 48;
    state.gN = n;
    state.xT = new Float32Array(n * 2);
    state.zBank = [];
    const rng = mulberry32(15);
    for (let i = 0; i < n * 2; i++) state.xT[i] = randn(rng);
    for (let t = 0; t < schedule.steps; t++) {
      const z = new Float32Array(n * 2);
      if (t > 0) for (let i = 0; i < z.length; i++) z[i] = randn(rng);
      state.zBank[t] = z;
    }
    state.frame = 0;
    state.dirty = true;
    state.frames = [];
  }

  function teach(n) {
    for (let s = 0; s < n; s++) {
      const { batch, index } = sampleBatch(data.points, 80, state.rng, state.buf);
      for (let i = 0; i < 80; i++) {
        const lab = data.labels[index[i]];
        state.lab[i * 2] = lab === 0 ? 1 : 0;
        state.lab[i * 2 + 1] = lab === 1 ? 1 : 0;
      }
      const stats = state.model.step(batch, state.lab, 80, state.rng, { lr: 2.2e-3 });
      state.mse = stats.mse;
      state.steps += 1;
      state.dirty = true;
    }
  }

  function rebuildGuidance() {
    const label = state.label === 0 ? [1, 0] : [0, 1];
    let x = state.xT.slice();
    const frames = [x.slice()];
    for (let t = schedule.steps - 1; t >= 0; t--) {
      x = state.model.reverseStep(x, t, state.zBank[t], label, state.w);
      frames.push(x.slice());
    }
    state.frames = frames;
    state.dirty = false;
  }

  boot();

  return {
    id: "diff",
    nav: "Diffusion",
    title: "Forward, then reverse",
    state,
    setMode(id) {
      state.mode = id;
      if (id === "train") queueTrain(state);
    },
    setControl(id, value) {
      if (id === "w" || id === "label") state.dirty = true;
      state[id] = id === "label" ? Number(value) : value;
      if (id === "label") state.dirty = true;
    },
    action(id) {
      if (id === "burst") queueTrain(state);
      if (id === "reset") boot();
      if (id === "newchain") resetChain();
      if (id === "newnoise") resetGuidance();
    },
    pointer() {},
    update(dt, ctx) {
      if (state.mode === "forward" && ticking(ctx) && !ctx.dragging) {
        state.clock += ctx.paused ? 1 : dt * 7;
        if (state.clock >= 1) {
          state.clock = 0;
          state.t += state.dir || 1;
          if (state.t >= schedule.steps - 1) { state.t = schedule.steps - 1; state.dir = -1; }
          if (state.t <= 0) { state.t = 0; state.dir = 1; }
        }
      }
      if (state.mode === "train") teach(consumeTrain(state, dt, ctx, 28));
      if (state.mode === "reverse" && ticking(ctx)) {
        if (state.revT >= 0) {
          state.clock += ctx.paused ? 1 : dt;
          if (state.clock > 0.07) {
            state.clock = 0;
            const z = new Float32Array(state.rev.length);
            if (state.revT > 0) for (let i = 0; i < z.length; i++) z[i] = randn(state.rng);
            const next = state.model.reverseStep(state.rev, state.revT, z, null, 0);
            state.rev = next;
            state.revT -= 1;
          }
        } else {
          state.hold += ctx.paused ? 0.4 : dt;
          if (state.hold > 1.3) resetChain();
        }
      }
      if (state.mode === "guide") {
        if (state.dirty) {
          state.guideWait = (state.guideWait || 0) + dt;
          if (!ctx.dragging || state.guideWait > 0.2 || !state.frames.length) {
            rebuildGuidance();
            state.guideWait = 0;
          }
        }
        if (ticking(ctx) && state.frames.length) {
          state.clock += ctx.paused ? 1 : dt;
          if (state.clock > 0.06) {
            state.clock = 0;
            state.frame += 1;
            if (state.frame >= state.frames.length) state.frame = 0;
          }
        }
      }
    },
    spec() {
      const ab = schedule.alphaBars[Math.max(0, Math.min(schedule.steps - 1, state.t | 0))];
      const copy = {
        forward: {
          sentence: "Any noise level is one draw. The same ε lets you watch the moons fade into a Gaussian without stepping through the times in between.",
          tryThis: "Scrub time. Early on, the colour of each moon is still visible. Late, the cloud is just the noise.",
          limit: "This is the known forward process. Generation is the reverse, and it needs the predictor.",
        },
        train: {
          sentence: "The network is shown a noisy point and a time. Its target is the noise that was added, and the pale segment is the clean point that noise implies.",
          tryThis: "Train the predictor, or Train 40 steps, runs a batch even while Play is off. The coral segment should move toward the pale one.",
          limit: "A falling noise error says the predictor is learning the objective. Samples come from the reverse mode.",
        },
        reverse: {
          sentence: "The chain starts at Gaussian noise. Each step removes part of the predicted noise and restores the small randomness the reverse transition asks for.",
          tryThis: "Train until the noise error drops, then generate again from a new noise draw.",
          limit: "You are watching one trajectory. It illustrates the update; it does not measure diversity.",
        },
        guide: {
          sentence: "The same noise is denoised again at every weight. Guidance follows the gap between the class-conditional prediction and the unconditional one.",
          tryThis: "Move the weight. Zero ignores the requested moon. Larger weights pull this fixed noise toward that moon.",
          limit: "A large weight follows the class more tightly and can exaggerate the errors of this small network.",
        },
      }[state.mode];
      const controls = [];
      if (state.mode === "forward") {
        controls.push({ id: "t", label: "Time", min: 0, max: schedule.steps - 1, step: 1, value: state.t, digits: 0 });
      }
      if (state.mode === "guide") {
        controls.push({ id: "w", label: "Guidance weight", min: 0, max: 3, step: 0.1, value: state.w, digits: 1 });
      }
      const readouts = [
        ...(state.mode === "train" ? [{ label: "Steps you queued", value: String(state.burst || 0) }] : []),
        { label: "Predictor steps", value: String(state.steps) },
        { label: "Noise error", value: fmt(state.mse, 3) },
      ];
      if (state.mode === "forward") {
        readouts.unshift({ label: "√ᾱ  signal", value: fmt(Math.sqrt(ab), 2) });
        readouts.unshift({ label: "√(1−ᾱ)  noise", value: fmt(Math.sqrt(1 - ab), 2) });
      }
      if (state.mode === "reverse") {
        readouts.unshift({ label: "Reverse time", value: state.revT < 0 ? "0" : String(state.revT) });
      }
      if (state.mode === "guide") {
        readouts.unshift({ label: "ε", value: "ε∅ + w (εc − ε∅)" });
      }
      return {
        ...copy,
        product: null,
        modes: [
          { id: "forward", label: "Forward" },
          { id: "train", label: "Train the predictor" },
          { id: "reverse", label: "Reverse" },
          { id: "guide", label: "Guidance" },
        ],
        mode: state.mode,
        controls,
        choices: state.mode === "guide" ? [
          { id: "label", label: "Requested moon", value: state.label, options: [
            { id: 0, label: "Moon A" }, { id: 1, label: "Moon B" },
          ] },
        ] : [],
        actions: state.mode === "train"
          ? [TRAIN_ACTION, { id: "reset", label: "Reset predictor" }]
          : state.mode === "reverse"
            ? [{ id: "newchain", label: "New noise" }, { id: "reset", label: "Reset predictor" }]
            : state.mode === "guide"
              ? [{ id: "newnoise", label: "New shared noise" }, { id: "reset", label: "Reset predictor" }]
              : [{ id: "reset", label: "Reset predictor" }],
        readouts,
      };
    },
    draw(stage) {
      const area = full(stage, 34);
      const cam = fitBox(area.w, area.h, 0, 0.15, 3.2);
      stage.plot("main", { ...area, ...cam });
      drawAxes(stage, cam);
      const tNow = Math.max(0, Math.min(schedule.steps - 1, state.t | 0));
      if (state.mode === "forward") {
        const xt = forwardDiffuse(state.x0, state.eps, tNow, schedule);
        for (let i = 0; i < state.n; i++) {
          const rgb = state.labels[i] ? GLA : CAD;
          stage.glow(xt[i * 2], xt[i * 2 + 1], rgb, 5.5, 0.8);
        }
        stage.screenText(area.x, 24, `Same ε at every time. t = ${tNow}`, { size: 13, color: "#c5ced6" });
      } else if (state.mode === "train") {
        for (let i = 0; i < data.points.length / 2; i += 2) {
          const rgb = data.labels[i] ? GLA : CAD;
          stage.point(data.points[i * 2], data.points[i * 2 + 1], [...rgb, 0.16], 4);
        }
        const subset = state.showN;
        const x0 = state.showX0;
        const xt = forwardDiffuse(x0, state.showEps, state.showT, schedule);
        const tArray = new Int32Array(subset);
        tArray.fill(state.showT);
        const pred = state.model.predict(xt, tArray, null, subset);
        const ab = schedule.alphaBars[tArray[0]];
        const s0 = Math.sqrt(ab);
        const s1 = Math.sqrt(1 - ab);
        for (let i = 0; i < subset; i++) {
          const xh = (xt[i * 2] - s1 * pred[i * 2]) / s0;
          const yh = (xt[i * 2 + 1] - s1 * pred[i * 2 + 1]) / s0;
          stage.line(xt[i * 2], xt[i * 2 + 1], x0[i * 2], x0[i * 2 + 1], [0.82, 0.86, 0.9, 0.8]);
          stage.line(xt[i * 2], xt[i * 2 + 1], xh, yh, [0.88, 0.48, 0.41, 0.9]);
          stage.glow(xt[i * 2], xt[i * 2 + 1], CHALK, 7, 0.95);
          stage.glow(x0[i * 2], x0[i * 2 + 1], CAD, 6, 0.85);
        }
        stage.screenText(area.x, 24, "Pale segment: true clean point. Coral: where the network thinks it is.", { size: 13, color: "#c5ced6" });
      } else if (state.mode === "reverse") {
        for (let i = 0; i < data.points.length / 2; i += 3) {
          stage.point(data.points[i * 2], data.points[i * 2 + 1], [0.8, 0.84, 0.88, 0.12], 4);
        }
        for (let i = 0; i < state.n; i++) {
          stage.glow(state.rev[i * 2], state.rev[i * 2 + 1], hsl(state.revHue[i], 0.58, 0.66), 5.4, 0.82);
        }
        const label = state.revT < 0 ? "arrived at t = 0" : `reverse step, t = ${state.revT}`;
        stage.screenText(area.x, 24, label, { size: 13, color: "#c5ced6" });
      } else {
        const frame = state.frames[Math.min(state.frame, Math.max(0, state.frames.length - 1))];
        for (let i = 0; i < data.points.length / 2; i += 3) {
          const rgb = data.labels[i] ? GLA : CAD;
          stage.point(data.points[i * 2], data.points[i * 2 + 1], [...rgb, 0.13], 4);
        }
        if (frame) {
          const rgb = state.label === 0 ? CAD : GLA;
          for (let i = 0; i < state.gN; i++) stage.glow(frame[i * 2], frame[i * 2 + 1], rgb, 6, 0.88);
        }
        const shownT = Math.max(0, schedule.steps - 1 - state.frame);
        stage.screenText(area.x, 24, `Shared noise, weight ${state.w.toFixed(1)}, t = ${shownT}`, { size: 13, color: "#c5ced6" });
      }
    },
  };
}

export function createScenes() {
  return [createAR(), createVAE(), createFlow(), createEBM(), createDiffusion()];
}
