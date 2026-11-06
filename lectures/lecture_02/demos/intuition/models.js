/** Small models for the Lecture 2 intuition instrument. Training runs in the page. */

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng) {
  let u = rng();
  if (u < 1e-12) u = 1e-12;
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function silu(x) {
  const s = 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));
  return x * s;
}

function siluPrime(x) {
  const s = 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));
  return s + x * s * (1 - s);
}

export class MLP {
  constructor(sizes, rng, activation = "silu", outScale = 1) {
    this.sizes = sizes;
    this.activation = activation;
    this.layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l];
      const nOut = sizes[l + 1];
      const scale = (activation === "tanh" ? 1 : Math.SQRT2) / Math.sqrt(nIn);
      const last = l === sizes.length - 2;
      const gain = last ? outScale : 1;
      const W = new Float32Array(nOut * nIn);
      const b = new Float32Array(nOut);
      for (let i = 0; i < W.length; i++) W[i] = randn(rng) * scale * gain;
      this.layers.push({
        W,
        b,
        dW: new Float32Array(W.length),
        db: new Float32Array(nOut),
        mW: new Float32Array(W.length),
        vW: new Float32Array(W.length),
        mB: new Float32Array(nOut),
        vB: new Float32Array(nOut),
      });
    }
    this.t = 0;
  }

  zeroGrads() {
    for (const layer of this.layers) {
      layer.dW.fill(0);
      layer.db.fill(0);
    }
  }

  forward(x, batch) {
    this._batch = batch;
    this._acts = [x];
    this._pres = [];
    let h = x;
    for (let l = 0; l < this.layers.length; l++) {
      const { W, b } = this.layers[l];
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const pre = new Float32Array(batch * nOut);
      for (let i = 0; i < batch; i++) {
        const base = i * nIn;
        const dest = i * nOut;
        for (let o = 0; o < nOut; o++) {
          let s = b[o];
          const row = o * nIn;
          for (let k = 0; k < nIn; k++) s += W[row + k] * h[base + k];
          pre[dest + o] = s;
        }
      }
      this._pres.push(pre);
      const last = l === this.layers.length - 1;
      if (last) {
        this._acts.push(pre);
        h = pre;
      } else if (this.activation === "tanh") {
        const act = new Float32Array(pre.length);
        for (let i = 0; i < pre.length; i++) act[i] = Math.tanh(pre[i]);
        this._acts.push(act);
        h = act;
      } else {
        const act = new Float32Array(pre.length);
        for (let i = 0; i < pre.length; i++) act[i] = silu(pre[i]);
        this._acts.push(act);
        h = act;
      }
    }
    return h;
  }

  /** dy is dL/dY summed over the batch. Returns dL/dX. Parameter gradients accumulate. */
  backward(dy, params = true) {
    const batch = this._batch;
    let g = Float32Array.from(dy);
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const layer = this.layers[l];
      if (l < this.layers.length - 1) {
        if (this.activation === "tanh") {
          const act = this._acts[l + 1];
          for (let i = 0; i < g.length; i++) g[i] *= 1 - act[i] * act[i];
        } else {
          const pre = this._pres[l];
          for (let i = 0; i < g.length; i++) g[i] *= siluPrime(pre[i]);
        }
      }
      const prev = this._acts[l];
      const dx = new Float32Array(batch * nIn);
      for (let i = 0; i < batch; i++) {
        const inBase = i * nIn;
        const outBase = i * nOut;
        for (let o = 0; o < nOut; o++) {
          const go = g[outBase + o];
          if (params) layer.db[o] += go;
          const row = o * nIn;
          for (let k = 0; k < nIn; k++) {
            const prevVal = prev[inBase + k];
            if (params) layer.dW[row + k] += go * prevVal;
            dx[inBase + k] += go * layer.W[row + k];
          }
        }
      }
      g = dx;
    }
    return g;
  }

  adam(lr, weightDecay = 0) {
    this.t += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - b1 ** this.t;
    const bc2 = 1 - b2 ** this.t;
    const batch = this._batch || 1;
    for (const layer of this.layers) {
      for (let i = 0; i < layer.W.length; i++) {
        const grad = layer.dW[i] / batch + weightDecay * layer.W[i];
        layer.mW[i] = b1 * layer.mW[i] + (1 - b1) * grad;
        layer.vW[i] = b2 * layer.vW[i] + (1 - b2) * grad * grad;
        layer.W[i] -= lr * (layer.mW[i] / bc1) / (Math.sqrt(layer.vW[i] / bc2) + eps);
      }
      for (let i = 0; i < layer.b.length; i++) {
        const grad = layer.db[i] / batch;
        layer.mB[i] = b1 * layer.mB[i] + (1 - b1) * grad;
        layer.vB[i] = b2 * layer.vB[i] + (1 - b2) * grad * grad;
        layer.b[i] -= lr * (layer.mB[i] / bc1) / (Math.sqrt(layer.vB[i] / bc2) + eps);
      }
    }
  }
}

export function makeTwoMoons(count, noise, rng) {
  const points = new Float32Array(count * 2);
  const labels = new Uint8Array(count);
  const first = count >> 1;
  for (let i = 0; i < count; i++) {
    const angle = rng() * Math.PI;
    let x;
    let y;
    if (i < first) {
      x = Math.cos(angle);
      y = Math.sin(angle);
      labels[i] = 0;
    } else {
      x = 1 - Math.cos(angle);
      y = 0.45 - Math.sin(angle);
      labels[i] = 1;
    }
    points[i * 2] = x + randn(rng) * noise;
    points[i * 2 + 1] = y + randn(rng) * noise;
  }
  let mx = 0;
  let my = 0;
  for (let i = 0; i < count; i++) {
    mx += points[i * 2];
    my += points[i * 2 + 1];
  }
  mx /= count;
  my /= count;
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < count; i++) {
    const dx = points[i * 2] - mx;
    const dy = points[i * 2 + 1] - my;
    vx += dx * dx;
    vy += dy * dy;
  }
  const sx = Math.sqrt(vx / count) || 1;
  const sy = Math.sqrt(vy / count) || 1;
  for (let i = 0; i < count; i++) {
    points[i * 2] = (points[i * 2] - mx) / sx;
    points[i * 2 + 1] = (points[i * 2 + 1] - my) / sy;
  }
  return { points, labels };
}

export function makeTwoBlobs(count, rng, separation = 1.15, sigma = 0.18) {
  const points = new Float32Array(count * 2);
  const labels = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const right = i >= count / 2;
    labels[i] = right ? 1 : 0;
    points[i * 2] = (right ? separation : -separation) + randn(rng) * sigma;
    points[i * 2 + 1] = randn(rng) * sigma;
  }
  return { points, labels };
}

export function sampleBatch(points, batch, rng, out) {
  const count = points.length / 2;
  const dest = out || new Float32Array(batch * 2);
  const index = new Int32Array(batch);
  for (let i = 0; i < batch; i++) {
    const j = (rng() * count) | 0;
    index[i] = j;
    dest[i * 2] = points[j * 2];
    dest[i * 2 + 1] = points[j * 2 + 1];
  }
  return { batch: dest, index };
}

const LOG_2PI = Math.log(2 * Math.PI);

function couplingActivate(raw, mask, batch) {
  const active0 = 1 - mask[0];
  const active1 = 1 - mask[1];
  const logS = new Float32Array(batch * 2);
  const shift = new Float32Array(batch * 2);
  for (let i = 0; i < batch; i++) {
    const actives = [active0, active1];
    for (let k = 0; k < 2; k++) {
      const th = Math.tanh(raw[i * 4 + k]);
      logS[i * 2 + k] = 1.5 * th * actives[k];
      shift[i * 2 + k] = raw[i * 4 + 2 + k] * actives[k];
    }
  }
  return { logS, shift };
}

export class RealNVP {
  constructor(layers, hidden, rng) {
    this.layers = [];
    for (let i = 0; i < layers; i++) {
      const mask = i % 2 === 0 ? new Float32Array([1, 0]) : new Float32Array([0, 1]);
      this.layers.push({
        mask,
        net: new MLP([2, hidden, hidden, 4], rng, "tanh", 0.05),
      });
    }
    this._batchBuf = null;
  }

  _inverseLayer(layer, y, batch) {
    const masked = new Float32Array(batch * 2);
    for (let i = 0; i < batch; i++) {
      masked[i * 2] = y[i * 2] * layer.mask[0];
      masked[i * 2 + 1] = y[i * 2 + 1] * layer.mask[1];
    }
    const raw = layer.net.forward(masked, batch);
    const { logS, shift } = couplingActivate(raw, layer.mask, batch);
    const x = new Float32Array(batch * 2);
    let sumS = 0;
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < 2; k++) {
        const active = 1 - layer.mask[k];
        if (active === 0) x[i * 2 + k] = y[i * 2 + k];
        else x[i * 2 + k] = (y[i * 2 + k] - shift[i * 2 + k]) * Math.exp(-logS[i * 2 + k]);
        sumS += logS[i * 2 + k];
      }
    }
    return { x, y, raw, logS, shift, masked, sumS };
  }

  inverse(x, batch) {
    let y = x;
    const caches = [];
    let totalS = 0;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const cache = this._inverseLayer(this.layers[l], y, batch);
      caches.push({ layerIndex: l, cache });
      totalS += cache.sumS;
      y = cache.x;
    }
    return { z: y, totalS, caches };
  }

  forward(z, batch) {
    let x = z;
    const mids = [z];
    let logDet = new Float32Array(batch);
    for (const layer of this.layers) {
      const masked = new Float32Array(batch * 2);
      for (let i = 0; i < batch; i++) {
        masked[i * 2] = x[i * 2] * layer.mask[0];
        masked[i * 2 + 1] = x[i * 2 + 1] * layer.mask[1];
      }
      const raw = layer.net.forward(masked, batch);
      const { logS, shift } = couplingActivate(raw, layer.mask, batch);
      const y = new Float32Array(batch * 2);
      for (let i = 0; i < batch; i++) {
        let s = 0;
        for (let k = 0; k < 2; k++) {
          const active = 1 - layer.mask[k];
          if (active === 0) y[i * 2 + k] = x[i * 2 + k];
          else y[i * 2 + k] = x[i * 2 + k] * Math.exp(logS[i * 2 + k]) + shift[i * 2 + k];
          s += logS[i * 2 + k];
        }
        logDet[i] += s;
      }
      x = y;
      mids.push(x);
    }
    return { x, logDet, mids };
  }

  nll(x, batch, backward = false) {
    if (backward) for (const layer of this.layers) layer.net.zeroGrads();
    const { z, caches } = this.inverse(x, batch);
    let sum = 0;
    for (let i = 0; i < batch; i++) {
      sum += 0.5 * (z[i * 2] ** 2 + z[i * 2 + 1] ** 2 + 2 * LOG_2PI);
    }
    // inverse log-det is -totalS, and loss = -log p = 0.5||z||^2 + const + totalS
    let totalS = 0;
    for (const entry of caches) totalS += entry.cache.sumS;
    sum += totalS;
    if (backward) {
      let g = Float32Array.from(z);
      for (let c = caches.length - 1; c >= 0; c--) {
        g = this._inverseBackward(this.layers[caches[c].layerIndex], caches[c].cache, g, batch);
      }
    }
    return sum / batch;
  }

  _inverseBackward(layer, cache, gX, batch) {
    const { y, raw, logS, shift, x } = cache;
    const gY = new Float32Array(batch * 2);
    const gRaw = new Float32Array(batch * 4);
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < 2; k++) {
        const active = 1 - layer.mask[k];
        if (active === 0) {
          gY[i * 2 + k] += gX[i * 2 + k];
        } else {
          const e = Math.exp(-logS[i * 2 + k]);
          const gx = gX[i * 2 + k];
          const xb = x[i * 2 + k];
          gY[i * 2 + k] += gx * e;
          const gShift = gx * -e;
          const gLogS = -gx * xb + 1;
          const th = Math.tanh(raw[i * 4 + k]);
          gRaw[i * 4 + k] += gLogS * (1.5 * (1 - th * th));
          gRaw[i * 4 + 2 + k] += gShift;
        }
      }
    }
    const gMasked = layer.net.backward(gRaw, true);
    for (let i = 0; i < batch; i++) {
      gY[i * 2] += gMasked[i * 2] * layer.mask[0];
      gY[i * 2 + 1] += gMasked[i * 2 + 1] * layer.mask[1];
    }
    return gY;
  }

  adam(lr) {
    for (const layer of this.layers) layer.net.adam(lr, 1e-5);
  }

  inverseFrames(x, batch) {
    let y = x;
    const frames = [x];
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const cache = this._inverseLayer(this.layers[l], y, batch);
      y = cache.x;
      frames.push(y);
    }
    return frames;
  }

  fillLogProb(vals, width, height, bounds, row0, row1) {
    const { xmin, xmax, ymin, ymax } = bounds;
    const rows = row1 - row0;
    const batch = rows * width;
    if (batch <= 0) return;
    const buf = new Float32Array(batch * 2);
    let p = 0;
    for (let j = row0; j < row1; j++) {
      const y = ymin + ((j + 0.5) / height) * (ymax - ymin);
      for (let i = 0; i < width; i++) {
        buf[p++] = xmin + ((i + 0.5) / width) * (xmax - xmin);
        buf[p++] = y;
      }
    }
    const { z, caches } = this.inverse(buf, batch);
    for (let i = 0; i < batch; i++) {
      let pointS = 0;
      for (const entry of caches) pointS += entry.cache.logS[i * 2] + entry.cache.logS[i * 2 + 1];
      const nll = 0.5 * (z[i * 2] ** 2 + z[i * 2 + 1] ** 2 + 2 * LOG_2PI) + pointS;
      vals[row0 * width + i] = -nll;
    }
  }

  logProbGrid(xmin, xmax, ymin, ymax, width, height) {
    const vals = new Float32Array(width * height);
    const chunk = 64;
    const buf = new Float32Array(chunk * 2);
    let cursor = 0;
    const flush = (n) => {
      if (!n) return;
      const { z, caches } = this.inverse(buf, n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        // recompute sum of log-scales from a dedicated pass: inverse already folded into z
        // Use stored caches sum per point.
        let pointS = 0;
        for (const entry of caches) {
          pointS += entry.cache.logS[i * 2] + entry.cache.logS[i * 2 + 1];
        }
        s = 0.5 * (z[i * 2] ** 2 + z[i * 2 + 1] ** 2 + 2 * LOG_2PI) + pointS;
        vals[cursor + i] = -s;
      }
      cursor += n;
    };
    let filled = 0;
    for (let j = 0; j < height; j++) {
      const y = ymin + ((j + 0.5) / height) * (ymax - ymin);
      for (let i = 0; i < width; i++) {
        const x = xmin + ((i + 0.5) / width) * (xmax - xmin);
        buf[filled * 2] = x;
        buf[filled * 2 + 1] = y;
        filled += 1;
        if (filled === chunk) {
          flush(filled);
          filled = 0;
        }
      }
    }
    flush(filled);
    return vals;
  }
}

export class VAE2D {
  constructor(rng, hidden = 32) {
    this.enc = new MLP([2, hidden, hidden, 4], rng, "silu", 0.2);
    this.dec = new MLP([2, hidden, hidden, 2], rng, "silu", 0.2);
  }

  encode(x, batch) {
    const h = this.enc.forward(x, batch);
    const mu = new Float32Array(batch * 2);
    const logvar = new Float32Array(batch * 2);
    const draw = new Float32Array(batch * 4);
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < 2; k++) {
        const raw = h[i * 4 + 2 + k];
        const th = Math.tanh(raw / 3);
        mu[i * 2 + k] = h[i * 4 + k];
        logvar[i * 2 + k] = 3 * th;
        draw[i * 4 + k] = 1;
        draw[i * 4 + 2 + k] = 1 - th * th;
      }
    }
    return { mu, logvar, draw };
  }

  decode(z, batch) {
    return this.dec.forward(z, batch);
  }

  step(x, batch, rng, { lr = 2e-3, beta = 1 } = {}) {
    this.enc.zeroGrads();
    this.dec.zeroGrads();
    const { mu, logvar, draw } = this.encode(x, batch);
    const eps = new Float32Array(batch * 2);
    const z = new Float32Array(batch * 2);
    for (let i = 0; i < batch * 2; i++) {
      eps[i] = randn(rng);
      const std = Math.exp(0.5 * logvar[i]);
      z[i] = mu[i] + std * eps[i];
    }
    const recon = this.decode(z, batch);
    const gRecon = new Float32Array(batch * 2);
    let rec = 0;
    let kl = 0;
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < 2; k++) {
        const d = recon[i * 2 + k] - x[i * 2 + k];
        rec += d * d;
        gRecon[i * 2 + k] = 2 * d;
        const lv = logvar[i * 2 + k];
        const m = mu[i * 2 + k];
        kl += -0.5 * (1 + lv - m * m - Math.exp(lv));
      }
    }
    const gZ = this.dec.backward(gRecon, true);
    const gH = new Float32Array(batch * 4);
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < 2; k++) {
        const lv = logvar[i * 2 + k];
        const m = mu[i * 2 + k];
        const std = Math.exp(0.5 * lv);
        const gz = gZ[i * 2 + k];
        const gMu = gz + beta * m;
        const gLv = gz * eps[i * 2 + k] * 0.5 * std + beta * (-0.5 + 0.5 * Math.exp(lv));
        gH[i * 4 + k] = gMu * draw[i * 4 + k];
        gH[i * 4 + 2 + k] = gLv * draw[i * 4 + 2 + k];
      }
    }
    this.enc.backward(gH, true);
    this.enc.adam(lr, 1e-5);
    this.dec.adam(lr, 1e-5);
    return { recon: rec / batch, kl: kl / batch, elbo: -(rec + beta * kl) / batch };
  }
}

export class EnergyModel {
  constructor(rng, hidden = 32) {
    this.net = new MLP([2, hidden, hidden, 1], rng, "silu", 0.3);
  }

  energy(x, batch) {
    return this.net.forward(x, batch);
  }

  step(data, negatives, batch, { lr = 1.5e-3 } = {}) {
    this.net.zeroGrads();
    const eData = this.energy(data, batch);
    const gPos = new Float32Array(batch);
    let pos = 0;
    for (let i = 0; i < batch; i++) {
      pos += eData[i];
      gPos[i] = 1 + 0.02 * eData[i];
    }
    this.net.backward(gPos, true);
    const eNeg = this.energy(negatives, batch);
    const gNeg = new Float32Array(batch);
    let neg = 0;
    for (let i = 0; i < batch; i++) {
      neg += eNeg[i];
      gNeg[i] = -1 + 0.02 * eNeg[i];
    }
    this.net.backward(gNeg, true);
    this.net.adam(lr, 2e-4);
    return { data: pos / batch, model: neg / batch, gap: (neg - pos) / batch };
  }

  inputGrad(x, batch) {
    this.energy(x, batch);
    const upstream = new Float32Array(batch);
    upstream.fill(1);
    return this.net.backward(upstream, false);
  }
}

export function diffusionSchedule(steps = 40, beta0 = 1e-3, beta1 = 0.09) {
  const betas = new Float32Array(steps);
  const alphas = new Float32Array(steps);
  const alphaBars = new Float32Array(steps);
  let ab = 1;
  for (let t = 0; t < steps; t++) {
    const u = steps === 1 ? 1 : t / (steps - 1);
    betas[t] = beta0 + (beta1 - beta0) * u;
    alphas[t] = 1 - betas[t];
    ab *= alphas[t];
    alphaBars[t] = ab;
  }
  return { steps, betas, alphas, alphaBars };
}

export function forwardDiffuse(x0, eps, t, schedule) {
  const ab = schedule.alphaBars[t];
  const s0 = Math.sqrt(ab);
  const s1 = Math.sqrt(1 - ab);
  const out = new Float32Array(x0.length);
  for (let i = 0; i < x0.length; i++) out[i] = s0 * x0[i] + s1 * eps[i];
  return out;
}

function writeCondInput(dest, offset, x, y, t, steps, label) {
  const u = steps <= 1 ? 0 : t / (steps - 1);
  dest[offset] = x;
  dest[offset + 1] = y;
  dest[offset + 2] = Math.sin(u * Math.PI);
  dest[offset + 3] = Math.cos(u * Math.PI);
  dest[offset + 4] = Math.sin(2 * u * Math.PI);
  dest[offset + 5] = Math.cos(2 * u * Math.PI);
  dest[offset + 6] = label ? label[0] : 0;
  dest[offset + 7] = label ? label[1] : 0;
}

export class DiffusionModel {
  constructor(rng, schedule, hidden = 48) {
    this.schedule = schedule;
    this.net = new MLP([8, hidden, hidden, 2], rng, "silu", 0.1);
    this._in = null;
  }

  _inputs(x, tArray, labels, batch) {
    const input = new Float32Array(batch * 8);
    for (let i = 0; i < batch; i++) {
      const label = labels ? [labels[i * 2], labels[i * 2 + 1]] : null;
      writeCondInput(input, i * 8, x[i * 2], x[i * 2 + 1], tArray[i], this.schedule.steps, label);
    }
    return input;
  }

  predict(x, tArray, labels, batch) {
    return this.net.forward(this._inputs(x, tArray, labels, batch), batch);
  }

  step(x0, labels, batch, rng, { lr = 2e-3, drop = 0.18 } = {}) {
    this.net.zeroGrads();
    const steps = this.schedule.steps;
    const tArray = new Int32Array(batch);
    const eps = new Float32Array(batch * 2);
    const xt = new Float32Array(batch * 2);
    const used = new Float32Array(batch * 2);
    for (let i = 0; i < batch; i++) {
      const t = (rng() * steps) | 0;
      tArray[i] = t;
      const ab = this.schedule.alphaBars[t];
      const s0 = Math.sqrt(ab);
      const s1 = Math.sqrt(1 - ab);
      eps[i * 2] = randn(rng);
      eps[i * 2 + 1] = randn(rng);
      xt[i * 2] = s0 * x0[i * 2] + s1 * eps[i * 2];
      xt[i * 2 + 1] = s0 * x0[i * 2 + 1] + s1 * eps[i * 2 + 1];
      if (rng() < drop) {
        used[i * 2] = 0;
        used[i * 2 + 1] = 0;
      } else if (labels) {
        used[i * 2] = labels[i * 2];
        used[i * 2 + 1] = labels[i * 2 + 1];
      }
    }
    const pred = this.predict(xt, tArray, labels ? used : null, batch);
    const g = new Float32Array(batch * 2);
    let mse = 0;
    for (let i = 0; i < batch * 2; i++) {
      const d = pred[i] - eps[i];
      mse += d * d;
      g[i] = 2 * d;
    }
    this.net.backward(g, true);
    this.net.adam(lr, 1e-5);
    return { mse: mse / (batch * 2) };
  }

  guided(x, t, label, w, batch) {
    const tArray = new Int32Array(batch);
    tArray.fill(t);
    const uncond = this.predict(x, tArray, null, batch);
    if (!label || w === 0) return uncond;
    const condIn = new Float32Array(batch * 2);
    for (let i = 0; i < batch; i++) {
      condIn[i * 2] = label[0];
      condIn[i * 2 + 1] = label[1];
    }
    const cond = this.predict(x, tArray, condIn, batch);
    const out = new Float32Array(batch * 2);
    for (let i = 0; i < out.length; i++) out[i] = uncond[i] + w * (cond[i] - uncond[i]);
    return { eps: out, uncond, cond };
  }

  reverseStep(x, t, z, label, w) {
    const batch = x.length / 2;
    const guided = this.guided(x, t, label, w, batch);
    const eps = guided.eps || guided;
    const { betas, alphas, alphaBars } = this.schedule;
    const beta = betas[t];
    const alpha = alphas[t];
    const ab = alphaBars[t];
    const coef = beta / Math.sqrt(Math.max(1e-8, 1 - ab));
    const inv = 1 / Math.sqrt(alpha);
    let sigma = 0;
    if (t > 0) {
      const abPrev = alphaBars[t - 1];
      sigma = Math.sqrt(Math.max(0, beta * (1 - abPrev) / (1 - ab)));
    }
    const next = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const mean = inv * (x[i] - coef * eps[i]);
      next[i] = mean + sigma * (z ? z[i] : 0);
    }
    return next;
  }
}

export function pOne(prefix) {
  if (prefix.length === 0) return 0.8;
  if (prefix.length === 1) return prefix[0] === 1 ? 0.25 : 0.55;
  if (prefix.length === 2) return prefix[0] === 1 && prefix[1] === 0 ? 0.7 : 0.4;
  if (prefix.length === 3) {
    const key = prefix.join(",");
    if (key === "1,0,1") return 0.9;
    if (key === "1,1,1") return 0.45;
    if (key === "1,0,0") return 0.3;
    return 0.5;
  }
  throw new Error("This demonstration has four pixels");
}

export function tempered(p, temperature) {
  const clipped = Math.min(1 - 1e-4, Math.max(1e-4, p));
  const logit = Math.log(clipped / (1 - clipped)) / temperature;
  return 1 / (1 + Math.exp(-logit));
}

export const OBSERVED = [1, 0, 1, 1];

export function factorsFor(image, temperature = 1) {
  return image.map((value, i) => {
    const p1 = tempered(pOne(image.slice(0, i)), temperature);
    return value === 1 ? p1 : 1 - p1;
  });
}
