import { mountStage } from "./render.js";
import { createScenes } from "./scenes.js";

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const scenes = createScenes();
const glCanvas = document.getElementById("gl");
const inkCanvas = document.getElementById("ink");
const boot = document.getElementById("boot");

const ui = {
  title: document.getElementById("title"),
  nav: document.getElementById("nav"),
  sentence: document.getElementById("sentence"),
  tryThis: document.getElementById("try"),
  modes: document.getElementById("modes"),
  controls: document.getElementById("controls"),
  choices: document.getElementById("choices"),
  actions: document.getElementById("actions"),
  product: document.getElementById("product"),
  readouts: document.getElementById("readouts"),
  limit: document.getElementById("limit"),
  play: document.getElementById("play"),
  step: document.getElementById("step"),
};

const ctx = {
  paused: reduced,
  stepOnce: false,
  dragging: false,
  width: 1,
  height: 1,
  reduced,
};

let scene = scenes[0];
let chromeKey = "";
let readoutKey = "";
let productKey = "";

function sig(spec) {
  return [
    scene.id,
    spec.mode,
    spec.controls.map((c) => c.id).join(","),
    spec.choices.map((c) => `${c.id}:${c.options.map((o) => o.id).join(",")}`).join(";"),
    spec.actions.map((a) => `${a.id}:${a.label}`).join(";"),
  ].join("|");
}

function productSignature(parts) {
  if (!parts) return "";
  return parts.map((part) => (typeof part === "string" ? part : `${part.tone}:${part.text}`)).join("|");
}

function button(label, pressed) {
  const el = document.createElement("button");
  el.type = "button";
  el.textContent = label;
  if (pressed != null) el.setAttribute("aria-pressed", pressed ? "true" : "false");
  return el;
}

function buildNav() {
  ui.nav.replaceChildren();
  scenes.forEach((item, index) => {
    const el = button(item.nav, item === scene);
    el.addEventListener("click", () => select(index));
    ui.nav.append(el);
  });
}

function select(index) {
  scene = scenes[index];
  chromeKey = "";
  buildNav();
  ui.title.textContent = scene.title;
}

function paintProduct(parts) {
  const key = productSignature(parts);
  if (key === productKey) return;
  productKey = key;
  ui.product.replaceChildren();
  if (!parts) {
    ui.product.hidden = true;
    return;
  }
  ui.product.hidden = false;
  const factors = parts.filter((part) => typeof part !== "string");
  factors.forEach((part, index) => {
    if (index) ui.product.append(document.createTextNode(" × "));
    const span = document.createElement("span");
    span.textContent = part.text;
    span.className = part.tone || "";
    ui.product.append(span);
  });
}

function paint(spec) {
  if (ui.sentence.textContent !== spec.sentence) ui.sentence.textContent = spec.sentence;
  if (ui.tryThis.textContent !== spec.tryThis) ui.tryThis.textContent = spec.tryThis;
  if (ui.limit.textContent !== spec.limit) ui.limit.textContent = spec.limit;
  inkCanvas.setAttribute("aria-label", spec.sentence);
  const playing = !ctx.paused;
  ui.play.querySelector("span").textContent = playing ? "Pause" : "Play";
  ui.play.querySelector(".icon-play").hidden = playing;
  ui.play.querySelector(".icon-pause").hidden = !playing;
  ui.play.setAttribute("aria-pressed", playing ? "true" : "false");
  paintProduct(spec.product);

  const key = sig(spec);
  if (key !== chromeKey) {
    chromeKey = key;
    ui.modes.replaceChildren();
    for (const mode of spec.modes) {
      const el = button(mode.label, mode.id === spec.mode);
      el.addEventListener("click", () => {
        scene.setMode(mode.id);
        chromeKey = "";
      });
      ui.modes.append(el);
    }
    ui.controls.replaceChildren();
    for (const control of spec.controls) {
      const label = document.createElement("label");
      label.className = "slider";
      const span = document.createElement("span");
      const name = document.createElement("span");
      name.textContent = control.label;
      const value = document.createElement("b");
      value.textContent = Number(control.value).toFixed(control.digits);
      span.append(name, value);
      const input = document.createElement("input");
      input.type = "range";
      input.min = String(control.min);
      input.max = String(control.max);
      input.step = String(control.step);
      input.value = String(control.value);
      input.dataset.id = control.id;
      input.setAttribute("aria-label", control.label);
      input.addEventListener("input", () => {
        const next = Number(input.value);
        scene.setControl(control.id, next);
        value.textContent = next.toFixed(control.digits);
      });
      label.append(span, input);
      ui.controls.append(label);
    }
    ui.choices.replaceChildren();
    for (const choice of spec.choices) {
      const wrap = document.createElement("div");
      wrap.className = "choice";
      const name = document.createElement("span");
      name.textContent = choice.label;
      const row = document.createElement("div");
      for (const option of choice.options) {
        const el = button(option.label, option.id === choice.value);
        el.dataset.value = String(option.id);
        el.addEventListener("click", () => {
          scene.setControl(choice.id, option.id);
          chromeKey = "";
        });
        row.append(el);
      }
      wrap.append(name, row);
      ui.choices.append(wrap);
    }
    ui.actions.replaceChildren();
    for (const action of spec.actions) {
      const el = button(action.label);
      if (action.button) el.className = "action-button";
      el.addEventListener("click", () => scene.action(action.id));
      ui.actions.append(el);
    }
    readoutKey = "";
  } else {
    for (const input of ui.controls.querySelectorAll("input")) {
      if (document.activeElement === input) continue;
      const control = spec.controls.find((item) => item.id === input.dataset.id);
      if (!control) continue;
      input.value = String(control.value);
      input.parentElement.querySelector("b").textContent = Number(control.value).toFixed(control.digits);
    }
  }

  const nextReadoutKey = spec.readouts.map((item) => item.label).join("|");
  if (nextReadoutKey !== readoutKey) {
    readoutKey = nextReadoutKey;
    ui.readouts.replaceChildren();
    for (const item of spec.readouts) {
      const row = document.createElement("div");
      row.className = "readout";
      const name = document.createElement("span");
      name.textContent = item.label;
      const value = document.createElement("strong");
      value.textContent = item.value;
      row.append(name, value);
      ui.readouts.append(row);
    }
  } else {
    const values = ui.readouts.querySelectorAll("strong");
    spec.readouts.forEach((item, index) => {
      if (values[index] && values[index].textContent !== item.value) values[index].textContent = item.value;
    });
  }
}

function main() {
  boot.hidden = true;
  const stage = mountStage(glCanvas, inkCanvas);
  buildNav();
  ui.title.textContent = scene.title;

  ui.play.addEventListener("click", () => {
    ctx.paused = !ctx.paused;
  });
  ui.step.addEventListener("click", () => {
    ctx.stepOnce = true;
  });
  document.addEventListener("pointerdown", (event) => {
    ctx.dragging = event.target instanceof HTMLInputElement && event.target.type === "range";
  });
  document.addEventListener("pointerup", () => {
    ctx.dragging = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.code === "Space") {
      event.preventDefault();
      ctx.paused = !ctx.paused;
    }
    const index = Number(event.key) - 1;
    if (index >= 0 && index < scenes.length && !event.metaKey && !event.ctrlKey) select(index);
  });
  inkCanvas.addEventListener("pointerdown", (event) => {
    const rect = inkCanvas.getBoundingClientRect();
    scene.pointer(stage.pick(event.clientX - rect.left, event.clientY - rect.top));
  });

  const box = glCanvas.parentElement;
  const observer = new ResizeObserver(() => stage.resize());
  observer.observe(box);
  stage.resize();

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    stage.resize();
    ctx.width = stage.cssW;
    ctx.height = stage.cssH;
    scene.update(dt, ctx);
    ctx.stepOnce = false;
    stage.begin();
    scene.draw(stage);
    stage.end();
    paint(scene.spec());
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

try {
  main();
} catch (error) {
  boot.hidden = false;
  boot.textContent = error instanceof Error ? error.message : "The demonstration could not start.";
}
