/** WebGL2 plate: density fields, thick curves, and soft particles. */

const MESH_VS = `#version 300 es
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;
uniform vec2 u_min;
uniform vec2 u_span;
out vec4 v_color;
void main() {
  vec2 n = (a_pos - u_min) / u_span;
  gl_Position = vec4(n * 2.0 - 1.0, 0.0, 1.0);
  v_color = a_color;
}`;

const MESH_FS = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 frag;
void main() { frag = v_color; }`;

const POINT_VS = `#version 300 es
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;
layout(location=2) in float a_size;
uniform vec2 u_min;
uniform vec2 u_span;
uniform float u_dpr;
out vec4 v_color;
void main() {
  vec2 n = (a_pos - u_min) / u_span;
  gl_Position = vec4(n * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = max(1.0, a_size * u_dpr);
  v_color = a_color;
}`;

const POINT_FS = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 frag;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  float alpha = smoothstep(0.5, 0.18, r);
  float core = smoothstep(0.28, 0.0, r);
  vec3 col = mix(v_color.rgb, vec3(1.0), core * 0.45);
  frag = vec4(col, alpha * v_color.a);
}`;

const FIELD_VS = `#version 300 es
layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_uv;
out vec2 v_uv;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
  v_uv = a_uv;
}`;

const FIELD_FS = `#version 300 es
precision mediump float;
in vec2 v_uv;
uniform sampler2D u_field;
out vec4 frag;
void main() { frag = texture(u_field, v_uv); }`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
  }
  return shader;
}

function program(gl, vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(prog) || "program link failed");
  }
  return prog;
}

function Grow(stride) {
  return { stride, data: new Float32Array(stride * 256), count: 0 };
}

function push(buf, values) {
  const need = buf.count + values.length;
  if (need > buf.data.length) {
    let size = buf.data.length;
    while (size < need) size *= 2;
    const next = new Float32Array(size);
    next.set(buf.data);
    buf.data = next;
  }
  buf.data.set(values, buf.count);
  buf.count += values.length;
}

export function mountStage(glCanvas, inkCanvas) {
  const gl = glCanvas.getContext("webgl2", {
    alpha: false,
    antialias: true,
    premultipliedAlpha: false,
  });
  if (!gl) throw new Error("WebGL2 is unavailable in this browser");
  const ink = inkCanvas.getContext("2d");
  const mesh = program(gl, MESH_VS, MESH_FS);
  const points = program(gl, POINT_VS, POINT_FS);
  const field = program(gl, FIELD_VS, FIELD_FS);
  const meshPos = gl.createBuffer();
  const pointPos = gl.createBuffer();
  const linePos = gl.createBuffer();
  const fieldPos = gl.createBuffer();
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fieldQuad = new Float32Array([
    -1, -1, 0, 0,
    1, -1, 1, 0,
    -1, 1, 0, 1,
    1, 1, 1, 1,
  ]);
  const maxPoint = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
  const tris = Grow(6);
  const lines = Grow(6);
  const dots = Grow(7);
  const plots = new Map();
  const stage = {
    gl,
    cssW: 1,
    cssH: 1,
    dpr: 1,
    current: null,
  };

  function useCamera(prog) {
    const plot = plots.get(stage.current);
    gl.uniform2f(gl.getUniformLocation(prog, "u_min"), plot.xmin, plot.ymin);
    gl.uniform2f(gl.getUniformLocation(prog, "u_span"), plot.xmax - plot.xmin, plot.ymax - plot.ymin);
  }

  function flush() {
    if (!stage.current) return;
    const plot = plots.get(stage.current);
    if (!plot || plot.w < 2 || plot.h < 2) {
      tris.count = 0;
      lines.count = 0;
      dots.count = 0;
      stage._field = null;
      return;
    }
    gl.enable(gl.SCISSOR_TEST);
    gl.viewport(plot.vx, plot.vy, plot.vw, plot.vh);
    gl.scissor(plot.vx, plot.vy, plot.vw, plot.vh);
    gl.disableVertexAttribArray(2);
    if (stage._field) {
      const { pix, w, h } = stage._field;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, pix);
      gl.useProgram(field);
      gl.bindBuffer(gl.ARRAY_BUFFER, fieldPos);
      gl.bufferData(gl.ARRAY_BUFFER, fieldQuad, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      stage._field = null;
    }
    if (tris.count) {
      gl.useProgram(mesh);
      useCamera(mesh);
      gl.bindBuffer(gl.ARRAY_BUFFER, meshPos);
      gl.bufferData(gl.ARRAY_BUFFER, tris.data.subarray(0, tris.count), gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 8);
      gl.drawArrays(gl.TRIANGLES, 0, tris.count / 6);
      tris.count = 0;
    }
    if (lines.count) {
      gl.useProgram(mesh);
      useCamera(mesh);
      gl.bindBuffer(gl.ARRAY_BUFFER, linePos);
      gl.bufferData(gl.ARRAY_BUFFER, lines.data.subarray(0, lines.count), gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 8);
      gl.drawArrays(gl.LINES, 0, lines.count / 6);
      lines.count = 0;
    }
    if (dots.count) {
      gl.useProgram(points);
      useCamera(points);
      gl.uniform1f(gl.getUniformLocation(points, "u_dpr"), stage.dpr);
      gl.bindBuffer(gl.ARRAY_BUFFER, pointPos);
      gl.bufferData(gl.ARRAY_BUFFER, dots.data.subarray(0, dots.count), gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 28, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 8);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 28, 24);
      gl.drawArrays(gl.POINTS, 0, dots.count / 7);
      dots.count = 0;
    }
  }

  stage.resize = function resize() {
    const rect = glCanvas.getBoundingClientRect();
    stage.cssW = Math.max(1, rect.width);
    stage.cssH = Math.max(1, rect.height);
    stage.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(stage.cssW * stage.dpr);
    const h = Math.round(stage.cssH * stage.dpr);
    if (glCanvas.width !== w || glCanvas.height !== h) {
      glCanvas.width = w;
      glCanvas.height = h;
    }
    if (inkCanvas.width !== w || inkCanvas.height !== h) {
      inkCanvas.width = w;
      inkCanvas.height = h;
    }
    ink.setTransform(stage.dpr, 0, 0, stage.dpr, 0, 0);
  };

  stage.begin = function begin() {
    plots.clear();
    stage.current = null;
    tris.count = 0;
    lines.count = 0;
    dots.count = 0;
    stage._field = null;
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.clearColor(0.063, 0.086, 0.114, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    ink.clearRect(0, 0, stage.cssW, stage.cssH);
  };

  stage.plot = function plot(name, spec) {
    flush();
    const dpr = stage.dpr;
    const vy = (stage.cssH - spec.y - spec.h) * dpr;
    plots.set(name, {
      ...spec,
      vx: spec.x * dpr,
      vy,
      vw: spec.w * dpr,
      vh: spec.h * dpr,
    });
    stage.current = name;
  };

  stage.field = function field(pix, w, h) {
    stage._field = { pix, w, h };
  };

  stage.quad = function quad(x, y, w, h, color) {
    const [r, g, b, a] = color;
    const x1 = x + w;
    const y1 = y + h;
    push(tris, [
      x, y, r, g, b, a, x1, y, r, g, b, a, x1, y1, r, g, b, a,
      x, y, r, g, b, a, x1, y1, r, g, b, a, x, y1, r, g, b, a,
    ]);
  };

  stage.line = function line(x0, y0, x1, y1, color) {
    const [r, g, b, a] = color;
    push(lines, [x0, y0, r, g, b, a, x1, y1, r, g, b, a]);
  };

  stage.ribbon = function ribbon(pts, width, color) {
    const [r, g, b, a] = color;
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[i + 1];
      const dx = x1 - x0;
      const dy = y1 - y0;
      const len = Math.hypot(dx, dy) || 1;
      const nx = (-dy / len) * width * 0.5;
      const ny = (dx / len) * width * 0.5;
      push(tris, [
        x0 + nx, y0 + ny, r, g, b, a,
        x0 - nx, y0 - ny, r, g, b, a,
        x1 - nx, y1 - ny, r, g, b, a,
        x0 + nx, y0 + ny, r, g, b, a,
        x1 - nx, y1 - ny, r, g, b, a,
        x1 + nx, y1 + ny, r, g, b, a,
      ]);
    }
  };

  stage.point = function point(x, y, color, size) {
    const [r, g, b, a] = color;
    push(dots, [x, y, r, g, b, a, Math.min(size, maxPoint)]);
  };

  stage.glow = function glow(x, y, rgb, size, alpha = 0.9) {
    stage.point(x, y, [rgb[0], rgb[1], rgb[2], alpha * 0.18], size * 2.7);
    stage.point(x, y, [rgb[0], rgb[1], rgb[2], alpha], size);
  };

  stage.grid = function grid(step, color) {
    const plot = plots.get(stage.current);
    if (!plot) return;
    const x0 = Math.ceil(plot.xmin / step) * step;
    const y0 = Math.ceil(plot.ymin / step) * step;
    for (let x = x0; x < plot.xmax; x += step) stage.line(x, plot.ymin, x, plot.ymax, color);
    for (let y = y0; y < plot.ymax; y += step) stage.line(plot.xmin, y, plot.xmax, y, color);
  };

  stage.end = function end() {
    flush();
    stage.current = null;
    gl.disable(gl.SCISSOR_TEST);
  };

  stage.worldToScreen = function worldToScreen(name, x, y) {
    const p = plots.get(name);
    return [
      p.x + ((x - p.xmin) / (p.xmax - p.xmin)) * p.w,
      p.y + ((p.ymax - y) / (p.ymax - p.ymin)) * p.h,
    ];
  };

  stage.screenText = function screenText(x, y, text, opt = {}) {
    ink.font = `${opt.size || 14}px "Avenir Next", "Segoe UI", sans-serif`;
    ink.fillStyle = opt.color || "#f4f0e8";
    ink.textAlign = opt.align || "left";
    ink.textBaseline = opt.baseline || "alphabetic";
    ink.fillText(text, x, y);
  };

  stage.worldText = function worldText(name, x, y, text, opt = {}) {
    const [sx, sy] = stage.worldToScreen(name, x, y);
    stage.screenText(sx, sy, text, opt);
  };

  stage.pick = function pick(cssX, cssY) {
    for (const [name, p] of plots) {
      if (cssX >= p.x && cssX <= p.x + p.w && cssY >= p.y && cssY <= p.y + p.h) {
        const x = p.xmin + ((cssX - p.x) / p.w) * (p.xmax - p.xmin);
        const y = p.ymax - ((cssY - p.y) / p.h) * (p.ymax - p.ymin);
        return { name, x, y };
      }
    }
    return null;
  };

  return stage;
}

export function fitBox(w, h, cx, cy, half) {
  const aspect = w / Math.max(h, 1);
  const hx = aspect >= 1 ? half * aspect : half;
  const hy = aspect >= 1 ? half : half / aspect;
  return { xmin: cx - hx, xmax: cx + hx, ymin: cy - hy, ymax: cy + hy };
}

export function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)));
  };
  return [f(0), f(8), f(4)];
}

export function mix(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
