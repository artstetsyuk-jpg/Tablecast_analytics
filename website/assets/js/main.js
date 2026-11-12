/* TableCast landing page: scene, parallax, scroll story, reveals. No dependencies. */
(() => {
  "use strict";

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wide = matchMedia("(min-width: 961px)");
  const finePointer = matchMedia("(pointer: fine)").matches;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

  // Deterministic randomness so the skyline looks the same on every visit.
  const seeded = (seed) => () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  /* ------------------------------------------------------------ links & small bits */
  const appUrl = window.TABLECAST_APP_URL || "#";
  $$("[data-app-link]").forEach((a) => { a.href = appUrl; a.rel = "noopener"; });
  const year = $("[data-year]");
  if (year) year.textContent = new Date().getFullYear();
  $$("[data-depth]").forEach((el) => el.style.setProperty("--d", el.dataset.depth));

  /* ------------------------------------------------------------ bokeh (foreground lights) */
  const bokeh = $("[data-bokeh]");
  if (bokeh) {
    const r = seeded(7);
    const colors = ["rgba(255,190,120,.9)", "rgba(249,115,22,.85)", "rgba(255,225,170,.8)", "rgba(251,113,133,.55)"];
    for (let i = 0; i < 18; i++) {
      const s = document.createElement("span");
      const size = 40 + r() * 170;
      s.style.cssText = [
        `left:${(r() * 100).toFixed(1)}%`, `top:${(38 + r() * 62).toFixed(1)}%`,
        `width:${size.toFixed(0)}px`, `height:${size.toFixed(0)}px`,
        `--c:${colors[Math.floor(r() * colors.length)]}`, `--b:${(8 + r() * 22).toFixed(0)}px`,
        `--o:${(0.22 + r() * 0.45).toFixed(2)}`, `--t:${(9 + r() * 10).toFixed(1)}s`,
        `--dx:${((r() - 0.5) * 90).toFixed(0)}px`, `--dy:${((r() - 0.5) * 50).toFixed(0)}px`,
        `animation-delay:${(-r() * 10).toFixed(1)}s`,
      ].join(";");
      bokeh.appendChild(s);
    }
  }

  /* ------------------------------------------------------------ skylines on canvas */
  const CITY = {
    far: { seed: 11, top: "#3A2236", bottom: "#1F1525", minH: 0.26, maxH: 0.6, minW: 34, maxW: 92, lit: 0.2, alpha: 0.5, win: [3, 5, 6, 9] },
    near: { seed: 23, top: "#1A121F", bottom: "#0B0A10", minH: 0.32, maxH: 0.94, minW: 56, maxW: 150, lit: 0.3, alpha: 0.95, win: [5, 7, 9, 13] },
    finale: { seed: 5, top: "#1C1322", bottom: "#0B0A10", minH: 0.24, maxH: 0.92, minW: 44, maxW: 128, lit: 0.32, alpha: 0.9, win: [4, 6, 8, 12] },
  };
  const WARM = ["255,206,140", "253,186,116", "255,226,170", "252,211,77"];

  function buildCity(canvas) {
    const cfg = CITY[canvas.dataset.city];
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return null;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const r = seeded(cfg.seed);

    // Static part (buildings) is painted once into an offscreen canvas.
    const base = document.createElement("canvas");
    base.width = canvas.width; base.height = canvas.height;
    const b = base.getContext("2d");
    b.scale(dpr, dpr);
    const windows = [];
    const [ww, wh, sx, sy] = cfg.win;
    let x = -20;
    while (x < W + 20) {
      const w = cfg.minW + r() * (cfg.maxW - cfg.minW);
      // Taller buildings toward the middle-right, like a real skyline's centre.
      const centre = 1 - Math.abs((x + w / 2) / W - 0.62) * 0.9;
      const h = H * (cfg.minH + r() * (cfg.maxH - cfg.minH)) * (0.65 + 0.35 * centre);
      const top = H - h;
      const g = b.createLinearGradient(0, top, 0, H);
      g.addColorStop(0, cfg.top); g.addColorStop(1, cfg.bottom);
      b.fillStyle = g;
      b.fillRect(x, top, w, h);
      if (r() < 0.25) { b.fillRect(x + w * 0.4, top - h * 0.08, Math.max(2, w * 0.06), h * 0.08); } // antenna
      if (r() < 0.3) { b.fillRect(x + w * 0.15, top - 10, w * 0.7, 10); }                       // roof block
      // Windows grid
      for (let wy = top + sy; wy < H - wh - 4; wy += wh + sy) {
        for (let wx = x + sx; wx < x + w - ww - sx / 2; wx += ww + sx) {
          if (r() < cfg.lit) {
            windows.push({ x: wx, y: wy, w: ww, h: wh, rgb: WARM[Math.floor(r() * WARM.length)],
              a: cfg.alpha * (0.45 + r() * 0.55), at: r() * 2.6, on: true });
          }
        }
      }
      x += w + (r() < 0.2 ? 6 + r() * 14 : 0);
    }
    return { canvas, ctx: canvas.getContext("2d"), base, windows, dpr, start: null, done: false };
  }

  function paintCity(city, now) {
    const { ctx, base, windows, dpr } = city;
    if (city.start === null) city.start = now;
    const t = reduceMotion ? 99 : (now - city.start) / 1000;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, city.canvas.width, city.canvas.height);
    ctx.drawImage(base, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (const w of windows) {
      if (!w.on || t < w.at) continue;
      const fade = clamp((t - w.at) / 0.35);
      ctx.fillStyle = `rgba(${w.rgb},${(w.a * fade).toFixed(3)})`;
      ctx.fillRect(w.x, w.y, w.w, w.h);
    }
    city.done = t > 3.2;
  }

  const cities = new Map();
  function setupCities() {
    $$("canvas[data-city]").forEach((c) => {
      const city = buildCity(c);
      if (city) { cities.set(c, Object.assign(city, { visible: cities.get(c)?.visible ?? false })); paintCity(city, performance.now()); }
    });
  }
  setupCities();
  const cityObserver = new IntersectionObserver((entries) => {
    for (const e of entries) { const city = cities.get(e.target); if (city) city.visible = e.isIntersecting; }
  });
  $$("canvas[data-city]").forEach((c) => cityObserver.observe(c));
  let resizeTimer;
  addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(setupCities, 200); });

  // Once the intro is over, a window switches on or off now and then: the city is alive.
  if (!reduceMotion) {
    setInterval(() => {
      for (const city of cities.values()) {
        if (!city.visible || !city.done || !city.windows.length) continue;
        const w = city.windows[Math.floor(Math.random() * city.windows.length)];
        w.on = !w.on;
        paintCity(city, performance.now());
      }
    }, 450);
  }

  /* ------------------------------------------------------------ hero parallax + nav */
  const hero = $("[data-hero]");
  const nav = $("[data-nav]");
  let target = { x: 0, y: 0 }, mouse = { x: 0, y: 0 }, lastP = -1;
  if (finePointer && !reduceMotion) {
    addEventListener("pointermove", (e) => {
      target.x = (e.clientX / innerWidth - 0.5) * 2;
      target.y = (e.clientY / innerHeight - 0.5) * 2;
    }, { passive: true });
  }

  /* ------------------------------------------------------------ pinned story */
  const story = $("[data-story]");
  let storyCycle = null;
  function setStep(step) { if (story && story.dataset.active !== String(step)) story.dataset.active = String(step); }
  function updateStory() {
    if (!story || !wide.matches || reduceMotion) return;
    const rect = story.getBoundingClientRect();
    const total = story.offsetHeight - innerHeight;
    const prog = clamp(-rect.top / total);
    story.style.setProperty("--sp", prog.toFixed(4));
    setStep(Math.min(2, Math.floor(prog * 3.0001)));
  }
  // Small screens / reduced motion: no pinning, the screen plays the steps by itself.
  if (story) {
    new IntersectionObserver(([e]) => {
      const auto = !wide.matches || reduceMotion;
      clearInterval(storyCycle);
      if (auto && e.isIntersecting) {
        storyCycle = setInterval(() => setStep((Number(story.dataset.active) + 1) % 3), 3400);
      }
    }, { threshold: 0.35 }).observe(story);
  }

  function frame(now) {
    // Hero: scroll progress and eased mouse position drive the depth layers via CSS variables.
    if (hero) {
      const p = clamp(scrollY / hero.offsetHeight);
      mouse.x += (target.x - mouse.x) * 0.05;
      mouse.y += (target.y - mouse.y) * 0.05;
      if (!reduceMotion && (Math.abs(p - lastP) > 0.0005 || Math.abs(target.x - mouse.x) > 0.001 || Math.abs(target.y - mouse.y) > 0.001)) {
        hero.style.setProperty("--p", p.toFixed(4));
        hero.style.setProperty("--mx", mouse.x.toFixed(4));
        hero.style.setProperty("--my", mouse.y.toFixed(4));
        lastP = p;
      }
    }
    nav?.classList.toggle("is-scrolled", scrollY > 40);
    updateStory();
    for (const city of cities.values()) if (city.visible && !city.done) paintCity(city, now);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  /* ------------------------------------------------------------ reveals */
  const revealObserver = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("is-in");
      revealObserver.unobserve(e.target);
    }
  }, { threshold: 0.18, rootMargin: "0px 0px -6% 0px" });
  $$(".reveal, .mask-reveal").forEach((el) => revealObserver.observe(el));

  // Feature tiles: the glow follows the pointer.
  $$(".tile").forEach((tile) => tile.addEventListener("pointermove", (e) => {
    const r = tile.getBoundingClientRect();
    tile.style.setProperty("--gx", `${((e.clientX - r.left) / r.width * 100).toFixed(1)}%`);
    tile.style.setProperty("--gy", `${((e.clientY - r.top) / r.height * 100).toFixed(1)}%`);
  }));

  /* ------------------------------------------------------------ charts */
  // Smooth line through points (Catmull-Rom -> cubic Bézier).
  function smoothPath(pts) {
    let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
      const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += ` C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
    }
    return d;
  }
  const scale = (values, W, H, pad, lo, hi) =>
    values.map((v, i) => [i / (values.length - 1) * W, H - pad - (v - lo) / (hi - lo) * (H - 2 * pad)]);

  // "Learning the rhythm" waves in the story screen.
  const weekly = [0.55, 0.6, 0.66, 0.72, 0.9, 1, 0.78];
  const waveA = Array.from({ length: 22 }, (_, i) => weekly[i % 7] * (1 + 0.06 * Math.sin(i / 3)));
  const waveB = Array.from({ length: 22 }, (_, i) => weekly[(i + 6) % 7] * 0.92 + 0.04 * Math.cos(i));
  const learnA = $('[data-wave="a"]'), learnB = $('[data-wave="b"]');
  if (learnA) learnA.setAttribute("d", smoothPath(scale(waveA, 400, 220, 22, 0.45, 1.1)));
  if (learnB) learnB.setAttribute("d", smoothPath(scale(waveB, 400, 220, 22, 0.45, 1.1)));
  [learnA, learnB].forEach((p) => p?.setAttribute("pathLength", "1"));

  // Accuracy chart: an illustrative five weeks of a café's guests.
  const accSvg = $("[data-acc-chart]");
  if (accSvg) {
    const r = seeded(42);
    const base = [96, 104, 112, 121, 168, 186, 139];
    const days = 42;
    const actual = Array.from({ length: days }, (_, i) => base[i % 7] * (1 + 0.1 * Math.sin(i / 6)) * (0.9 + r() * 0.2));
    const shown = actual.slice(7);
    const forecast = shown.map((v, i) => base[(i + 7) % 7] * (1 + 0.1 * Math.sin((i + 7) / 6)) * (0.97 + r() * 0.06) * 0.5 + v * 0.5);
    const baseline = actual.slice(0, days - 7);
    const lo = Math.min(...shown, ...baseline) * 0.9, hi = Math.max(...shown, ...baseline) * 1.05;
    const series = { actual: shown, forecast, baseline };
    for (const [name, vals] of Object.entries(series)) {
      const path = $(`[data-series="${name}"]`, accSvg);
      path.setAttribute("d", smoothPath(scale(vals, 560, 300, 26, lo, hi)));
      if (name !== "baseline") path.setAttribute("pathLength", "1");
    }
  }

  // Count-up numbers + draw the chart when the accuracy section enters.
  const accuracy = $("[data-accuracy]");
  if (accuracy) {
    new IntersectionObserver(([e], obs) => {
      if (!e.isIntersecting) return;
      accuracy.classList.add("is-in");
      $$("[data-countup]", accuracy).forEach((el) => {
        const end = Number(el.dataset.countup);
        if (reduceMotion) { el.textContent = end; return; }
        const t0 = performance.now();
        const tick = (now) => {
          const k = clamp((now - t0) / 1600);
          el.textContent = Math.round(end * (1 - Math.pow(1 - k, 3)));
          if (k < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
      obs.disconnect();
    }, { threshold: 0.3 }).observe(accuracy);
  }

  /* ------------------------------------------------------------ go */
  // Fonts first, so the title doesn't reveal in a fallback font and then jump.
  const start = () => requestAnimationFrame(() => document.documentElement.classList.add("is-loaded"));
  (document.fonts?.ready || Promise.resolve()).then(start);
  setTimeout(start, 1500); // never wait forever on slow font loads
})();
