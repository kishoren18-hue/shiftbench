// "Move the lamp": fixed vs adaptive cutoff on simulated episode scores.
// Same rule as shiftbench/adaptive.py, on the log scale (adaptive_log).
(() => {
  const cv = document.getElementById("sim"); if (!cv) return;
  const ctx = cv.getContext("2d");
  const $ = id => document.getElementById(id);
  let seed = 7;
  const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
  const randn = () => { let u = 0, v = 0; while (!u) u = rand(); v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

  function run() {
    const shift = +$("shift").value, alpha = +$("alpha").value, etaMult = +$("eta").value;
    $("shiftv").textContent = shift.toFixed(2); $("alphav").textContent = (alpha * 100).toFixed(0) + "%"; $("etav").textContent = etaMult.toFixed(1) + "×IQR";
    const s0 = seed;
    // calibration: 60 nominal successes, log-score ~ N(0, 0.25)
    const cal = Array.from({ length: 60 }, () => 0.25 * randn()).sort((a, b) => a - b);
    const k = Math.ceil((cal.length + 1) * (1 - alpha));
    const tauFix = k > cal.length ? Infinity : cal[k - 1];
    const iqr = cal[Math.floor(0.75 * cal.length)] - cal[Math.floor(0.25 * cal.length)];
    const eta = etaMult * iqr, sMax = cal[cal.length - 1] + 20 * iqr;
    const N = 300, SH = 100, eps = [];
    let tau = tauFix;
    for (let i = 0; i < N; i++) {
      const fail = rand() < 0.2;
      let s = 0.25 * randn() + (fail ? 0.45 + 0.35 * rand() : 0) + (i >= SH ? shift : 0);
      const aF = s > tauFix, aA = Math.min(s, sMax) > tau;
      eps.push({ s, fail, aF, aA, tau });
      if (!fail) tau = tau + eta * ((aA ? 1 : 0) - alpha);
    }
    seed = s0;
    draw(eps, tauFix, alpha, SH);
    const after = eps.slice(SH), succ = after.filter(e => !e.fail), fails = after.filter(e => e.fail);
    const rate = (arr, key) => arr.length ? arr.filter(e => e[key]).length / arr.length : NaN;
    const last = succ.slice(-100);
    $("simstats").innerHTML = [
      ["Fixed: false alarms after the shift", rate(succ, "aF")],
      ["Adaptive: false alarms after the shift", rate(succ, "aA")],
      ["Adaptive: last 100 good runs", rate(last, "aA")],
      ["Fixed: failures caught", rate(fails, "aF")],
      ["Adaptive: failures caught", rate(fails, "aA")],
    ].map(([l, v]) => `<div class="stat"><b>${(100 * v).toFixed(0)}%</b><br><span class="note">${l}</span></div>`).join("");
  }

  function draw(eps, tauFix, alpha, SH) {
    const W = cv.width, H = cv.height, P = 46, h1 = 240, top2 = 285, h2 = 105;
    ctx.clearRect(0, 0, W, H);
    const ys = eps.map(e => e.s).concat([tauFix]);
    const lo = Math.min(...ys) - 0.1, hi = Math.max(...ys) + 0.1;
    const X = i => P + (W - P - 12) * i / (eps.length - 1);
    const Y = v => 14 + h1 * (1 - (v - lo) / (hi - lo));
    ctx.font = "13px system-ui"; ctx.fillStyle = "#8b949e";
    ctx.fillText("episode score (log)", 6, 12);
    ctx.strokeStyle = "#30363d"; ctx.beginPath(); ctx.moveTo(X(SH), 10); ctx.lineTo(X(SH), H - 8); ctx.stroke();
    ctx.fillText("lamp moves", X(SH) + 4, 28);
    eps.forEach((e, i) => { ctx.fillStyle = e.fail ? "#f85149" : "#58a6ff"; ctx.beginPath(); ctx.arc(X(i), Y(e.s), 2.6, 0, 7); ctx.fill(); });
    ctx.setLineDash([6, 5]); ctx.strokeStyle = "#e6edf3"; ctx.beginPath(); ctx.moveTo(X(0), Y(tauFix)); ctx.lineTo(X(eps.length - 1), Y(tauFix)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 2; ctx.beginPath(); eps.forEach((e, i) => i ? ctx.lineTo(X(i), Y(e.tau)) : ctx.moveTo(X(i), Y(e.tau))); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#e6edf3"; ctx.fillText("- - fixed cutoff", W - 260, 30); ctx.fillStyle = "#3fb950"; ctx.fillText("— adaptive cutoff", W - 140, 30);
    // bottom: rolling false-alarm rate over last 25 successes
    const roll = key => { const out = []; const buf = []; eps.forEach((e, i) => { if (!e.fail) { buf.push(e[key] ? 1 : 0); if (buf.length > 25) buf.shift(); } out.push(buf.length >= 5 ? buf.reduce((a, b) => a + b, 0) / buf.length : null); }); return out; };
    const Y2 = v => top2 + h2 * (1 - v);
    ctx.fillStyle = "#8b949e"; ctx.fillText("false-alarm rate, last 25 good runs", 6, top2 - 8);
    ctx.strokeStyle = "#30363d"; [0, 0.5, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(P, Y2(v)); ctx.lineTo(W - 12, Y2(v)); ctx.stroke(); ctx.fillText((v * 100) + "%", 8, Y2(v) + 4); });
    ctx.setLineDash([2, 4]); ctx.strokeStyle = "#8b949e"; ctx.beginPath(); ctx.moveTo(P, Y2(alpha)); ctx.lineTo(W - 12, Y2(alpha)); ctx.stroke(); ctx.setLineDash([]);
    [["aF", "#e6edf3", [6, 5]], ["aA", "#3fb950", []]].forEach(([k, c, d]) => {
      const r = roll(k); ctx.setLineDash(d); ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath(); let started = false;
      r.forEach((v, i) => { if (v === null) return; started ? ctx.lineTo(X(i), Y2(v)) : (ctx.moveTo(X(i), Y2(v)), started = true); }); ctx.stroke(); ctx.lineWidth = 1; ctx.setLineDash([]);
    });
  }
  ["shift", "alpha", "eta"].forEach(id => $(id).addEventListener("input", run));
  $("reroll").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e9); run(); });
  run();
})();
