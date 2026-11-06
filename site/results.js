// Renders results.json (written by scripts/make_site_results.py from the real run).
(async () => {
  let R;
  try { R = await (await fetch("results.json", { cache: "no-store" })).json(); } catch (e) { return; }
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  if (R.headline) {
    document.getElementById("headline").innerHTML = R.headline.map(h =>
      `<div class="big"><div class="num">${esc(h.value)}</div><div class="lab">${esc(h.label)}</div></div>`).join("");
  }
  const body = document.getElementById("results-body");
  body.innerHTML = (R.intro ? `<p>${R.intro}</p>` : "") + (R.tables || []).map(t => {
    const head = `<tr>${t.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const rows = t.rows.map(r => `<tr>${r.map((v, j) => {
      let cls = "";
      if (t.flag && j > 0 && typeof v === "number") cls = v > t.flag ? "bad" : "ok";
      const txt = typeof v === "number" ? (t.pct ? (100 * v).toFixed(0) + "%" : v.toFixed(2)) : (v ?? "–");
      return `<td class="${cls}">${esc(txt)}</td>`;
    }).join("")}</tr>`).join("");
    return `<h3>${esc(t.title)}</h3>${t.note ? `<p class="note">${t.note}</p>` : ""}<div style="overflow-x:auto"><table>${head}${rows}</table></div>`;
  }).join("") + (R.figures || []).map(f => `<figure><img src="${f.src}" alt="${esc(f.caption)}" style="width:100%;border-radius:10px;border:1px solid #30363d"><figcaption class="note">${esc(f.caption)}</figcaption></figure>`).join("")
    + (R.footer ? `<p class="note">${R.footer}</p>` : "");
  if (R.videos && R.videos.length) {
    document.getElementById("videos").innerHTML = R.videos.map(v =>
      `<div><video controls muted playsinline preload="metadata" src="${v.src}"></video><p class="note">${esc(v.caption)}</p></div>`).join("");
  }
})();
