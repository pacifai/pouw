"use strict";

const DETECTED = [
  { id: "replay", name: "Replay", desc: "Resend an accepted proof.", check: "replay protection" },
  { id: "precompute", name: "Precomputation", desc: "Submit a proof after its job expired.", check: "freshness" },
  { id: "difficulty", name: "Difficulty tampering", desc: "Claim an easier target than the job's.", check: "job binding" },
  { id: "forged", name: "Forged proof", desc: "Change one activation in an opened row.", check: "proof verification" },
  { id: "model", name: "Model substitution", desc: "Prove with one weight changed.", check: "weight commitment" },
  { id: "theft", name: "Proof theft", desc: "Move the H100's proof onto another GPU's job.", check: "proof verification" },
];
const UNDETECTED = [
  { id: "synthetic", name: "Synthetic inputs", desc: "Multiply random activations with the declared weights.", defense: "Defense: re-run sampled requests; zero-knowledge proofs" },
  { id: "spoof", name: "GPU ID spoofing", desc: "The laptop claims the H100's UUID.", defense: "Defense: CC attestation" },
  { id: null, name: "Covert training", desc: "Train in batches, layers, or time that earn no proofs.", defense: "Defense: capacity accounting; training-vs-inference monitoring" },
  { id: null, name: "Relay to remote GPUs", desc: "Forward a GPU's jobs to GPUs elsewhere.", defense: "Defense: attestation tied to the job; response-time limits" },
];

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => n.toLocaleString("en-US");
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function sci(x) {
  if (!x) return "0";
  const e = Math.floor(Math.log10(x));
  const m = x / 10 ** e;
  return `${m.toFixed(2)} × 10${String(e).split("").map((c) => SUP[c]).join("")}`;
}
const shortId = (g) => (g ? (g.length > 16 ? g.slice(0, 13) + "…" : g) : "none");
const shortWeight = (w) => (w ? w.replace(/^model\./, "").replace(/\.weight$/, "") : "");
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

const outcomes = {};

function renderCards() {
  for (const a of DETECTED) {
    const out = el("span", { class: "outcome mono" });
    outcomes[a.id] = out;
    $("detected").append(
      el("div", { class: "card" },
        el("h3", {}, a.name),
        el("div", { class: "desc" }, a.desc),
        out,
        el("div", { class: "foot" },
          el("span", { class: "check mono" }, a.check),
          el("button", { type: "button", class: "run-detected", "data-attack": a.id, onclick: () => runAttack(a.id) }, "Run"))));
  }
  for (const a of UNDETECTED) {
    const out = el("span", { class: "outcome mono" });
    const card = el("div", { class: "card undetected" },
      el("h3", {}, a.name),
      el("div", { class: "desc" }, a.desc),
      el("div", { class: "defense" }, a.defense));
    if (a.id) {
      outcomes[a.id] = out;
      card.append(out, el("div", { class: "foot" }, el("span"),
        el("button", { type: "button", class: "run-undetected", "data-attack": a.id, onclick: () => runAttack(a.id) }, "Run")));
    }
    $("undetected").append(card);
  }
}

async function runAttack(id) {
  const button = document.querySelector(`button[data-attack="${id}"]`);
  button.disabled = true;
  outcomes[id].textContent = "running…";
  outcomes[id].className = "outcome mono";
  try {
    const r = await (await fetch(`/api/attack/${id}`, { method: "POST" })).json();
    if (!r.ok) {
      outcomes[id].textContent = r.error;
      return;
    }
    const entry = await (await fetch(`/api/entry/${r.entry}`)).json();
    outcomes[id].textContent = entry.accepted ? "accepted" : "rejected";
    outcomes[id].className = `outcome mono ${entry.accepted ? "accepted" : "rejected"}`;
    showDetail(r.entry);
  } finally {
    button.disabled = false;
  }
}

function drawThroughput(points) {
  const svg = $("throughput");
  svg.innerHTML = "";
  if (!points.length) return;
  const last = points[points.length - 1].t;
  const first = last - 119;
  const max = Math.max(1, ...points.map((p) => p.noisy + p.plain));
  const w = 600 / 120;
  for (const p of points) {
    if (p.t < first) continue;
    const x = (p.t - first) * w;
    const hn = (p.noisy / max) * 170;
    const hp = (p.plain / max) * 170;
    svg.append(svgRect(x, 180 - hn, w - 1, hn, "var(--accepted)"));
    svg.append(svgRect(x, 180 - hn - hp, w - 1, hp, "var(--plain)"));
  }
  const now = points[points.length - 1];
  $("tps-now").textContent = `peak ${fmtInt(max)} tokens/s`;
}

function drawBatches(batches) {
  const svg = $("batches");
  svg.innerHTML = "";
  const w = 640 / 160;
  const maxLog = Math.log2(16384);
  batches.forEach((b, i) => {
    const h = Math.max(2, (Math.log2(Math.max(1, b.m)) / maxLog) * 62);
    svg.append(svgRect(i * w, 64 - h, w - 1, h, b.noisy ? "var(--accepted)" : "var(--plain)"));
  });
}

function svgRect(x, y, w, h, fill) {
  const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  r.setAttribute("x", x); r.setAttribute("y", y);
  r.setAttribute("width", Math.max(0, w)); r.setAttribute("height", Math.max(0, h));
  r.setAttribute("fill", fill);
  return r;
}

function renderLog(rows) {
  const body = $("log");
  body.innerHTML = "";
  for (const r of rows) {
    const warmDup = r.source === "H100 warm-up" && r.reason.startsWith("duplicate");
    const reason = el("button", { type: "button", class: "link", onclick: () => showDetail(r.id) },
      warmDup ? "duplicate: vLLM ran the same warm-up batch twice (expected)" : r.reason);
    const tr = el("tr", { class: `${r.accepted || warmDup ? "" : "rej"} ${r.attack ? "attack" : ""}` },
      el("td", {}, `${r.t.toFixed(1)} s`),
      el("td", {}, r.attack ? `${r.source} · ${r.attack}` : r.source),
      el("td", { title: r.gpu_id || "" }, shortId(r.gpu_id)),
      el("td", {}, shortWeight(r.weight)),
      el("td", {}, r.m ? `${fmtInt(r.m)} × ${fmtInt(r.k)}` : ""),
      el("td", { class: r.accepted ? "result-ok" : warmDup ? "result-dup" : "result-no" }, r.accepted ? "accepted" : warmDup ? "duplicate" : "rejected"),
      el("td", { class: "delay-col" }, r.delay_s == null ? "" : `${r.delay_s.toFixed(2)} s`),
      el("td", {}, reason));
    body.append(tr);
  }
}

function renderPerGpu(rows) {
  const body = $("per-gpu");
  body.innerHTML = "";
  for (const r of rows) {
    body.append(el("tr", {},
      el("td", {}, r.gpu_id),
      el("td", {}, fmtInt(r.proofs)),
      el("td", {}, sci(r.macs)),
      el("td", {}, r.sources.join(", "))));
  }
}

let runsShown = "";
function renderRuns(s) {
  const key = JSON.stringify(s.runs.map((r) => [r.label, r.ready])) + s.current;
  if (key === runsShown) return;
  runsShown = key;
  const select = $("run-select");
  select.innerHTML = "";
  for (const r of s.runs) {
    const o = el("option", { value: r.index }, r.mode === "live" ? "Live" : `${r.description || r.label}${r.ready ? "" : " (checking…)"}`);
    if (r.index === s.current) o.selected = true;
    select.append(o);
  }
  select.parentElement.hidden = s.runs.length < 2;
}
$("run-select").addEventListener("change", (ev) => {
  fetch(`/api/run/${ev.target.value}`, { method: "POST" }).then(poll);
  for (const o of Object.values(outcomes)) o.textContent = "";
});

async function showDetail(id) {
  const e = await (await fetch(`/api/entry/${id}`)).json();
  $("detail-title").textContent = e.attack || "Proof";
  $("detail-result").textContent = e.accepted ? "accepted" : "rejected";
  $("detail-result").className = `mono ${e.accepted ? "accepted" : "rejected"}`;
  $("detail-meta").innerHTML = "";
  const meta = [`t = ${e.t.toFixed(1)} s`, `source: ${e.source}`, `GPU ID: ${e.gpu_id || "none"}`];
  if (e.weight) meta.push(`weight: ${shortWeight(e.weight)}`);
  if (e.m) meta.push(`m × k: ${fmtInt(e.m)} × ${fmtInt(e.k)}`);
  if (e.win_probability) meta.push(`win probability: ${e.win_probability.toExponential(2)}`);
  if (e.delay_s != null) meta.push(`verdict delay: ${e.delay_s.toFixed(2)} s`);
  for (const m of meta) $("detail-meta").append(el("span", {}, m));
  $("detail-changes").innerHTML = "";
  $("detail-changes-table").hidden = !e.changes.length;
  for (const c of e.changes) $("detail-changes").append(el("tr", {}, el("td", {}, c.field), el("td", {}, c.original), el("td", {}, c.submitted)));
  $("detail-checks").innerHTML = "";
  for (const c of e.checks) {
    const cls = c.status === "pass" ? "pass" : c.status === "fail" ? "fail" : "skip";
    $("detail-checks").append(el("li", {}, el("span", { class: `s ${cls}` }, c.status), c.name));
  }
  $("detail-reason").textContent = `verifier: "${e.reason}"`;
  if (!$("detail").open) $("detail").showModal();
}

async function poll() {
  try {
    const s = await (await fetch("/api/state")).json();
    $("t-tokens").textContent = fmtInt(s.totals.tokens);
    $("t-accepted").textContent = fmtInt(s.totals.accepted);
    $("t-rejected").textContent = fmtInt(s.totals.rejected);
    $("t-rejected-sub").textContent = s.totals.warmup_duplicates ? `incl. ${s.totals.warmup_duplicates} warm-up duplicate (expected)` : "";
    $("t-macs").textContent = sci(s.totals.proven_macs);
    if (s.timing) {
      const t = s.timing;
      const gemm = t.proofs_off_s / t.plain_s - 1;
      $("t-overhead").textContent = `+${(gemm * 100).toFixed(0)}%`;
      $("t-overhead-sub").textContent = `noisy vs. plain GEMM; with proof building: +${(t.overhead * 100).toFixed(0)}%`;
    }
    $("share").textContent = s.totals.proven_share == null ? "" : `int7 GEMM work in noisy GEMMs: ${(s.totals.proven_share * 100).toFixed(1)}%`;
    const live = s.mode === "live";
    document.body.classList.toggle("replay", !live);
    renderRuns(s);
    $("run-label").innerHTML = "";
    if (live) $("run-label").append(el("span", { class: "live-dot", "aria-hidden": "true" }), `Live, ${s.elapsed_s.toFixed(0)} s`);
    else $("run-label").append(s.description || `Replay of ${s.label}`);
    $("progress").hidden = live;
    $("start").hidden = live;
    if (!live) {
      $("progress-fill").style.width = `${Math.min(100, (100 * s.elapsed_s) / s.duration_s)}%`;
      $("progress-text").textContent = `${Math.max(0, s.elapsed_s).toFixed(0)} / ${s.duration_s.toFixed(0)} s`;
      $("start").disabled = s.started || !s.ready;
      $("start").textContent = !s.ready ? "Checking proofs…" : s.finished ? "Finished" : s.started ? "Running" : "Start";
    }
    drawThroughput(s.throughput);
    drawBatches(s.batches);
    renderLog(s.log);
    renderPerGpu(s.per_gpu);
  } catch (err) {
    console.error(err);
  }
}

$("start").addEventListener("click", () => fetch("/api/start", { method: "POST" }).then(poll));
$("reset").addEventListener("click", () => {
  fetch("/api/reset", { method: "POST" }).then(poll);
  for (const o of Object.values(outcomes)) o.textContent = "";
});
$("detail-close").addEventListener("click", () => $("detail").close());
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t === tab);
    $("view-dashboard").hidden = tab.dataset.view !== "dashboard";
    $("view-limits").hidden = tab.dataset.view !== "limits";
  });
}

renderCards();
poll();
setInterval(poll, 700);
