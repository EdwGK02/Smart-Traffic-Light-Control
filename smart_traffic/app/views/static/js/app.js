// Smart Traffic Light Control · Etapa 1 — lógica de la página
const $ = (sel) => document.querySelector(sel);

// De menor (verde) a mayor congestión (rojo), igual que las gráficas
const PALETTE = ["#2E7D32", "#7CB342", "#FBC02D", "#FB8C00", "#E53935", "#8E24AA", "#5D4037"];
const PLOT_CAPTIONS = {
  "01_seleccion_k.png": "Selección de k: codo (inercia) y silhouette",
  "02_momentos_por_hora.png": "Momento de la intersección en cada hora del día",
  "03_mapa_semanal.png": "Momento más frecuente por día y hora",
  "04_clusters_pca.png": "Clusters proyectados en 2D (PCA)",
};
let momentColors = {};   // nombre del momento -> color

function colorsFor(names) {
  const n = names.length;
  const pos = n <= 5 ? names.map((_, i) => Math.round((i * 4) / Math.max(n - 1, 1))) : names.map((_, i) => i);
  const map = {};
  names.forEach((name, i) => (map[name] = PALETTE[pos[i] % PALETTE.length]));
  return map;
}

async function api(url, options = {}) {
  const res = await fetch(url, options);
  let data;
  try { data = await res.json(); } catch { data = { ok: false, error: `Error ${res.status}` }; }
  if (!res.ok || data.ok === false) throw new Error(data.error || `Error ${res.status}`);
  return data;
}

function setMsg(el, text, type = "") {
  el.textContent = text;
  el.className = `msg ${type}`;
}

const fmt = (x, d = 2) => Number(x).toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x) => `${Math.round(Number(x) * 100)}%`;

// ------------------------------------------------------------- estado
async function loadStatus() {
  const data = await api("/api/status");
  const setDot = (id, on, onTxt, offTxt) => {
    $(`#${id}`).className = `dot ${on ? "on" : "off"}`;
    $(`#${id}-txt`).textContent = on ? onTxt : offTxt;
  };
  setDot("st-model", data.model_ready, "entrenado", "sin entrenar");
  setDot("st-kaggle", data.datasets.kaggle, "cargado", "falta archivo");
  setDot("st-bogota", data.datasets.bogota, "cargado", "falta archivo");

  if (data.summary) {
    const s = data.summary;
    $("#st-trained").textContent = `Último entrenamiento: ${s.trained_at.replace("T", " ")} · ${s.source}`;
    renderSummary(s);
  }
}

// ---------------------------------------------------------- resultados
function renderSummary(s) {
  $("#results").hidden = false;
  const m = s.metrics;
  momentColors = colorsFor(s.profiles.map((p) => p.moment));

  $("#metrics").innerHTML = [
    ["Clusters (k)", m.k, ""],
    ["Silhouette", fmt(m.silhouette, 3), "−1 a 1 · mayor es mejor"],
    ["Davies-Bouldin", fmt(m.davies_bouldin, 3), "menor es mejor"],
    ["Horas analizadas", Number(m.n_rows).toLocaleString("es-CO"), `${s.date_range[0]} → ${s.date_range[1]}`],
    ["Intersecciones", m.n_intersections, ""],
  ].map(([label, val, sub]) => `<div class="metric"><span>${label}</span><b>${val}</b>${sub ? `<small>${sub}</small>` : ""}</div>`).join("");

  $("#profiles tbody").innerHTML = s.profiles.map((p) => `
    <tr>
      <td>${badge(p.moment)}</td>
      <td class="num-cell">${fmt(p.pct_datos * 100, 1)}%</td>
      <td class="num-cell">${fmt(p.vehiculos_prom, 1)}</td>
      <td class="num-cell">${fmt(p.carga_relativa)}×</td>
      <td class="num-cell">${p.tendencia >= 0 ? "▲" : "▼"} ${fmt(Math.abs(p.tendencia), 3)}</td>
      <td class="num-cell">${pct(p.pct_fin_semana)}</td>
      <td>${p.horas_tipicas}</td>
    </tr>`).join("");

  const bust = encodeURIComponent(s.trained_at);
  // Primero las gráficas anchas (02, 03) y luego las dos pequeñas lado a lado (01, 04)
  const isWide = (f) => f.startsWith("02") || f.startsWith("03");
  const plots = [...s.plots.filter(isWide), ...s.plots.filter((f) => !isWide(f))];
  $("#plots").innerHTML = plots.map((f) => `
    <figure class="${isWide(f) ? "wide" : ""}">
      <img src="/plots/${f}?v=${bust}" alt="${PLOT_CAPTIONS[f] || f}" loading="lazy">
      <figcaption>${PLOT_CAPTIONS[f] || f}</figcaption>
    </figure>`).join("");

  const select = $("#junction-select");
  const current = select.value;
  select.innerHTML = Object.entries(s.intersections)
    .map(([id, mean]) => `<option value="${id}">${id} · prom. ${fmt(mean, 1)} veh/h</option>`).join("");
  if (current && s.intersections[current] !== undefined) select.value = current;
}

function badge(name) {
  const c = momentColors[name] || "#9ca3af";
  return `<span class="badge"><i style="background:${c}"></i>${name}</span>`;
}

// ------------------------------------------------------------- subir
$("#upload-form input[type=file]").addEventListener("change", (e) => {
  const f = e.target.files[0];
  $("#file-label").textContent = f ? `${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)` : "Elegir archivo…";
});

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#upload-msg");
  setMsg(msg, "Subiendo…");
  try {
    const data = await api("/api/upload", { method: "POST", body: new FormData(e.target) });
    setMsg(msg, `Guardado como data/raw/${data.saved_as}`, "ok");
    loadStatus();
  } catch (err) {
    setMsg(msg, err.message, "err");
  }
});

// ----------------------------------------------------------- entrenar
$("#train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#train-btn"), msg = $("#train-msg");
  const form = new FormData(e.target);
  btn.disabled = true;
  btn.textContent = "Entrenando…";
  setMsg(msg, "Probando valores de k y generando gráficas. Puede tardar un poco.");
  const t0 = performance.now();
  try {
    const data = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: form.get("source"), k: form.get("k") }),
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setMsg(msg, `Listo en ${secs} s · ${data.summary.metrics.k} momentos encontrados`, "ok");
    await loadStatus();
    $("#results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    setMsg(msg, err.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "Entrenar modelo";
  }
});

// ----------------------------------------------------------- predecir
$("#predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#predict-msg");
  const form = new FormData(e.target);
  setMsg(msg, "");
  try {
    const { result: r } = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        junction: form.get("junction"),
        datetime: form.get("datetime"),
        vehicles: form.get("vehicles"),
        prev: form.get("prev"),
      }),
    });
    renderPrediction(r);
  } catch (err) {
    setMsg(msg, err.message, "err");
  }
});

function renderPrediction(r) {
  const color = momentColors[r.moment] || "#9ca3af";
  const when = new Date(r.timestamp);
  const day = when.toLocaleDateString("es-CO", { weekday: "long", day: "numeric", month: "short", year: "numeric" });
  const hour = when.toLocaleTimeString("es-CO", { hour: "2-digit", minute: "2-digit" });

  const dists = Object.entries(r.distances || {}).sort((a, b) => a[1] - b[1]);
  const max = Math.max(...dists.map((d) => d[1]), 1);
  // similitud = barra más larga para el momento más cercano
  const bars = dists.map(([name, d]) => `
    <div class="bar">
      <span>${badge(name)}</span>
      <div class="bar__track"><div class="bar__fill" style="width:${(100 * (1 - d / (max * 1.15))).toFixed(0)}%;background:${momentColors[name] || "#9ca3af"}"></div></div>
      <span class="bar__val">${fmt(d, 2)}</span>
    </div>`).join("");

  $("#predict-result").innerHTML = `
    <div class="result__head">
      <div class="light-big" style="background:${color};--glow:${color}"></div>
      <div>
        <div class="result__meta">${r.intersection_id} · ${day} · ${hour}</div>
        <div class="result__moment">${r.moment}</div>
        <div class="result__meta">${r.vehicles} vehículos · carga ${fmt(r.relative_load)}× su promedio</div>
      </div>
    </div>
    <div class="result__meta">Distancia a cada momento (menor = más parecido)</div>
    <div class="bars">${bars}</div>`;
}

// ------------------------------------------------------ ampliar gráficas
document.addEventListener("click", (e) => {
  if (e.target.matches(".plots img")) {
    const box = document.createElement("div");
    box.className = "lightbox";
    box.innerHTML = `<img src="${e.target.src}" alt="">`;
    box.addEventListener("click", () => box.remove());
    document.body.appendChild(box);
  }
});

// Fecha por defecto: ahora, redondeada a la hora
(() => {
  const d = new Date();
  d.setMinutes(0, 0, 0);
  const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  $("#predict-form [name=datetime]").value = local;
})();

loadStatus().catch((err) => setMsg($("#train-msg"), err.message, "err"));
