const API_BASE = "https://cp-analisis1.onrender.com";

const $ = (id) => document.getElementById(id);

const paste = $("paste");
const btnParse = $("btnParse");
const btnAnalyze = $("btnAnalyze");
const btnClear = $("btnClear");
const status = $("status");
const preview = $("preview");

const valueColSel = $("valueCol");
const treatmentColSel = $("treatmentCol");
const groupColsBox = $("groupCols");
const alphaInput = $("alpha");
const progressWrap = $("progressWrap");
const progressBar = $("progressBar");
const progressText = $("progressText");
const seNameModPanel = $("seNameModPanel");
const seNameModRulesBox = $("seNameModRules");

let currentRows = [];
let currentCols = [];
let selectedGroupCols = new Set();
let seNameModValues = [];

function resetProgress() {
  progressBar.style.width = "0%";
  progressText.textContent = "";
  progressWrap.style.display = "none";
}

function showProgress() {
  progressWrap.style.display = "block";
}

function updateProgress(progress, current, total) {
  const safeProgress = Number.isFinite(progress) ? progress : 0;
  const safeCurrent = Number.isFinite(current) ? current : 0;
  const safeTotal = Number.isFinite(total) ? total : 0;

  progressBar.style.width = `${safeProgress}%`;
  progressText.textContent =
    safeTotal > 0
      ? `Procesando ${safeCurrent}/${safeTotal} grupos (${safeProgress}%)`
      : `Preparando análisis... (${safeProgress}%)`;
}

function parseTable(text) {
  const raw = text.trim();
  if (!raw) return { cols: [], rows: [] };

  const firstLine = raw.split(/\r?\n/)[0];
  let delim = "\t";

  if (!firstLine.includes("\t")) {
    delim = firstLine.includes(",")
      ? ","
      : firstLine.includes(";")
      ? ";"
      : "\t";
  }

  const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (!lines.length) return { cols: [], rows: [] };

  const header = lines[0].split(delim).map((h) => h.trim());
  const cols = header;

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(delim);
    const obj = {};
    cols.forEach((c, idx) => {
      obj[c] = (parts[idx] ?? "").trim();
    });
    rows.push(obj);
  }

  return { cols, rows };
}

function renderPreview(cols, rows, maxRows = 30) {
  preview.innerHTML = "";
  if (!cols.length) return;

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");

  cols.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    trh.appendChild(th);
  });

  thead.appendChild(trh);
  preview.appendChild(thead);

  const tbody = document.createElement("tbody");

  rows.slice(0, maxRows).forEach((r) => {
    const tr = document.createElement("tr");

    cols.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = r[c] ?? "";
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  preview.appendChild(tbody);
}

function fillSelect(selectEl, cols, preferredName) {
  selectEl.innerHTML = "";

  cols.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    selectEl.appendChild(opt);
  });

  if (cols.includes(preferredName)) {
    selectEl.value = preferredName;
  }
}

function normSeName(value) {
  return String(value ?? "").trim();
}

function shouldExcludeControlByDefault(seName) {
  const clean = normSeName(seName).toLowerCase();
  return clean === "fitotoxicidad (%)" || clean === "eficacia (%)";
}

function uniqueSeNameModValues(rows) {
  const out = [];
  const seen = new Set();

  rows.forEach((row) => {
    const value = normSeName(row.se_name_mod);
    if (!value) return;
    const key = value.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push(value);
  });

  return out.sort((a, b) => a.localeCompare(b, "es", { sensitivity: "base" }));
}

function renderSeNameModRules(rows, cols) {
  seNameModRulesBox.innerHTML = "";
  seNameModValues = [];

  if (!cols.includes("se_name_mod")) {
    seNameModPanel.style.display = "none";
    return;
  }

  seNameModValues = uniqueSeNameModValues(rows);
  if (!seNameModValues.length) {
    seNameModPanel.style.display = "none";
    return;
  }

  seNameModPanel.style.display = "block";

  seNameModValues.forEach((name) => {
    const defaultExclude = shouldExcludeControlByDefault(name);

    const card = document.createElement("div");
    card.className = "seRule";
    card.dataset.seName = name;

    const titleBox = document.createElement("div");
    const title = document.createElement("div");
    title.className = "seRuleName";
    title.textContent = name;

    const subtitle = document.createElement("div");
    subtitle.className = "seRuleDefault";
    subtitle.textContent = defaultExclude
      ? "Default: testigo NO incluido en el análisis."
      : "Default: testigo incluido en el análisis.";

    titleBox.appendChild(title);
    titleBox.appendChild(subtitle);

    const includeBox = document.createElement("div");
    const includeLabel = document.createElement("label");
    includeLabel.textContent = "¿Incluye testigo?";
    const includeSelect = document.createElement("select");
    includeSelect.className = "seIncludeSelect";
    includeSelect.innerHTML = `
      <option value="yes">Sí, analizarlo con el resto</option>
      <option value="no">No, excluirlo del análisis</option>
    `;
    includeSelect.value = defaultExclude ? "no" : "yes";
    includeBox.appendChild(includeLabel);
    includeBox.appendChild(includeSelect);

    const controlBox = document.createElement("div");
    const controlLabel = document.createElement("label");
    controlLabel.textContent = "Treatment testigo";
    const controlInput = document.createElement("input");
    controlInput.className = "seControlInput";
    controlInput.type = "text";
    controlInput.value = "1";
    controlInput.placeholder = "1";
    controlInput.disabled = includeSelect.value !== "no";
    controlBox.appendChild(controlLabel);
    controlBox.appendChild(controlInput);

    includeSelect.addEventListener("change", () => {
      controlInput.disabled = includeSelect.value !== "no";
      if (includeSelect.value === "no" && !controlInput.value.trim()) {
        controlInput.value = "1";
      }
    });

    card.appendChild(titleBox);
    card.appendChild(includeBox);
    card.appendChild(controlBox);
    seNameModRulesBox.appendChild(card);
  });
}

function collectSeNameModRules() {
  const rules = {};
  const cards = Array.from(document.querySelectorAll(".seRule"));

  cards.forEach((card) => {
    const name = card.dataset.seName;
    const includeSelect = card.querySelector(".seIncludeSelect");
    const controlInput = card.querySelector(".seControlInput");

    if (!name || !includeSelect || !controlInput) return;

    rules[name] = {
      include_control: includeSelect.value === "yes",
      control_treatment: (controlInput.value || "1").trim() || "1",
    };
  });

  return rules;
}

function renderGroupChips(cols, exclude = []) {
  groupColsBox.innerHTML = "";
  selectedGroupCols = new Set();

  cols.forEach((c) => {
    if (exclude.includes(c)) return;

    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = c;

    if (c === "se_name_mod") {
      selectedGroupCols.add(c);
      chip.classList.add("on");
      chip.title = "Se marca automáticamente para analizar cada se_name_mod por separado.";
    }

    chip.addEventListener("click", () => {
      if (selectedGroupCols.has(c)) {
        selectedGroupCols.delete(c);
        chip.classList.remove("on");
      } else {
        selectedGroupCols.add(c);
        chip.classList.add("on");
      }
    });

    groupColsBox.appendChild(chip);
  });
}

btnClear.addEventListener("click", () => {
  paste.value = "";
  currentRows = [];
  currentCols = [];
  selectedGroupCols = new Set();

  preview.innerHTML = "";
  valueColSel.innerHTML = "";
  treatmentColSel.innerHTML = "";
  groupColsBox.innerHTML = "";
  seNameModRulesBox.innerHTML = "";
  seNameModPanel.style.display = "none";
  seNameModValues = [];

  btnAnalyze.disabled = true;
  status.textContent = "Tabla limpiada.";
  resetProgress();
});

btnParse.addEventListener("click", () => {
  try {
    const { cols, rows } = parseTable(paste.value);
    currentCols = cols;
    currentRows = rows;

    if (!cols.length || !rows.length) {
      status.textContent =
        "No se detectaron datos. Revisá que haya encabezados y filas.";
      btnAnalyze.disabled = true;
      renderPreview([], []);
      resetProgress();
      return;
    }

    fillSelect(valueColSel, cols, "assessment_value");
    fillSelect(treatmentColSel, cols, "treatment");

    const exclude = [valueColSel.value, treatmentColSel.value];
    renderGroupChips(cols, exclude);
    renderSeNameModRules(rows, cols);

    renderPreview(cols, rows);
    status.textContent = `Tabla cargada: ${rows.length} filas, ${cols.length} columnas.`;
    btnAnalyze.disabled = false;
    resetProgress();
  } catch (e) {
    status.textContent =
      "Error al parsear la tabla. Probá pegar desde Excel (TSV) o CSV.";
    btnAnalyze.disabled = true;
    console.error(e);
    resetProgress();
  }
});

[valueColSel, treatmentColSel].forEach((sel) => {
  sel.addEventListener("change", () => {
    if (!currentCols.length) return;
    const exclude = [valueColSel.value, treatmentColSel.value];
    renderGroupChips(currentCols, exclude);
    renderSeNameModRules(currentRows, currentCols);
  });
});

btnAnalyze.addEventListener("click", async () => {
  if (!currentRows.length) return;

  const analysisName = prompt(
    "Nombre del análisis (se usará en el Excel y en el nombre del archivo):",
    "ANOVA_Tukey_LSD"
  );

  if (!analysisName || !analysisName.trim()) {
    status.textContent = "Cancelado: se requiere un nombre de análisis.";
    return;
  }

  btnAnalyze.disabled = true;
  btnParse.disabled = true;
  btnClear.disabled = true;

  resetProgress();
  showProgress();
  updateProgress(0, 0, 0);
  status.textContent = "Iniciando análisis...";

  const payload = {
    rows: currentRows,
    value_col: valueColSel.value,
    treatment_col: treatmentColSel.value,
    group_cols: Array.from(selectedGroupCols),
    alpha: Number(alphaInput.value || 0.05),
    analysis_name: analysisName.trim(),
    se_name_mod_col: currentCols.includes("se_name_mod") ? "se_name_mod" : "",
    se_name_mod_control_rules: collectSeNameModRules(),
  };

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }

    const { job_id } = await res.json();

    const poll = setInterval(async () => {
      try {
        const s = await fetch(`${API_BASE}/status/${job_id}`);
        if (!s.ok) {
          throw new Error(`No se pudo consultar el estado. HTTP ${s.status}`);
        }

        const data = await s.json();

        updateProgress(
          Number(data.progress || 0),
          Number(data.current || 0),
          Number(data.total || 0)
        );

        if (data.status === "done") {
          clearInterval(poll);

          progressBar.style.width = "100%";
          progressText.textContent = "Análisis finalizado. Descargando archivo...";
          status.textContent = "Descargando resultado...";

          const file = await fetch(`${API_BASE}/download/${job_id}`);
          if (!file.ok) {
            throw new Error(`No se pudo descargar el archivo. HTTP ${file.status}`);
          }

          const blob = await file.blob();
          const url = URL.createObjectURL(blob);

          const safe =
            analysisName.trim().replace(/[^\w \-]/g, "_").trim() || "analysis";

          const a = document.createElement("a");
          a.href = url;
          a.download = `${safe}_anova_tukey_lsd.xlsx`;
          document.body.appendChild(a);
          a.click();
          a.remove();

          URL.revokeObjectURL(url);
          status.textContent = "Listo. Se descargó el Excel con ANOVA + Tukey + LSD Fisher.";
        }

        if (data.status === "error") {
          clearInterval(poll);
          status.textContent = `Error: ${data.error || "falló el análisis."}`;
          progressText.textContent = "El análisis terminó con error.";
        }
      } catch (err) {
        clearInterval(poll);
        console.error(err);
        status.textContent = "Error consultando el progreso del análisis.";
        progressText.textContent = "No se pudo continuar el seguimiento del job.";
      } finally {
        if (
          progressText.textContent.includes("error") ||
          status.textContent.startsWith("Listo.") ||
          status.textContent.startsWith("Error")
        ) {
          btnAnalyze.disabled = false;
          btnParse.disabled = false;
          btnClear.disabled = false;
        }
      }
    }, 1000);
  } catch (e) {
    console.error(e);
    status.textContent = "Error al iniciar el análisis en el backend.";
    progressText.textContent = "No se pudo crear el job.";
    btnAnalyze.disabled = false;
    btnParse.disabled = false;
    btnClear.disabled = false;
  }
});

resetProgress();
