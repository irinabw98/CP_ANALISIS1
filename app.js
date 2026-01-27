const API_BASE = "https://cp-analisis1.onrender.com";
const $ = (id) => document.getElementById(id);
const paste = $("paste");
const btnParse = $("btnParse");
const btnAnalyze = $("btnAnalyze");
const status = $("status");
const preview = $("preview");
const btnClear = $("btnClear");


const valueColSel = $("valueCol");
const treatmentColSel = $("treatmentCol");
const groupColsBox = $("groupCols");
const alphaInput = $("alpha");

let currentRows = [];
let currentCols = [];
let selectedGroupCols = new Set();

function parseTable(text) {
  const raw = text.trim();
  if (!raw) return { cols: [], rows: [] };

  const firstLine = raw.split(/\r?\n/)[0];
  let delim = "\t";
  if (!firstLine.includes("\t")) {
    delim = firstLine.includes(",") ? "," : (firstLine.includes(";") ? ";" : "\t");
  }

  const lines = raw.split(/\r?\n/).filter(l => l.trim().length > 0);
  const header = lines[0].split(delim).map(h => h.trim());
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
  cols.forEach(c => {
    const th = document.createElement("th");
    th.textContent = c;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  preview.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.slice(0, maxRows).forEach(r => {
    const tr = document.createElement("tr");
    cols.forEach(c => {
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
  cols.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    selectEl.appendChild(opt);
  });
  if (cols.includes(preferredName)) selectEl.value = preferredName;
}

function renderGroupChips(cols, exclude = []) {
  groupColsBox.innerHTML = "";
  selectedGroupCols = new Set();

  cols.forEach(c => {
    if (exclude.includes(c)) return;

    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = c;

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

btnParse.addEventListener("click", () => {
  try {
    const { cols, rows } = parseTable(paste.value);
    currentCols = cols;
    currentRows = rows;

    if (!cols.length || !rows.length) {
      status.textContent = "No se detectaron datos (revisá que haya encabezados y filas).";
      btnAnalyze.disabled = true;
      renderPreview([], []);
      return;
    }

    fillSelect(valueColSel, cols, "assessment_value");
    fillSelect(treatmentColSel, cols, "treatment");

    const exclude = [valueColSel.value, treatmentColSel.value];
    renderGroupChips(cols, exclude);

    renderPreview(cols, rows);
    status.textContent = `Tabla cargada: ${rows.length} filas, ${cols.length} columnas.`;
    btnAnalyze.disabled = false;
  } catch (e) {
    status.textContent = "Error al parsear la tabla. Probá pegar desde Excel (TSV) o CSV.";
    btnAnalyze.disabled = true;
  }
});

[valueColSel, treatmentColSel].forEach(sel => {
  sel.addEventListener("change", () => {
    if (!currentCols.length) return;
    const exclude = [valueColSel.value, treatmentColSel.value];
    renderGroupChips(currentCols, exclude);
  });
});

btnAnalyze.addEventListener("click", async () => {
  if (!currentRows.length) return;

  const analysisName = prompt("Nombre del análisis (se usará en el Excel y el nombre del archivo):", "ANOVA_Tukey");
  if (!analysisName || !analysisName.trim()) {
    status.textContent = "Cancelado: se requiere un nombre de análisis.";
    return;
  }

  btnAnalyze.disabled = true;
  status.textContent = "Ejecutando análisis en backend...";

  const payload = {
    rows: currentRows,
    value_col: valueColSel.value,
    treatment_col: treatmentColSel.value,
    group_cols: Array.from(selectedGroupCols),
    alpha: Number(alphaInput.value || 0.05),
    analysis_name: analysisName.trim()
  };

  try {
const res = await fetch(`${API_BASE}/analyze`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});
;

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const safe = analysisName.trim().replace(/[^\w \-]/g, "_").trim() || "analysis";
    const a = document.createElement("a");
    a.href = url;
    a.download = `${safe}_anova_tukey.xlsx`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    btnClear.addEventListener("click", () => {
  paste.value = "";
  currentRows = [];
  currentCols = [];
  selectedGroupCols = new Set();
  preview.innerHTML = "";
  valueColSel.innerHTML = "";
  treatmentColSel.innerHTML = "";
  groupColsBox.innerHTML = "";
  btnAnalyze.disabled = true;
  status.textContent = "Tabla limpiada.";
});


    URL.revokeObjectURL(url);
    status.textContent = "Listo. Se descargó el Excel con tu tabla + columnas de resultado.";
  } catch (e) {
    status.textContent = "Error ejecutando el análisis. Verificá que el backend esté corriendo en :8000.";
    console.error(e);
  } finally {
    btnAnalyze.disabled = false;
  }
});
