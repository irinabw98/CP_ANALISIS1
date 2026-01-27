# 🌙 CP_ANALISIS1 — ANOVA + Tukey (por grupos)

Una app web para convertir una tabla pegada desde Excel en un análisis estadístico **listo para descargar**.

👉 Pegás datos → elegís cómo agrupar → corrés **ANOVA + Tukey (α = 0.05)** → descargás un Excel con tu misma tabla + columnas de resultado.

---

## ✨ ¿Qué vas a poder hacer acá?

- **Pegar tu tabla directamente** (TSV desde Excel o CSV).
- Elegir:
  - **Columna de valores** (por defecto `assessment_value`)
  - **Columna de tratamientos** (`treatment`)
  - **Columnas para agrupar** (las que “parten” el análisis: protocolo, trial, momento, se_name, etc.)
- Ejecutar el análisis por cada grupo definido.
- Descargar un Excel con:
  - Tu tabla original
  - `group_key` (identificador del grupo)
  - `analysis_name` (nombre del análisis)
  - `assessment_value_num` (la versión numérica real del valor, robusta a coma decimal)
  - Estadísticos por tratamiento: `n`, `mean`, `sd`
  - Letras de Tukey: `tukey_letters`
  - ANOVA por grupo: `F`, `pvalue`, `df`, `df_resid`
  - Hojas extra con detalle: `anova_detail` y `tukey_pairs_detail`

---

## 🧠 Cómo pensar el flujo (mentalidad de experimento)

Tu tabla tiene muchas columnas, pero el análisis necesita 3 conceptos:

1. **Valor**: lo que mediste → `assessment_value`
2. **Tratamiento**: lo que comparás → `treatment`
3. **Grupo**: el contexto del experimento → columnas que segmentan el análisis  
   (por ejemplo: `trial + se_name + timing + part_rated_code`)

Cada combinación de “grupo” genera un ANOVA independiente.

---

## ✅ Requisitos de la tabla

La tabla debe incluir al menos:

- `assessment_value`
- `treatment`

Y puede incluir cualquier cantidad de columnas extra (no hay límite práctico).

📌 Tip: pegá la tabla desde Excel (copiar y pegar) para que se detecte como TSV automáticamente.

---

## 🖥️ Tecnologías

**Frontend**
- HTML + CSS + JavaScript (GitHub Pages)

**Backend**
- Python + FastAPI
- Pandas / NumPy
- Statsmodels (ANOVA + Tukey)
- Export a Excel con OpenPyXL

**Deploy**
- Frontend: GitHub Pages
- Backend: Render

---

## 🌐 Uso online

- Frontend: GitHub Pages del repo
- Backend: Render (endpoint `/analyze`)

---

## 🧪 Estado del proyecto

MVP funcional ✅  
Próximos upgrades:
- Estética mejorada (violeta/lila)
- Validaciones y mensajes más “humanos”
- Gráficos
- Historial / trazabilidad de análisis
