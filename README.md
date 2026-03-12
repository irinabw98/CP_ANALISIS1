<div align="center">

# 💜 CP_ANALISIS1
### ANOVA + Tukey + LSD Fisher (por grupos)

![Python](https://img.shields.io/badge/Python-3.11-7F3FBF?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-8A2BE2?style=for-the-badge&logo=fastapi&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ANOVA%20%2B%20PostHoc-6A0DAD?style=for-the-badge)
![Render](https://img.shields.io/badge/Render-Online-A855F7?style=for-the-badge)
![Excel](https://img.shields.io/badge/Export-Excel-C084FC?style=for-the-badge)

Aplicación web para pegar tablas desde Excel, correr **ANOVA por grupos** y devolver un archivo Excel con resultados de:

**ANOVA + Tukey HSD + Fisher LSD**

</div>

---

## 💡 ¿Qué hace esta app?

Esta herramienta permite:

- pegar una tabla copiada desde Excel o CSV
- elegir la columna de valores numéricos
- elegir la columna de tratamientos
- definir columnas de agrupamiento
- correr un **ANOVA independiente por cada grupo**
- calcular **dos post hoc en paralelo**:
  - **Tukey HSD**
  - **LSD Fisher**
- descargar un Excel con:
  - la tabla original enriquecida
  - resumen por tratamiento
  - detalle de ANOVA
  - detalle de comparaciones Tukey
  - detalle de comparaciones LSD Fisher

---

## 🧪 Flujo de análisis

La lógica general es:

1. Pegás la tabla
2. Elegís:
   - `value_col`
   - `treatment_col`
   - `group_cols`
   - `alpha`
3. El backend:
   - convierte la variable a numérica
   - arma los grupos
   - corre **ANOVA**
   - corre **Tukey**
   - corre **LSD Fisher protegido por ANOVA**
4. Se descarga un Excel con todos los resultados

---

## 📦 Estructura del proyecto

```bash
CP_ANALISIS1/
│
├── app.py             # Backend FastAPI
├── app.js             # Lógica frontend
├── index.html         # Interfaz principal
├── styles.css         # Estilos
├── requirements.txt   # Dependencias Python
├── runtime.txt        # Versión de Python en Render
└── README.md
