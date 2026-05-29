<div align="center">

# 💜 CP_ANALISIS1
### ANOVA + Tukey + LSD Fisher por grupos
### Con configuración de testigo por `se_name_mod`

![Python](https://img.shields.io/badge/Python-3.11-7F3FBF?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-8A2BE2?style=for-the-badge&logo=fastapi&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ANOVA%20%2B%20PostHoc-6A0DAD?style=for-the-badge)
![Render](https://img.shields.io/badge/Render-Online-A855F7?style=for-the-badge)
![Excel](https://img.shields.io/badge/Export-Excel-C084FC?style=for-the-badge)

Aplicación web para pegar tablas desde Excel, correr **ANOVA por grupos** y descargar un Excel con resultados de:

**ANOVA + Tukey HSD + Fisher LSD**

</div>

---

## ¿Qué hace esta versión?

Esta versión mantiene el flujo original y agrega una mejora específica para datos que traen la columna `se_name_mod`.

Cuando la app detecta `se_name_mod`:

- la marca automáticamente como columna de agrupamiento
- lista cada valor diferente de `se_name_mod`
- pregunta si el testigo se incluye o no en el análisis para cada variable
- si se elige “No”, permite indicar cuál es el tratamiento testigo
- el tratamiento testigo trae por defecto el valor `1`
- para `Fitotoxicidad (%)` y `Eficacia (%)`, el default es “No incluir testigo en el análisis”
- el testigo excluido no entra al ANOVA, Tukey ni LSD Fisher
- el testigo excluido sigue apareciendo en el Excel descargable
- las columnas estadísticas del testigo excluido se completan con `-`

---

## Flujo de uso

1. Pegá la tabla copiada desde Excel o CSV.
2. Tocá **Cargar / Previsualizar**.
3. Elegí:
   - columna de valores, por ejemplo `assessment_value`
   - columna de tratamientos, por ejemplo `treatment`
   - alpha, por defecto `0.05`
   - columnas de agrupamiento
4. Si existe `se_name_mod`, revisá la configuración de cada variable.
5. Tocá **Ejecutar y descargar Excel**.
6. Ingresá el nombre del análisis.
7. Descargá el Excel con los resultados.

---

## Regla del testigo

Si para una variable se indica que el testigo no se incluye:

- se filtra del cálculo estadístico solamente para esa `se_name_mod`
- no se elimina de la tabla final
- no participa en el ANOVA
- no participa en Tukey
- no participa en LSD Fisher
- en el Excel queda con `stats_status = excluded_control`
- sus columnas estadísticas quedan como `-`

Ejemplo:

| se_name_mod | treatment | ¿entra al análisis? |
|---|---:|---|
| Fitotoxicidad (%) | 1 | No |
| Fitotoxicidad (%) | 2 | Sí |
| Fitotoxicidad (%) | 3 | Sí |
| Altura | 1 | Sí |
| Altura | 2 | Sí |
| Altura | 3 | Sí |

---

## Estructura del proyecto

```bash
CP_ANALISIS1/
├── app.py
├── app.js
├── index.html
├── styles.css
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Backend

El backend está hecho con FastAPI y usa:

- pandas
- numpy
- scipy
- statsmodels
- openpyxl

Endpoints principales:

- `POST /analyze`
- `GET /status/{job_id}`
- `GET /download/{job_id}`
- `GET /health`
- `GET /version`

---

## Deploy sugerido

Frontend:

- GitHub Pages

Backend:

- Render

Recordá revisar en `app.js` la constante:

```js
const API_BASE = "https://cp-analisis1.onrender.com";
```

Si creás otro backend en Render, cambiá esa URL por la nueva.
