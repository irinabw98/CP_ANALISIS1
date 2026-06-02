<div align="center">

# 💜 CP ANÁLISIS
### ANOVA + Tukey + LSD Fisher por grupos
### Con análisis por localidad, por protocolo o ambos

![Python](https://img.shields.io/badge/Python-3.11-7F3FBF?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-8A2BE2?style=for-the-badge&logo=fastapi&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ANOVA%20%2B%20PostHoc-6A0DAD?style=for-the-badge)
![Render](https://img.shields.io/badge/Render-Online-A855F7?style=for-the-badge)
![Excel](https://img.shields.io/badge/Export-Excel-00B5E2?style=for-the-badge)

Aplicación web para pegar tablas desde Excel, correr **ANOVA por grupos** y descargar un Excel con:

**ANOVA + Tukey HSD + Fisher LSD**

</div>

---

## Cambios de esta versión

Esta versión replica una estética tipo **CP Correlación / Irina Labs**: fondo violeta, índice lateral, bloques numerados, tarjetas de configuración y Excel exportado con encabezados celeste Bayer.

También agrega el nuevo selector de alcance del análisis:

- **Por localidad:** cada localidad se analiza por separado.
- **Por protocolo:** se juntan todas las localidades y el modelo no discrimina por localidad.
- **Ambas:** el Excel incluye dos lecturas. Una por localidad y otra por protocolo.

Cuando se elige **Ambas**, el Excel agrega columnas de trazabilidad:

| columna | uso |
|---|---|
| `analysis_scope` | indica si la fila corresponde a `Por localidad` o `Por protocolo` |
| `analysis_basis` | indica `por_localidad` o `por_protocolo` |
| `location_analysis_note` | aclara cómo fue analizada esa línea |
| `group_key` | muestra el corte estadístico usado para ese resultado |

Además, se agrega una hoja `analysis_scope_readme` explicando cómo interpretar esas columnas.

---

## Regla de localidad

La app ahora pregunta qué columna corresponde a localidad. Intenta detectar automáticamente nombres como:

- `localidad`
- `location`
- `loc`
- `site`
- `trial_site`
- `lugar`

Si se analiza **por localidad**, esa columna se suma al corte del análisis.

Si se analiza **por protocolo**, esa columna se conserva en la tabla descargada, pero no se usa para separar el análisis.

---

## Regla del testigo por `se_name_mod`

Si existe la columna `se_name_mod`, la app:

- la marca automáticamente como agrupamiento
- lista cada valor diferente de `se_name_mod`
- pregunta si el testigo se incluye o no en el análisis
- permite indicar cuál es el treatment testigo
- trae por defecto el treatment testigo `1`
- para `Fitotoxicidad (%)` y `Eficacia (%)`, el default es **No incluir testigo en el análisis**
- el testigo excluido sigue apareciendo en el Excel descargable
- sus columnas estadísticas quedan como `-`
- queda marcado con `stats_status = excluded_control`

---

## Flujo de uso

1. Pegá la tabla copiada desde Excel o CSV.
2. Tocá **Cargar / Previsualizar**.
3. Elegí:
   - columna de valores
   - columna de tratamiento
   - columna de localidad
   - alpha
   - columnas de agrupamiento
4. Elegí si querés analizar por localidad, por protocolo o ambas.
5. Revisá la configuración por `se_name_mod` si aplica.
6. Tocá **Ejecutar y descargar Excel**.
7. Ingresá el nombre del análisis.
8. Descargá el Excel.

---

## Estructura del proyecto

```bash
repo/
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

Si este repo usa otro backend en Render, reemplazá esa URL por la URL nueva.
