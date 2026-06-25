# ANOVA Tukey

Aplicacion web para pegar tablas desde Excel, correr analisis por grupos y descargar un Excel con:

- ANOVA
- Tukey HSD
- LSD Fisher
- resumen por tratamiento
- detalle de comparaciones por pares

## Identidad visual

La interfaz usa una estetica corporativa Bayer, con el logo local `bayer-logo.jpg`, base blanca, azul/verde Bayer y acentos violetas.

## Alcance del analisis

La app permite elegir:

- **Por localidad**: cada localidad se analiza por separado.
- **Por protocolo**: se juntan las localidades y no se usan como corte estadistico.
- **Ambas**: el Excel incluye las dos lecturas.

Cuando se elige **Ambas**, el Excel agrega columnas de trazabilidad al final de la hoja `results`, por ejemplo:

- `analysis_scope`
- `analysis_basis`
- `location_analysis_note`
- `group_key`

## Orden del exportable

La hoja `results` conserva primero las columnas originales pegadas por la usuaria, en el mismo orden. Las columnas calculadas por el analisis se agregan despues.

## Testigos por `se_name_mod`

Si existe la columna `se_name_mod`, la app:

- la marca automaticamente como agrupamiento
- detecta cada variable
- pregunta si el testigo se incluye o no en el analisis
- para `Fitotoxicidad (%)` y `Eficacia (%)`, el default es excluir testigo
- el testigo excluido sigue apareciendo en el Excel descargable

## Como usarlo

1. Pega la tabla copiada desde Excel o CSV.
2. Toca **Cargar / Previsualizar**.
3. Selecciona columna de valores, tratamientos y localidad.
4. Marca las columnas de grupo.
5. Define el alcance: por localidad, por protocolo o ambas.
6. Ajusta reglas de testigo si aparece `se_name_mod`.
7. Toca **Ejecutar y descargar Excel**.
8. Ingresa el nombre del analisis.

## Backend

Endpoint principal:

- `POST /analyze`

El frontend usa `API_BASE` en `app.js`.

Comando sugerido para hostear el backend FastAPI:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Para Render, este repo incluye `.python-version` con Python 3.11.9. Si Render igualmente usa Python 3.14, `requirements.txt` usa versiones de pandas/scipy compatibles con Python 3.14 y el backend calcula ANOVA/Tukey/LSD con SciPy, sin depender de statsmodels.

Archivos que deben subirse al repo:

- `.python-version`
- `render.yaml`
- `Dockerfile`
- `.dockerignore`
- `app.py`
- `requirements.txt`
- `runtime.txt`
- `index.html`
- `styles.css`
- `app.js`
- `bayer-logo.jpg`
- `README.md`

Si Render sigue usando Python 3.14 aunque exista `PYTHON_VERSION=3.11.9`, crear el servicio como **Docker** usando el `Dockerfile` incluido. Ese archivo fija `python:3.11.9-slim` y evita que pandas intente compilarse con Python 3.14.
