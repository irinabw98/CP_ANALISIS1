# ANOVA + Tukey (por grupos) - MVP

Este proyecto permite:
- Pegar una tabla (copiada desde Excel) en el frontend.
- Elegir columnas para segmentar/agrupamiento (por ejemplo: trial, se_name, timing, part, etc.).
- Ejecutar ANOVA dentro de cada grupo y clasificar con Tukey HSD (alpha por defecto 0.05).
- Descargar un Excel con **la misma tabla** + columnas agregadas:
  - `group_key` (concatenación de columnas de agrupamiento)
  - `analysis_name`
  - `n`, `mean`, `sd`, `tukey_letters` (por tratamiento dentro de cada grupo)
  - `df`, `F`, `pvalue`, `df_resid` (ANOVA por grupo, replicado por fila)

## Ejecutar backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Ejecutar frontend

Abrí `frontend/index.html` en el navegador.
