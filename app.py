from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

app = FastAPI(title="ANOVA + Tukey (grouped) API")

# Permitir llamadas desde tu GitHub Pages
ALLOWED_ORIGINS = [
    "https://irinabw98.github.io",
    "https://irinabw98.github.io/CP_ANALISIS1",
    "https://irinabw98.github.io/CP_ANALISIS1/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _compact_letter_display(tukey_df: pd.DataFrame, treatments: List[str]) -> Dict[str, str]:
    """
    Construye letras tipo 'a', 'ab', 'b' desde resultados pareados (Tukey HSD).
    Implementación greedy (MVP) que funciona bien en la mayoría de casos.
    """
    tset = list(treatments)
    idx = {t: i for i, t in enumerate(tset)}
    n = len(tset)

    # nodiff[i,j] = True si NO hay diferencia significativa entre i y j
    nodiff = np.eye(n, dtype=bool)

    for _, r in tukey_df.iterrows():
        g1 = str(r["group1"])
        g2 = str(r["group2"])
        rej = bool(r["reject"])
        if g1 in idx and g2 in idx:
            i, j = idx[g1], idx[g2]
            nodiff[i, j] = (not rej)
            nodiff[j, i] = (not rej)

    remaining = set(tset)
    letter_groups: List[Tuple[str, List[str]]] = []
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]

    letter_i = 0
    while remaining:
        # soporte > 26 letras: a..z, aa..az, ba..bz, ...
        if letter_i < 26:
            letter = letters[letter_i]
        else:
            prefix = letters[(letter_i // 26) - 1]
            suffix = letters[letter_i % 26]
            letter = prefix + suffix

        rem_list = list(remaining)

        # seed por mayor grado de no-diferencia dentro del conjunto restante
        degrees = []
        for t in rem_list:
            i = idx[t]
            deg = sum(nodiff[i, idx[x]] for x in rem_list)
            degrees.append((deg, t))
        degrees.sort(reverse=True)
        seed = degrees[0][1]

        # armar "clique" greedy: agregar cand si es nodiff con todos los del grupo actual
        group = [seed]
        for cand in rem_list:
            if cand == seed:
                continue
            ok = True
            for member in group:
                if not nodiff[idx[cand], idx[member]]:
                    ok = False
                    break
            if ok:
                group.append(cand)

        letter_groups.append((letter, group))
        for t in group:
            remaining.discard(t)

        letter_i += 1

    # Asignación final: un tratamiento recibe una letra si es nodiff con todos los miembros del grupo-letra
    out = {t: "" for t in tset}
    for letter, members in letter_groups:
        for t in tset:
            if all(nodiff[idx[t], idx[m]] for m in members):
                out[t] += letter

    # fallback por si alguno quedó vacío
    for t in tset:
        if out[t] == "":
            out[t] = "a"

    return out


def _run_group_analysis(
    gdf: pd.DataFrame,
    value_col: str,
    trt_col: str,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - summary: treatment, n, mean, sd, tukey_letters
      - anova: df, F, pvalue, df_resid
      - pairs: tabla pareada de tukey
    """
    gdf = gdf.copy()

    # limpiar NA
    gdf = gdf[[value_col, trt_col]].dropna()

    if gdf.empty:
        raise ValueError("Grupo sin datos luego de limpiar NA.")

    # asegurar numérico
    gdf[value_col] = pd.to_numeric(gdf[value_col], errors="coerce")
    gdf = gdf.dropna(subset=[value_col])

    if gdf.empty:
        raise ValueError("Grupo sin valores numéricos en assessment_value.")

    # treatment como string
    gdf[trt_col] = gdf[trt_col].astype(str)
    uniq_trt = sorted(gdf[trt_col].unique().tolist())

    if len(uniq_trt) < 2:
        raise ValueError("Grupo con menos de 2 tratamientos (no se puede ANOVA/Tukey).")

    # ANOVA (OLS)
    model = ols(f"Q('{value_col}') ~ C(Q('{trt_col}'))", data=gdf).fit()
    an = anova_lm(model, typ=2)

    factor_row = an.iloc[0]
    anova_out = pd.DataFrame([{
        "df": float(factor_row.get("df", np.nan)),
        "F": float(factor_row.get("F", np.nan)),
        "pvalue": float(factor_row.get("PR(>F)", np.nan)),
        "df_resid": float(an.iloc[1].get("df", np.nan)),
    }])

    # Tukey
    tuk = pairwise_tukeyhsd(endog=gdf[value_col].values, groups=gdf[trt_col].values, alpha=alpha)
    tuk_df = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    tuk_df["reject"] = tuk_df["reject"].astype(bool)
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        tuk_df[c] = pd.to_numeric(tuk_df[c], errors="coerce")

    letters = _compact_letter_display(tuk_df, uniq_trt)

    summary = (
        gdf.groupby(trt_col, dropna=False)[value_col]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
        .rename(columns={trt_col: "treatment"})
    )
    summary["tukey_letters"] = summary["treatment"].astype(str).map(letters)

    return summary, anova_out, tuk_df


def _make_group_key(row: pd.Series, group_cols: List[str]) -> str:
    parts = []
    for c in group_cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        parts.append(f"{c}={v}")
    return " | ".join(parts)


@app.post("/analyze")
def analyze(payload: Dict[str, Any] = Body(...)) -> StreamingResponse:
    """
    payload esperado:
    {
      "rows": [ {col: val, ...}, ... ],
      "value_col": "assessment_value",
      "treatment_col": "treatment",
      "group_cols": ["trial", "se_name", ...],
      "alpha": 0.05,
      "analysis_name": "Mi analisis"
    }
    """
    rows = payload.get("rows")
    value_col = payload.get("value_col", "assessment_value")
    treatment_col = payload.get("treatment_col", "treatment")
    group_cols = payload.get("group_cols", [])
    alpha = float(payload.get("alpha", 0.05))
    analysis_name = str(payload.get("analysis_name", "")).strip()

    if not isinstance(rows, list) or len(rows) == 0:
        raise HTTPException(status_code=400, detail="rows vacío o inválido.")
    if analysis_name == "":
        raise HTTPException(status_code=400, detail="analysis_name es requerido.")

    df = pd.DataFrame(rows)

    # validar columnas base
    missing = [c for c in [value_col, treatment_col] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas requeridas: {missing}")

    # validar group_cols
    for c in group_cols:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Columna de agrupamiento no existe: {c}")

    # normalizaciones
    df[treatment_col] = df[treatment_col].astype(str)

    # columnas requeridas por vos
    df["analysis_name"] = analysis_name
    if group_cols:
        df["group_key"] = df.apply(lambda r: _make_group_key(r, group_cols), axis=1)
    else:
        df["group_key"] = "ALL"

    # correr análisis por grupo
    summaries = []
    anovas = []
    pairs = []

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False, sort=False)
        for keys, gdf in grouped:
            key_dict = {}
            if isinstance(keys, tuple):
                for col, val in zip(group_cols, keys):
                    key_dict[col] = val
            else:
                key_dict[group_cols[0]] = keys

            try:
                s, a, p = _run_group_analysis(gdf, value_col, treatment_col, alpha)

                for col, val in key_dict.items():
                    s[col] = val
                    a[col] = val
                    p[col] = val

                summaries.append(s)
                anovas.append(a)
                pairs.append(p)
            except Exception as e:
                err = {"error": str(e)}
                for col, val in key_dict.items():
                    err[col] = val
                anovas.append(pd.DataFrame([err]))
    else:
        s, a, p = _run_group_analysis(df, value_col, treatment_col, alpha)
        summaries.append(s)
        anovas.append(a)
        pairs.append(p)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    anova_df = pd.concat(anovas, ignore_index=True) if anovas else pd.DataFrame()
    pairs_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

    # construir tabla final: "misma tabla" + resultados
    base_df = df.copy().rename(columns={treatment_col: "treatment"})
    if not summary_df.empty:
        merge_keys = (group_cols if group_cols else []) + ["treatment"]
        final_df = base_df.merge(summary_df, on=merge_keys, how="left")
    else:
        final_df = base_df

    # anexar ANOVA por grupo como columnas
    if not anova_df.empty:
        anova_cols = [c for c in ["df", "F", "pvalue", "df_resid", "error"] if c in anova_df.columns]
        if group_cols:
            a_small = anova_df[group_cols + anova_cols].copy()
            final_df = final_df.merge(a_small, on=group_cols, how="left", suffixes=("", "_anova"))
        else:
            for c in anova_cols:
                final_df[c] = anova_df.iloc[0][c] if len(anova_df) else np.nan

# reordenar: group_key primero, después las originales, después las agregadas
original_cols = list(pd.DataFrame(rows).columns)

front = ["group_key"] + [c for c in original_cols if c in final_df.columns and c != "group_key"]
added = [c for c in final_df.columns if c not in front]

final_df = final_df[front + added]


    # export excel
    output = io.BytesIO()
    safe_name = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in analysis_name).strip()
    if safe_name == "":
        safe_name = "analysis"

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="results")
        if not anova_df.empty:
            anova_df.to_excel(writer, index=False, sheet_name="anova_detail")
        if not pairs_df.empty:
            pairs_df.to_excel(writer, index=False, sheet_name="tukey_pairs_detail")

    output.seek(0)
    filename = f"{safe_name}_anova_tukey.xlsx"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/health")
def health():
    return {"status": "ok"}

