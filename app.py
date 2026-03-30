from __future__ import annotations

import io
import threading
import time
from itertools import combinations
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from scipy.stats import t as student_t
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

app = FastAPI(title="ANOVA + Tukey + LSD Fisher (grouped) API")

ALLOWED_ORIGINS = [
    "https://irinabw98.github.io",
    "https://irinabw98.github.io/CP_ANALISIS1",
    "https://irinabw98.github.io/CP_ANALISIS1/",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

MAX_ROWS = 100000
MAX_GROUPS = 5000
MAX_TREATMENTS_PER_GROUP = 80
JOB_TTL_SECONDS = 60 * 60 * 6


def _now_ts() -> float:
    return time.time()


def _touch_job(job_id: str) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["updated_at"] = _now_ts()


def _set_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
            jobs[job_id]["updated_at"] = _now_ts()


def _cleanup_old_jobs() -> None:
    cutoff = _now_ts() - JOB_TTL_SECONDS
    to_delete: List[str] = []

    with jobs_lock:
        for job_id, payload in jobs.items():
            if float(payload.get("updated_at", 0)) < cutoff:
                to_delete.append(job_id)

        for job_id in to_delete:
            del jobs[job_id]


def _compact_letter_display(pairs_df: pd.DataFrame, treatments: List[str]) -> Dict[str, str]:
    tset = list(treatments)
    idx = {t: i for i, t in enumerate(tset)}
    n = len(tset)

    nodiff = np.eye(n, dtype=bool)

    for _, r in pairs_df.iterrows():
        g1 = str(r["group1"])
        g2 = str(r["group2"])
        rej = bool(r["reject"])
        if g1 in idx and g2 in idx:
            i, j = idx[g1], idx[g2]
            nodiff[i, j] = not rej
            nodiff[j, i] = not rej

    remaining = set(tset)
    letter_groups: List[Tuple[str, List[str]]] = []
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]

    letter_i = 0
    while remaining:
        if letter_i < 26:
            letter = letters[letter_i]
        else:
            prefix = letters[(letter_i // 26) - 1]
            suffix = letters[letter_i % 26]
            letter = prefix + suffix

        rem_list = list(remaining)

        degrees = []
        for t in rem_list:
            i = idx[t]
            deg = sum(nodiff[i, idx[x]] for x in rem_list)
            degrees.append((deg, t))
        degrees.sort(reverse=True)
        seed = degrees[0][1]

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

    out = {t: "" for t in tset}
    for letter, members in letter_groups:
        for t in tset:
            if all(nodiff[idx[t], idx[m]] for m in members):
                out[t] += letter

    for t in tset:
        if out[t] == "":
            out[t] = "a"

    return out


def _relabel_letters_by_mean(
    summary_df: pd.DataFrame,
    letters_map: Dict[str, str],
) -> Dict[str, str]:
    if summary_df.empty or not letters_map:
        return letters_map

    df = summary_df[["treatment", "mean"]].copy()
    df["treatment"] = df["treatment"].astype(str)
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)

    seen = []
    for _, row in df.iterrows():
        trt = row["treatment"]
        raw_letters = str(letters_map.get(trt, "")).strip().lower()
        for ch in raw_letters:
            if ch not in seen:
                seen.append(ch)

    if not seen:
        return {k: "A" for k in letters_map.keys()}

    new_symbols = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    remap = {}

    for i, old_sym in enumerate(seen):
        if i < 26:
            remap[old_sym] = new_symbols[i]
        else:
            prefix = new_symbols[(i // 26) - 1]
            suffix = new_symbols[i % 26]
            remap[old_sym] = prefix + suffix

    out = {}
    for trt, raw_letters in letters_map.items():
        raw_letters = str(raw_letters).strip().lower()

        rebuilt = []
        for ch in raw_letters:
            if ch in remap and remap[ch] not in rebuilt:
                rebuilt.append(remap[ch])

        out[str(trt)] = "".join(rebuilt) if rebuilt else "A"

    return out


def _to_numeric_series_strong(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(r"[^0-9,\.\-]", "", regex=True)

    def _one(x: str):
        x = str(x).strip()
        if x in ("", "-", ".", ","):
            return np.nan

        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        else:
            if "," in x:
                x = x.replace(".", "")
                x = x.replace(",", ".")

        try:
            return float(x)
        except Exception:
            return np.nan

    return s2.apply(_one)


def _reject_between(pairs_df: pd.DataFrame, a: str, b: str) -> bool:
    row = pairs_df[
        ((pairs_df["group1"] == a) & (pairs_df["group2"] == b)) |
        ((pairs_df["group1"] == b) & (pairs_df["group2"] == a))
    ]
    if row.empty:
        return False
    return bool(row.iloc[0]["reject"])


def _make_class_from_pairs(summary_df: pd.DataFrame, pairs_df: pd.DataFrame, class_col_name: str) -> pd.Series:
    df = summary_df.sort_values("mean", ascending=False).reset_index(drop=True)

    if df.empty:
        return pd.Series(dtype=str)

    current = "A"
    classes = [current]

    for i in range(1, len(df)):
        t_prev = str(df.loc[i - 1, "treatment"])
        t_curr = str(df.loc[i, "treatment"])

        if _reject_between(pairs_df, t_prev, t_curr):
            current = chr(ord(current) + 1)

        classes.append(current)

    df[class_col_name] = classes
    return df.set_index("treatment")[class_col_name]


def _run_tukey(
    gdf: pd.DataFrame,
    value_col_num: str,
    trt_col: str,
    alpha: float,
) -> pd.DataFrame:
    tuk = pairwise_tukeyhsd(
        endog=gdf[value_col_num].values,
        groups=gdf[trt_col].values,
        alpha=alpha,
    )
    tuk_df = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    tuk_df["reject"] = tuk_df["reject"].astype(bool)

    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in tuk_df.columns:
            tuk_df[c] = pd.to_numeric(tuk_df[c], errors="coerce")

    tuk_df["method"] = "tukey"
    tuk_df["pvalue"] = pd.to_numeric(tuk_df.get("p-adj", np.nan), errors="coerce")
    tuk_df = tuk_df.rename(columns={"p-adj": "p_adj"})

    return tuk_df


def _run_lsd_fisher(
    gdf: pd.DataFrame,
    value_col_num: str,
    trt_col: str,
    alpha: float,
    model,
    anova_pvalue: float,
) -> pd.DataFrame:
    means = (
        gdf.groupby(trt_col, dropna=False)[value_col_num]
        .agg(n="count", mean="mean")
        .reset_index()
        .rename(columns={trt_col: "treatment"})
    )

    uniq_trt = means["treatment"].astype(str).tolist()
    n_map = means.set_index("treatment")["n"].to_dict()
    mean_map = means.set_index("treatment")["mean"].to_dict()

    mse = float(model.mse_resid)
    df_resid = float(model.df_resid)

    rows = []
    for g1, g2 in combinations(uniq_trt, 2):
        mean1 = float(mean_map[g1])
        mean2 = float(mean_map[g2])
        n1 = float(n_map[g1])
        n2 = float(n_map[g2])

        se = np.sqrt(mse * ((1.0 / n1) + (1.0 / n2))) if n1 > 0 and n2 > 0 else np.nan
        diff = mean2 - mean1

        if np.isnan(se) or se == 0 or np.isnan(df_resid) or df_resid <= 0:
            t_stat = np.nan
            p_val = np.nan
            t_crit = np.nan
            lsd_value = np.nan
            lower = np.nan
            upper = np.nan
            reject = False
        else:
            t_stat = abs(diff) / se
            p_val = 2 * (1 - student_t.cdf(abs(t_stat), df_resid))
            t_crit = student_t.ppf(1 - (alpha / 2), df_resid)
            lsd_value = t_crit * se
            lower = diff - lsd_value
            upper = diff + lsd_value
            reject = bool(abs(diff) > lsd_value)

        if np.isnan(anova_pvalue) or anova_pvalue >= alpha:
            reject = False

        rows.append({
            "group1": g1,
            "group2": g2,
            "mean1": mean1,
            "mean2": mean2,
            "meandiff": diff,
            "se": se,
            "t_stat": t_stat,
            "pvalue": p_val,
            "lower": lower,
            "upper": upper,
            "lsd_value": lsd_value,
            "reject": reject,
            "method": "lsd_fisher",
        })

    lsd_df = pd.DataFrame(rows)
    if lsd_df.empty:
        lsd_df = pd.DataFrame(columns=[
            "group1", "group2", "mean1", "mean2", "meandiff", "se", "t_stat",
            "pvalue", "lower", "upper", "lsd_value", "reject", "method"
        ])

    return lsd_df


def _run_group_analysis(
    gdf: pd.DataFrame,
    value_col_num: str,
    trt_col: str,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gdf = gdf.copy()

    gdf = gdf[[value_col_num, trt_col]].dropna()
    if gdf.empty:
        raise ValueError("Grupo sin datos luego de limpiar NA.")

    gdf[value_col_num] = pd.to_numeric(gdf[value_col_num], errors="coerce")
    gdf = gdf.dropna(subset=[value_col_num])
    if gdf.empty:
        raise ValueError(f"Grupo sin valores numéricos en {value_col_num}.")

    gdf[trt_col] = gdf[trt_col].astype(str)
    uniq_trt = sorted(gdf[trt_col].unique().tolist())
    if len(uniq_trt) < 2:
        raise ValueError("Grupo con menos de 2 tratamientos (no se puede ANOVA/post hoc).")

    if len(uniq_trt) > MAX_TREATMENTS_PER_GROUP:
        raise ValueError(
            f"Grupo con demasiados tratamientos ({len(uniq_trt)}). "
            f"Máximo permitido: {MAX_TREATMENTS_PER_GROUP}."
        )

    model = ols(f"Q('{value_col_num}') ~ C(Q('{trt_col}'))", data=gdf).fit()
    an = anova_lm(model, typ=2)

    factor_row = an.iloc[0]
    p_anova = float(factor_row.get("PR(>F)", np.nan))

    anova_out = pd.DataFrame([{
        "df": float(factor_row.get("df", np.nan)),
        "F": float(factor_row.get("F", np.nan)),
        "pvalue": p_anova,
        "df_resid": float(an.iloc[1].get("df", np.nan)),
    }])

    summary = (
        gdf.groupby(trt_col, dropna=False)[value_col_num]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
        .rename(columns={trt_col: "treatment"})
    )

    tukey_pairs_df = _run_tukey(gdf, value_col_num, trt_col, alpha)
    tukey_letters = _compact_letter_display(tukey_pairs_df, uniq_trt)
    tukey_letters = _relabel_letters_by_mean(summary[["treatment", "mean"]].copy(), tukey_letters)

    summary["tukey_letters"] = (
        summary["treatment"].astype(str).map(tukey_letters).fillna("A").str.upper()
    )

    tukey_class_map = _make_class_from_pairs(
        summary[["treatment", "mean"]].copy(),
        tukey_pairs_df,
        "tukey_class",
    )
    summary["tukey_class"] = (
        summary["treatment"].astype(str).map(tukey_class_map).fillna("A").str.upper()
    )

    lsd_pairs_df = _run_lsd_fisher(
        gdf=gdf,
        value_col_num=value_col_num,
        trt_col=trt_col,
        alpha=alpha,
        model=model,
        anova_pvalue=p_anova,
    )

    lsd_letters = _compact_letter_display(lsd_pairs_df, uniq_trt) if not lsd_pairs_df.empty else {t: "a" for t in uniq_trt}
    lsd_letters = _relabel_letters_by_mean(summary[["treatment", "mean"]].copy(), lsd_letters)

    summary["lsd_letters"] = (
        summary["treatment"].astype(str).map(lsd_letters).fillna("A").str.upper()
    )

    lsd_class_map = (
        _make_class_from_pairs(
            summary[["treatment", "mean"]].copy(),
            lsd_pairs_df,
            "lsd_class",
        )
        if not lsd_pairs_df.empty
        else pd.Series({t: "A" for t in uniq_trt})
    )
    summary["lsd_class"] = (
        summary["treatment"].astype(str).map(lsd_class_map).fillna("A").str.upper()
    )

    pairs_df = pd.concat([tukey_pairs_df, lsd_pairs_df], ignore_index=True, sort=False)

    return summary, anova_out, pairs_df


def _make_group_key(row: pd.Series, group_cols: List[str]) -> str:
    parts = []
    for c in group_cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        parts.append(f"{c}={v}")
    return " | ".join(parts)


def _build_excel_output(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    treatment_col: str,
    group_cols: List[str],
    analysis_name: str,
) -> io.BytesIO:
    base_df = df.copy().rename(columns={treatment_col: "treatment"})

    if not summary_df.empty:
        merge_keys = (group_cols if group_cols else []) + ["treatment"]
        final_df = base_df.merge(summary_df, on=merge_keys, how="left")
    else:
        final_df = base_df

    if not anova_df.empty:
        anova_cols = [c for c in ["df", "F", "pvalue", "df_resid", "error"] if c in anova_df.columns]
        if group_cols:
            a_small = anova_df[group_cols + anova_cols].copy()
            final_df = final_df.merge(a_small, on=group_cols, how="left", suffixes=("", "_anova"))
        else:
            for c in anova_cols:
                final_df[c] = anova_df.iloc[0][c] if len(anova_df) else np.nan

    if "group_key" in final_df.columns:
        gk = final_df.pop("group_key")
        final_df.insert(0, "group_key", gk)

    preferred_front = ["analysis_name", "assessment_value_num", "assessment_value_x1"]
    for col in reversed(preferred_front):
        if col in final_df.columns:
            s_col = final_df.pop(col)
            final_df.insert(1, col, s_col)

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="results")

        if not summary_df.empty:
            summary_df.to_excel(writer, index=False, sheet_name="summary_by_treatment")

        if not anova_df.empty:
            anova_df.to_excel(writer, index=False, sheet_name="anova_detail")

        if not pairs_df.empty:
            tukey_df = pairs_df[pairs_df["method"] == "tukey"].copy()
            lsd_df = pairs_df[pairs_df["method"] == "lsd_fisher"].copy()

            if not tukey_df.empty:
                tukey_df.to_excel(writer, index=False, sheet_name="tukey_pairs_detail")
            if not lsd_df.empty:
                lsd_df.to_excel(writer, index=False, sheet_name="lsd_pairs_detail")

    output.seek(0)
    return output


def _run_analysis_job(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        rows = payload.get("rows")
        value_col = payload.get("value_col", "assessment_value")
        treatment_col = payload.get("treatment_col", "treatment")
        group_cols = payload.get("group_cols", [])
        alpha = float(payload.get("alpha", 0.05))
        analysis_name = str(payload.get("analysis_name", "")).strip()

        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("rows vacío o inválido.")
        if analysis_name == "":
            raise ValueError("analysis_name es requerido.")

        if len(rows) > MAX_ROWS:
            raise ValueError(f"Dataset demasiado grande. Máximo permitido: {MAX_ROWS} filas.")

        df = pd.DataFrame(rows)

        missing = [c for c in [value_col, treatment_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        for c in group_cols:
            if c not in df.columns:
                raise ValueError(f"Columna de agrupamiento no existe: {c}")

        df[treatment_col] = df[treatment_col].astype(str)

        _set_job(job_id, progress=2, message="Convirtiendo valores a numéricos...")

        df["assessment_value_num"] = _to_numeric_series_strong(df[value_col])
        df["assessment_value_x1"] = df["assessment_value_num"] * 1.0
        df = df.dropna(subset=["assessment_value_num"])

        if df.empty:
            raise ValueError(f"No quedaron filas con valores numéricos en '{value_col}'.")

        df["analysis_name"] = analysis_name
        if group_cols:
            df["group_key"] = df.apply(lambda r: _make_group_key(r, group_cols), axis=1)
        else:
            df["group_key"] = "ALL"

        summaries: List[pd.DataFrame] = []
        anovas: List[pd.DataFrame] = []
        pairs: List[pd.DataFrame] = []

        value_col_num = "assessment_value_num"

        if group_cols:
            grouped_items = list(df.groupby(group_cols, dropna=False, sort=False))
        else:
            grouped_items = [("ALL", df)]

        total_groups = len(grouped_items)

        if total_groups > MAX_GROUPS:
            raise ValueError(f"Demasiados grupos ({total_groups}). Máximo permitido: {MAX_GROUPS}.")

        _set_job(
            job_id,
            total=total_groups,
            current=0,
            progress=5,
            status="running",
            message=f"Procesando 0/{total_groups} grupos..."
        )

        for idx_group, (keys, gdf) in enumerate(grouped_items, start=1):
            key_dict: Dict[str, Any] = {}

            if group_cols:
                if isinstance(keys, tuple):
                    for col, val in zip(group_cols, keys):
                        key_dict[col] = val
                else:
                    key_dict[group_cols[0]] = keys

            try:
                s, a, p = _run_group_analysis(gdf, value_col_num, treatment_col, alpha)

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

            progress = int(5 + (idx_group / max(total_groups, 1)) * 85)

            _set_job(
                job_id,
                current=idx_group,
                total=total_groups,
                progress=min(progress, 95),
                message=f"Procesando {idx_group}/{total_groups} grupos..."
            )

        _set_job(job_id, progress=96, message="Armando resultados...")

        summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        anova_df = pd.concat(anovas, ignore_index=True) if anovas else pd.DataFrame()
        pairs_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

        output = _build_excel_output(
            df=df,
            summary_df=summary_df,
            anova_df=anova_df,
            pairs_df=pairs_df,
            treatment_col=treatment_col,
            group_cols=group_cols,
            analysis_name=analysis_name,
        )

        safe_name = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in analysis_name).strip()
        if safe_name == "":
            safe_name = "analysis"

        filename = f"{safe_name}_anova_tukey_lsd.xlsx"

        _set_job(
            job_id,
            status="done",
            progress=100,
            message="Análisis finalizado.",
            result_bytes=output.getvalue(),
            filename=filename,
        )

    except Exception as e:
        _set_job(
            job_id,
            status="error",
            progress=100,
            message="El análisis terminó con error.",
            error=str(e),
        )


@app.post("/analyze")
def analyze(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...),
):
    _cleanup_old_jobs()

    job_id = str(uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "current": 0,
            "total": 0,
            "message": "Job creado.",
            "error": None,
            "result_bytes": None,
            "filename": None,
            "created_at": _now_ts(),
            "updated_at": _now_ts(),
        }

    background_tasks.add_task(_run_analysis_job, job_id, payload)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job no encontrado")

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "current": job["current"],
            "total": job["total"],
            "message": job.get("message"),
            "error": job.get("error"),
            "filename": job.get("filename"),
        }


@app.get("/download/{job_id}")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job no encontrado")

        if job["status"] != "done":
            raise HTTPException(status_code=400, detail="El archivo todavía no está listo.")

        result_bytes = job.get("result_bytes")
        filename = job.get("filename") or "analysis_anova_tukey_lsd.xlsx"

    if result_bytes is None:
        raise HTTPException(status_code=500, detail="No se encontró el archivo generado.")

    output = io.BytesIO(result_bytes)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "2026-03-30-jobs-progress-download-v2"}
