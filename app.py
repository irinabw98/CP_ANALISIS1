from __future__ import annotations

import io
import threading
from uuid import uuid4
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# IMPORTÁS TUS FUNCIONES EXISTENTES
from app import (
    _run_group_analysis,
    _to_numeric_series_strong,
)

app = FastAPI()

jobs: Dict[str, Dict[str, Any]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🚀 BACKGROUND PROCESS
# =========================

def run_analysis(job_id: str, payload: Dict[str, Any]):
    try:
        df = pd.DataFrame(payload["rows"])

        value_col = payload["value_col"]
        treatment_col = payload["treatment_col"]
        group_cols = payload["group_cols"]
        alpha = float(payload.get("alpha", 0.05))
        analysis_name = payload["analysis_name"]

        df["assessment_value_num"] = _to_numeric_series_strong(df[value_col])
        df = df.dropna(subset=["assessment_value_num"])

        if len(df) == 0:
            raise Exception("No hay datos numéricos válidos.")

        grouped = list(df.groupby(group_cols)) if group_cols else [("ALL", df)]

        total = len(grouped)

        jobs[job_id]["total"] = total

        summaries = []
        anovas = []
        pairs = []

        for i, (keys, gdf) in enumerate(grouped):

            try:
                s, a, p = _run_group_analysis(
                    gdf,
                    "assessment_value_num",
                    treatment_col,
                    alpha,
                )

                summaries.append(s)
                anovas.append(a)
                pairs.append(p)

            except Exception as e:
                print(f"Error en grupo {keys}: {e}")

            # 🔥 PROGRESO REAL
            jobs[job_id]["current"] = i + 1
            jobs[job_id]["progress"] = int(((i + 1) / total) * 100)

        summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        anova_df = pd.concat(anovas, ignore_index=True) if anovas else pd.DataFrame()
        pairs_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="summary")
            anova_df.to_excel(writer, index=False, sheet_name="anova")
            pairs_df.to_excel(writer, index=False, sheet_name="pairs")

        output.seek(0)

        jobs[job_id]["result"] = output
        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# =========================
# 🚀 ENDPOINTS
# =========================

@app.post("/analyze")
def analyze(payload: Dict[str, Any] = Body(...)):
    job_id = str(uuid4())

    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "current": 0,
        "total": 0,
        "result": None
    }

    thread = threading.Thread(target=run_analysis, args=(job_id, payload))
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job no encontrado")
    return jobs[job_id]


@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(404, "Job no encontrado")

    if job["status"] != "done":
        raise HTTPException(400, "No listo")

    return StreamingResponse(
        job["result"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="resultado.xlsx"'}
    )
