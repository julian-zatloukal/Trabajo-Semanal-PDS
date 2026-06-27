#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 2026

@author: julian

Descarga de temperatura para Buenos Aires desde NOAA / NCEI.

Buenos Aires NO esta en USCRN (red de EE.UU.), asi que se usa la base
global ISD (Integrated Surface Database), producto "global-hourly".
Resolucion ~15 min irregular (observaciones METAR del aeroparque).

    https://www.ncei.noaa.gov/data/global-hourly/access/{AÑO}/{ID}.csv

Estacion: AEROPARQUE JORGE NEWBERY (SABE), ID 87582099999.
Campo TMP: "+0228,1" -> +22.8 °C (decimas de grado) + codigo de calidad.
Faltante = +9999.

Se remuestrea a una rejilla uniforme de 30 min (fs = 48 muestras/dia),
requisito para filtrar digitalmente.
"""

import numpy as np
import pandas as pd

STATION = "87582099999"      # Aeroparque Jorge Newbery
YEAR = 2024                  # 2025 solo llega hasta el 24-ago -> se usa 2024 completo
GRID = "1h"                  # rejilla uniforme de salida (ISD es ~horario util)


def descargar_ba(station=STATION, year=YEAR, grid=GRID):
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"
    print(f"Descargando {url} ...")
    df = pd.read_csv(url, usecols=["DATE", "TMP"], parse_dates=["DATE"])

    # TMP = "valor,calidad" ; valor en decimas de °C, +9999 = faltante
    val = pd.to_numeric(df["TMP"].str.split(",").str[0], errors="coerce") / 10.0
    val[val >= 999.9] = np.nan
    s = pd.Series(val.values, index=df["DATE"]).sort_index()
    s = s[~s.index.duplicated(keep="first")]   # quita timestamps repetidos (METAR + SYNOP)

    # Rejilla uniforme: promedio por intervalo y luego interpolacion de huecos
    out = s.resample(grid).mean()
    faltan = int(out.isna().sum())
    out = out.interpolate(limit_direction="both")
    out = out.rename("TEMP").to_frame()
    out.index.name = "TIME"

    print(f"muestras: {len(out)} (fs = {out.index.freq}) | interpolados: {faltan} "
          f"({100*faltan/len(out):.2f}%)")
    print("rango:", out.index.min(), "->", out.index.max())
    return out


if __name__ == "__main__":
    df = descargar_ba()
    df.to_csv("TS7/analisis/buenos_aires_2024.csv")
    print(df["TEMP"].describe()[["min", "mean", "max"]].to_string())
    print("guardado TS7/buenos_aires_2024.csv")
