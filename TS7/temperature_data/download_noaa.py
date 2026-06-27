#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga de temperatura sub-horaria (cada 5 minutos) desde NOAA / NCEI.

Fuente: U.S. Climate Reference Network (USCRN), producto "subhourly01".
Es el dato de temperatura de mayor resolucion que publica NOAA: 1 muestra
cada 5 minutos -> fs = 1/300 Hz = 288 muestras/dia.

No requiere token ni registro. Un archivo .txt por estacion y por año:

    https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/{AÑO}/
        CRNS0101-05-{AÑO}-{ESTACION}.txt

Formato (23 columnas, separadas por espacios, sin encabezado):
  col  9 -> AIR_TEMPERATURE [°C]   (valor faltante = -9999.0)
  cols 4,5 -> LST_DATE (YYYYMMDD) + LST_TIME (HHmm), hora local estandar.

@author: julian
"""

import numpy as np
import pandas as pd

# Estacion y rango de años. Buscar estaciones listando el directorio del año:
#   https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2023/
STATION = "TX_Austin_33_NW"
YEARS = range(2021, 2024)  # 2021, 2022, 2023
BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01"

# Nombres de las 23 columnas (ver HEADERS.txt del directorio)
COLS = [
    "WBANNO", "UTC_DATE", "UTC_TIME", "LST_DATE", "LST_TIME", "CRX_VN",
    "LONGITUDE", "LATITUDE", "AIR_TEMPERATURE", "PRECIPITATION",
    "SOLAR_RADIATION", "SR_FLAG", "SURFACE_TEMPERATURE", "ST_TYPE", "ST_FLAG",
    "RELATIVE_HUMIDITY", "RH_FLAG", "SOIL_MOISTURE_5", "SOIL_TEMPERATURE_5",
    "WETNESS", "WET_FLAG", "WIND_1_5", "WIND_FLAG",
]


def _leer_año(station, year):
    url = f"{BASE}/{year}/CRNS0101-05-{year}-{station}.txt"
    print(f"Descargando {url} ...")
    df = pd.read_csv(url, sep=r"\s+", header=None, names=COLS,
                     dtype={"LST_DATE": str, "LST_TIME": str})
    # timestamp en hora local estandar -> ciclo diario solar bien alineado
    df["TIME"] = pd.to_datetime(df["LST_DATE"] + df["LST_TIME"], format="%Y%m%d%H%M")
    return df[["TIME", "AIR_TEMPERATURE"]]


def descargar_temperatura(station=STATION, years=YEARS):
    """Devuelve un DataFrame indexado por tiempo con temp del aire [°C] cada 5 min."""
    df = pd.concat((_leer_año(station, y) for y in years), ignore_index=True)

    # Sentinela de faltante -> NaN
    df.loc[df["AIR_TEMPERATURE"] <= -9999.0, "AIR_TEMPERATURE"] = np.nan

    df = df.set_index("TIME").sort_index()

    # Rejilla uniforme de 5 min (requisito para filtrar digitalmente).
    df = df.asfreq("5min")
    df["AIR_TEMPERATURE"] = df["AIR_TEMPERATURE"].interpolate(limit_direction="both")

    return df.rename(columns={"AIR_TEMPERATURE": "TEMP"})


if __name__ == "__main__":
    df = descargar_temperatura()
    df[["TEMP"]].to_csv("TS7/analisis/temperatura_5min.csv")
    print(df["TEMP"].describe())
    print(f"\nMuestras: {len(df)}  (fs = 1/300 Hz = 288 muestras/dia)")
    print("Guardado en TS7/temperatura_5min.csv")
