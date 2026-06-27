#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 2026

@author: julian

Grafica de la temperatura del aire (dato crudo, sin filtrar).
Fuente: NOAA USCRN sub-horaria (5 min). Ver download_noaa.py.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Archivo a graficar (cambiar por palestine_2026-05.csv, austin_..., etc.)
CSV = "TS7/analisis/ithaca_2025.csv"

# Carga: indice temporal, columna TEMP en °C
df = pd.read_csv(CSV, parse_dates=["TIME"], index_col="TIME")

fs_dia = 288  # muestras por dia (1 cada 5 min)

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["TEMP"], lw=0.7, color="tab:red")
plt.title(f"Temperatura del aire - {CSV.split('/')[-1]}  (USCRN, 5 min)")
plt.xlabel("Fecha")
plt.ylabel("Temperatura [°C]")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guarda la figura en el directorio TS7
plt.savefig("TS7/analisis/temperatura_2025.png", dpi=120)
plt.show()
