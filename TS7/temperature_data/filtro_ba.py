#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 2026

@author: julian

Filtro y graficas para la temperatura anual de Buenos Aires (Aeroparque,
2024, ISD horario). Equivalente al analisis de Ithaca pero:
  - fs = 24 muestras/dia (horario, no 5 min)
  - estaciones del HEMISFERIO SUR (verano en Dic-Mar, invierno en Jun-Sep)

Pasa-bajos Butterworth de fase cero (filtfilt). Produce tres imagenes:
  - ba_2024.png             (cruda)
  - ba_2024_filtrada.png    (cruda + corte 5 dias + corte 30 dias)
  - ba_2024_estaciones.png  (solo corte 30 dias, fondo por estacion)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.patches import Patch

CSV = "TS7/analisis/buenos_aires_2024.csv"

df = pd.read_csv(CSV, parse_dates=["TIME"], index_col="TIME")
temp = df["TEMP"].interpolate(limit_direction="both").to_numpy()
t = df.index

# --- Diseño de los filtros ---
fs = 24.0                       # muestras por dia (horario)
orden = 4
b, a = signal.butter(orden, 1.0 / 5.0, btype="low", fs=fs)    # corte ~5 dias
temp_filt = signal.filtfilt(b, a, temp)
periodo_corte_dias_2 = 30.0
b2, a2 = signal.butter(orden, 1.0 / periodo_corte_dias_2, btype="low", fs=fs)
temp_filt2 = signal.filtfilt(b2, a2, temp)

# --- Imagen 1: señal cruda ---
plt.figure(figsize=(14, 5))
plt.plot(t, temp, lw=0.5, color="tab:red")
plt.title("Temperatura del aire - Buenos Aires 2024 (cruda, ISD horario)")
plt.xlabel("Fecha"); plt.ylabel("Temperatura [°C]")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("TS7/analisis/ba_2024.png", dpi=120)

# --- Imagen 2: cruda + dos filtros ---
plt.figure(figsize=(14, 5))
plt.plot(t, temp, lw=0.4, color="0.7", label="Cruda (horaria)")
plt.plot(t, temp_filt, lw=1.8, color="tab:blue", label="Pasa-bajos (corte ~5 dias)")
plt.plot(t, temp_filt2, lw=2.2, color="tab:orange",
         label=f"Pasa-bajos mas suave (corte ~{periodo_corte_dias_2:.0f} dias)")
plt.title("Temperatura del aire - Buenos Aires 2024: filtro pasa-bajos")
plt.xlabel("Fecha"); plt.ylabel("Temperatura [°C]")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("TS7/analisis/ba_2024_filtrada.png", dpi=120)

# --- Imagen 3: solo 30 dias, fondo por estacion (HEMISFERIO SUR) ---
estaciones = [
    ("2024-01-01", "2024-03-20", "Verano",    "#fff2b3"),
    ("2024-03-20", "2024-06-20", "Otoño",     "#f5d9b8"),
    ("2024-06-20", "2024-09-22", "Invierno",  "#cfe8ff"),
    ("2024-09-22", "2024-12-21", "Primavera", "#d7f5d0"),
    ("2024-12-21", "2025-01-01", "Verano",    "#fff2b3"),
]

plt.figure(figsize=(14, 5))
for ini, fin, nombre, color in estaciones:
    plt.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=color, zorder=0)
linea, = plt.plot(t, temp_filt2, lw=2.4, color="tab:orange",
                  label=f"Pasa-bajos (corte ~{periodo_corte_dias_2:.0f} dias)", zorder=3)
plt.title("Temperatura del aire - Buenos Aires 2024: tendencia estacional")
plt.xlabel("Fecha"); plt.ylabel("Temperatura [°C]")
plt.xlim(t.min(), t.max())
parches = [Patch(facecolor=c, label=n)
           for n, c in dict((n, c) for _, _, n, c in estaciones).items()]
plt.legend(handles=[linea] + parches, loc="lower center", ncol=5, framealpha=0.9)
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("TS7/analisis/ba_2024_estaciones.png", dpi=120)

plt.show()
print("Imagenes guardadas: ba_2024.png, ba_2024_filtrada.png y ba_2024_estaciones.png")
