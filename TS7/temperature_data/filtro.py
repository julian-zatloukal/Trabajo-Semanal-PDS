#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 2026

@author: julian

Filtro de prueba aplicado a la temperatura anual (Ithaca NY, 2025, 5 min).

Se usa un pasa-bajos Butterworth de fase cero (filtfilt) para quitar el
ciclo diario y el ruido meteorologico, dejando la tendencia estacional.

  fs    = 1 muestra / 5 min  = 288 muestras/dia
  ciclo diario  -> periodo 288 muestras
  corte elegido -> periodo de 5 dias (deja pasar lo mas lento que el clima)

Produce dos imagenes:
  - temperatura_2025.png            (señal cruda)
  - temperatura_2025_filtrada.png   (cruda + filtrada superpuesta)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

CSV = "TS7/analisis/ithaca_2025.csv"

# --- Carga ---
df = pd.read_csv(CSV, parse_dates=["TIME"], index_col="TIME")
# El filtrado no admite NaN: se interpolan los pocos faltantes (-9999 -> NaN).
temp = df["TEMP"].interpolate(limit_direction="both").to_numpy()
t = df.index

# --- Diseño del filtro ---
fs = 288.0                      # muestras por dia
periodo_corte_dias = 5.0        # quita lo mas rapido que ~5 dias (incluye ciclo diario)
fc = 1.0 / periodo_corte_dias   # frecuencia de corte [ciclos/dia]
orden = 4
b, a = signal.butter(orden, fc, btype="low", fs=fs)

# Fase cero (no desplaza la señal en el tiempo)
temp_filt = signal.filtfilt(b, a, temp)

# Segundo pasa-bajos, mas agresivo (corte mas largo -> mas suave)
periodo_corte_dias_2 = 30.0
fc2 = 1.0 / periodo_corte_dias_2
b2, a2 = signal.butter(orden, fc2, btype="low", fs=fs)
temp_filt2 = signal.filtfilt(b2, a2, temp)

# --- Imagen 1: señal cruda ---
plt.figure(figsize=(14, 5))
plt.plot(t, temp, lw=0.5, color="tab:red")
plt.title("Temperatura del aire - Ithaca 2025 (cruda, USCRN 5 min)")
plt.xlabel("Fecha"); plt.ylabel("Temperatura [°C]")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("TS7/analisis/temperatura_2025.png", dpi=120)

# --- Imagen 2: cruda + filtrada ---
plt.figure(figsize=(14, 5))
plt.plot(t, temp, lw=0.4, color="0.7", label="Cruda (5 min)")
plt.plot(t, temp_filt, lw=1.8, color="tab:blue",
         label=f"Pasa-bajos (corte ~{periodo_corte_dias:.0f} dias)")
plt.plot(t, temp_filt2, lw=2.2, color="tab:orange",
         label=f"Pasa-bajos mas suave (corte ~{periodo_corte_dias_2:.0f} dias)")
plt.title("Temperatura del aire - Ithaca 2025: filtro pasa-bajos")
plt.xlabel("Fecha"); plt.ylabel("Temperatura [°C]")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("TS7/analisis/temperatura_2025_filtrada.png", dpi=120)

# --- Imagen 3: solo filtro de 30 dias, con fondo por estacion ---
# Estaciones hemisferio norte (Ithaca, NY). Bordes abruptos via axvspan.
estaciones = [
    ("2025-01-01", "2025-03-20", "Invierno", "#cfe8ff"),
    ("2025-03-20", "2025-06-20", "Primavera", "#d7f5d0"),
    ("2025-06-20", "2025-09-22", "Verano",    "#fff2b3"),
    ("2025-09-22", "2025-12-21", "Otoño",     "#f5d9b8"),
    ("2025-12-21", "2026-01-01", "Invierno",  "#cfe8ff"),
]

from matplotlib.patches import Patch

plt.figure(figsize=(14, 5))
for ini, fin, nombre, color in estaciones:
    plt.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=color, zorder=0)
linea, = plt.plot(t, temp_filt2, lw=2.4, color="tab:orange",
                  label=f"Pasa-bajos (corte ~{periodo_corte_dias_2:.0f} dias)", zorder=3)
plt.title("Temperatura del aire - Ithaca 2025: tendencia estacional")
plt.xlabel("Fecha"); plt.ylabel("Temperatura [°C]")
plt.xlim(t.min(), t.max())

# Leyenda: la linea + un parche de color por estacion (sin repetir invierno)
parches = [Patch(facecolor=c, label=n)
           for n, c in dict((n, c) for _, _, n, c in estaciones).items()]
plt.legend(handles=[linea] + parches, loc="lower center", ncol=5, framealpha=0.9)
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("TS7/analisis/temperatura_2025_estaciones.png", dpi=120)

plt.show()
print("Imagenes guardadas: temperatura_2025.png, temperatura_2025_filtrada.png y temperatura_2025_estaciones.png")
