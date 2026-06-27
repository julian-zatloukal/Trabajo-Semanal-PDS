#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 09:26:51 2026

@author: julian
"""

import numpy as np
import pandas as pd
from scipy import signal as sig
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Muestreo
fs = 1000
nyq = fs / 2

# Carga de señal
mat_struct = sio.loadmat('./signals/ecg.mat')
ecg = mat_struct['ecg_lead'].flatten().astype(float)
t = np.arange(len(ecg)) / fs

# -----------------------------------------------------------------------------
# Plantilla de diseño (filtro pasa-banda)
#
# Se elige un pasa-banda porque el enunciado pide dos cosas, una por cada borde:
#
#   1) Nivel isoeléctrico nulo  -> borde inferior (ws1=0.1, wp1=1 Hz):
#      atenúa la deriva de línea de base (respiración/electrodos, < 1 Hz),
#      llevando la señal a cero entre latidos.
#
#   2) Trazos suaves            -> borde superior (wp2=35, ws2=45 Hz):
#      elimina el ruido de alta frecuencia (EMG y red de 50 Hz) sin recortar
#      el contenido útil del QRS (la energía del ECG está por debajo de ~35 Hz).
#
# Tolerancias: 1 dB de ripple en banda de paso, 40 dB de atenuación en rechazo.
# -----------------------------------------------------------------------------
ws1, wp1, wp2, ws2 = 0.1, 1.0, 35.0, 45.0
gpass, gstop = 1.0, 40.0

# Configuración de filtro, Butterworth
N, wn = sig.buttord([wp1, wp2], [ws1, ws2], gpass, gstop, fs=fs)
sos = sig.butter(N, wn, btype='bandpass', output='sos', fs=fs)
ecg_f = sig.sosfiltfilt(sos, ecg)

# Configuración de filtro, Chebyshev tipo I
ripple_db = 1.0
N_cheb, wn_cheb = sig.cheb1ord([wp1, wp2], [ws1, ws2], gpass, gstop, fs=fs)
sos_cheb = sig.cheby1(N_cheb, ripple_db, wn_cheb, btype='bandpass', output='sos', fs=fs)
ecg_f_cheb = sig.sosfiltfilt(sos_cheb, ecg)

# -----------------------------------------------------------------------------
# Filtros FIR sobre la misma plantilla
#
# Se respeta el mismo pasa-banda (mismos bordes y tolerancias). El borde inferior
# (0.1-1 Hz) impone la transición más angosta y, por lo tanto, la cantidad de
# coeficientes necesarios. La estimo con la fórmula de Kaiser a partir de la
# atenuación de rechazo (gstop) y el ancho de transición normalizado a Nyquist.
# -----------------------------------------------------------------------------
trans_width = min(wp1 - ws1, ws2 - wp2)        # transición más angosta [Hz]
numtaps, beta = sig.kaiserord(gstop, trans_width / nyq)
numtaps |= 1                                   # fuerzo orden impar (FIR tipo I)

# Método de ventanas (firwin con ventana de Kaiser)
# Las frecuencias de corte se ubican en el centro de cada banda de transición.
cutoff = [(ws1 + wp1) / 2, (wp2 + ws2) / 2]
fir_win = sig.firwin(numtaps, cutoff, window=('kaiser', beta),
                     pass_zero=False, fs=fs)
ecg_f_win = sig.filtfilt(fir_win, 1.0, ecg)

# Método de cuadrados mínimos (firls)
# Se definen bandas CONTIGUAS, describiendo las transiciones como rampas
# lineales. Si se dejaran huecos "don't care" entre el paso y el rechazo,
# firls no controla esas zonas y, con tantos coeficientes, aparecen ripples
# enormes (sobre-picos de decenas de dB) en la banda de transición 35-45 Hz.
bands   = [0, ws1, ws1, wp1, wp1, wp2, wp2, ws2, ws2, nyq]
desired = [0, 0,   0,   1,   1,   1,   1,   0,   0,   0]
fir_ls = sig.firls(numtaps, bands, desired, fs=fs)
ecg_f_ls = sig.filtfilt(fir_ls, 1.0, ecg)

# -----------------------------------------------------------------------------
# d) Evaluación de desempeño
#
# Se compara cada señal filtrada contra la cruda en regiones de interés:
#   1) CON interferencia  -> debe eliminar deriva de línea de base / ruido.
#   2) SIN interferencia   -> debe ser inocuo (no deformar el latido).
#
# Todos los filtros se aplicaron con filtrado bidireccional (filtfilt /
# sosfiltfilt), de fase cero: no introducen retardo, por lo que demora = 0 y
# las señales quedan alineadas con la cruda sin necesidad de compensación.
#
# Estas figuras se crean primero a propósito: matplotlib trae al frente la
# última figura creada, así que dejándolas al inicio quedan al fondo de la pila
# y se ven al final, después de las respuestas en frecuencia.
# -----------------------------------------------------------------------------
cant_muestras = len(ecg)
demora = 0

filtrados = [
    (ecg_f,      'Butterworth'),
    (ecg_f_cheb, 'Chebyshev I'),
    (ecg_f_win,  'FIR ventanas (Kaiser)'),
    (ecg_f_ls,   'FIR cuadrados mínimos'),
]

# Regiones CON interferencia
regs_con_ruido = (
    [4000, 5500],     # muestras
    [10000, 11000],   # muestras
)
for ii in regs_con_ruido:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    plt.figure()
    plt.plot(zoom_region, ecg[zoom_region], label='ECG crudo', lw=2)
    for y, nombre in filtrados:
        plt.plot(zoom_region, y[zoom_region + demora], label=nombre, lw=1)
    plt.title(f'Región CON interferencia: muestras {int(ii[0])} a {int(ii[1])}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Adimensional')
    plt.legend()
    plt.grid(True)

# Regiones SIN interferencia
regs_sin_ruido = (
    np.array([5, 5.2]) * 60 * fs,    # minutos -> muestras
    np.array([12, 12.4]) * 60 * fs,  # minutos -> muestras
    np.array([15, 15.2]) * 60 * fs,  # minutos -> muestras
)
for ii in regs_sin_ruido:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    plt.figure()
    plt.plot(zoom_region, ecg[zoom_region], label='ECG crudo', lw=2)
    for y, nombre in filtrados:
        plt.plot(zoom_region, y[zoom_region + demora], label=nombre, lw=1)
    plt.title(f'Región SIN interferencia: muestras {int(ii[0])} a {int(ii[1])}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Adimensional')
    plt.legend()
    plt.grid(True)

# Gráficos

# Aumento la resolución en las secciones de interés
f_low  = np.logspace(np.log10(0.01), np.log10(5), 3000)
f_t1   = np.linspace(ws1, wp1, 1000)
f_t2   = np.linspace(wp2, ws2, 1000)
f_rest = np.linspace(0.01, 80, 2000)
freqs = np.unique(np.concatenate([f_low, f_t1, f_t2, f_rest]))


ini, fin = 5000, 12000
plt.figure()
plt.plot(t[ini:fin], ecg[ini:fin], lw=0.6, label='crudo')
plt.plot(t[ini:fin], ecg_f[ini:fin], lw=0.6, label='filtrado')
plt.title('ECG: Filtrado Butterworth')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

w, h = sig.sosfreqz(sos, worN=freqs, fs=fs)
plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
 
plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axhline(-gstop, color='r', ls='--', lw=0.8)
plt.axvline(ws1, color='r', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.axvline(wp2, color='g', ls='--', lw=0.8)
plt.axvline(ws2, color='r', ls='--', lw=0.8)
 
plt.fill_between([wp1, wp2], -gpass, 5, color='g', alpha=0.1)
plt.fill_between([0, ws1], -60, -gstop, color='r', alpha=0.1)
plt.fill_between([ws2, 80], -60, -gstop, color='r', alpha=0.1)
 
plt.title(f'Respuesta en frecuencia - Butterworth orden {N}')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(0, 80)
plt.ylim(-60, 5)
plt.grid(True)

# Zoom al inicio de la banda de paso - Butterworth
plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.fill_between([wp1, 2], -gpass, 1, color='g', alpha=0.1)
plt.title(f'Respuesta en frecuencia (zoom banda de paso) - Butterworth orden {N}')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(1, 2)
plt.ylim(-2, 1)
plt.grid(True)


# Gráfico Chebyshev Tipo I

ini, fin = 5000, 12000
plt.figure()
plt.plot(t[ini:fin], ecg[ini:fin], lw=0.6, label='crudo')
plt.plot(t[ini:fin], ecg_f_cheb[ini:fin], lw=0.6, label='filtrado')
plt.title('ECG: Filtro Chebyshev Tipo I')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)



w_cheb, h_cheb = sig.sosfreqz(sos_cheb, worN=freqs, fs=fs)
plt.figure()
plt.plot(w_cheb, 20 * np.log10(np.abs(h_cheb) + 1e-12))
 
plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axhline(-gstop, color='r', ls='--', lw=0.8)
plt.axvline(ws1, color='r', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.axvline(wp2, color='g', ls='--', lw=0.8)
plt.axvline(ws2, color='r', ls='--', lw=0.8)
 
plt.fill_between([wp1, wp2], -gpass, 5, color='g', alpha=0.1)
plt.fill_between([0, ws1], -60, -gstop, color='r', alpha=0.1)
plt.fill_between([ws2, 80], -60, -gstop, color='r', alpha=0.1)
 
plt.title(f'Respuesta en frecuencia: Chebyshev orden {N_cheb}')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(0, 80)
plt.ylim(-60, 5)
plt.grid(True)

# Zoom al inicio de la banda de paso - Chebyshev Tipo I
plt.figure()
plt.plot(w_cheb, 20 * np.log10(np.abs(h_cheb) + 1e-12))
plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.fill_between([wp1, 2], -gpass, 1, color='g', alpha=0.1)
plt.title(f'Respuesta en frecuencia (zoom banda de paso): Chebyshev orden {N_cheb}')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(1, 2)
plt.ylim(-2, 1)
plt.grid(True)


# Gráfico FIR - Método de ventanas (firwin)

ini, fin = 5000, 12000
plt.figure()
plt.plot(t[ini:fin], ecg[ini:fin], lw=0.6, label='crudo')
plt.plot(t[ini:fin], ecg_f_win[ini:fin], lw=0.6, label='filtrado')
plt.title('ECG: FIR - Método de ventanas (Kaiser)')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)


w_win, h_win = sig.freqz(fir_win, worN=freqs, fs=fs)
plt.figure()
plt.plot(w_win, 20 * np.log10(np.abs(h_win) + 1e-12))

plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axhline(-gstop, color='r', ls='--', lw=0.8)
plt.axvline(ws1, color='r', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.axvline(wp2, color='g', ls='--', lw=0.8)
plt.axvline(ws2, color='r', ls='--', lw=0.8)

plt.fill_between([wp1, wp2], -gpass, 5, color='g', alpha=0.1)
plt.fill_between([0, ws1], -60, -gstop, color='r', alpha=0.1)
plt.fill_between([ws2, 80], -60, -gstop, color='r', alpha=0.1)

plt.title(f'Respuesta en frecuencia: FIR ventanas (Kaiser), {numtaps} taps')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(0, 80)
plt.ylim(-60, 5)
plt.grid(True)

# Zoom al inicio de la banda de paso - FIR ventanas
plt.figure()
plt.plot(w_win, 20 * np.log10(np.abs(h_win) + 1e-12))
plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.fill_between([wp1, 2], -gpass, 1, color='g', alpha=0.1)
plt.title(f'Respuesta en frecuencia (zoom banda de paso): FIR ventanas, {numtaps} taps')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(1, 2)
plt.ylim(-2, 1)
plt.grid(True)


# Gráfico FIR - Método de cuadrados mínimos (firls)

ini, fin = 5000, 12000
plt.figure()
plt.plot(t[ini:fin], ecg[ini:fin], lw=0.6, label='crudo')
plt.plot(t[ini:fin], ecg_f_ls[ini:fin], lw=0.6, label='filtrado')
plt.title('ECG: FIR - Método de cuadrados mínimos')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)


w_ls, h_ls = sig.freqz(fir_ls, worN=freqs, fs=fs)
plt.figure()
plt.plot(w_ls, 20 * np.log10(np.abs(h_ls) + 1e-12))

plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axhline(-gstop, color='r', ls='--', lw=0.8)
plt.axvline(ws1, color='r', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.axvline(wp2, color='g', ls='--', lw=0.8)
plt.axvline(ws2, color='r', ls='--', lw=0.8)

plt.fill_between([wp1, wp2], -gpass, 5, color='g', alpha=0.1)
plt.fill_between([0, ws1], -60, -gstop, color='r', alpha=0.1)
plt.fill_between([ws2, 80], -60, -gstop, color='r', alpha=0.1)

plt.title(f'Respuesta en frecuencia: FIR cuadrados mínimos, {numtaps} taps')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(0, 80)
plt.ylim(-60, 5)
plt.grid(True)

# Zoom al inicio de la banda de paso - FIR cuadrados mínimos
plt.figure()
plt.plot(w_ls, 20 * np.log10(np.abs(h_ls) + 1e-12))
plt.axhline(-gpass, color='g', ls='--', lw=0.8)
plt.axvline(wp1, color='g', ls='--', lw=0.8)
plt.fill_between([wp1, 2], -gpass, 1, color='g', alpha=0.1)
plt.title(f'Respuesta en frecuencia (zoom banda de paso): FIR cuadrados mínimos, {numtaps} taps')
plt.xlabel('f [Hz]')
plt.ylabel('|H| [dB]')
plt.xlim(1, 2)
plt.ylim(-2, 1)
plt.grid(True)


# =============================================================================
# BONUS: Temperatura del aire en Buenos Aires (2024)
#
# Naturaleza de los datos:
#   - Variable: temperatura del aire en superficie [°C].
#   - Lugar:    estación AEROPARQUE JORGE NEWBERY (SABE, ID 87582099999),
#               Buenos Aires, Argentina -> HEMISFERIO SUR (verano Dic-Mar).
#   - Período:  año 2024 completo (366 días, bisiesto) -> 8784 muestras.
#   - Muestreo: horario, fs = 24 muestras/día (Nyquist = 12 ciclos/día).
#               Las observaciones METAR son ~horarias irregulares; al descargar
#               se remuestrearon a rejilla uniforme y se interpolaron huecos.
#   - Fuente:   NOAA / NCEI, Integrated Surface Database (ISD), producto
#               "global-hourly" (Buenos Aires no pertenece a la red USCRN).
#
# Selección de la plantilla:
#   El objetivo es un pasa-bajos que actúe como "promediador" y deje ver la
#   tendencia estacional, separándola del ciclo diario (1 ciclo/día) y de las
#   fluctuaciones meteorológicas (días). Por eso se piden dos cortes:
#     - "30 días": fc = 1/30 ciclo/día  -> tendencia estacional.
#     - "5 días" : fc = 1/5  ciclo/día  -> suaviza el día a día, conserva olas
#                  de calor/frío de pocos días.
#   Cada plantilla se arma geométricamente alrededor de fc con un factor r=1.5:
#     fpass = fc/r (banda de paso, períodos largos)
#     fstop = fc*r (banda de rechazo, períodos cortos)
#   Tolerancias: gpass = 1 dB, gstop = 20 dB. Se usa 20 dB (atenuación ~10x) y
#   no 40 dB a propósito: con un registro de sólo un año, exigir 40 dB obliga a
#   FIR de miles de coeficientes cuyos transitorios de borde abarcarían meses.
#   Con 20 dB los FIR resultan de ~727 (30 días) y ~123 (5 días) coeficientes,
#   que coinciden con las ventanas de 30 y 5 días (30*24 y 5*24) -> los FIR son,
#   en la práctica, promedios móviles suavizados.
# =============================================================================
df_temp = pd.read_csv('./temperature_data/buenos_aires_2024.csv',
                      parse_dates=['TIME'], index_col='TIME')
temp = df_temp['TEMP'].interpolate(limit_direction='both').to_numpy()
t_temp = df_temp.index

fs_temp = 24.0          # muestras por día (horario)
nyq_temp = fs_temp / 2
gpass_t, gstop_t = 1.0, 20.0

def plantilla_lp(Tc, r=1.5):
    """Bordes (fpass, fstop) de un pasa-bajos centrado en fc = 1/Tc [ciclo/día]."""
    fc = 1.0 / Tc
    return fc / r, fc * r

fpass30, fstop30 = plantilla_lp(30)
fpass5,  fstop5  = plantilla_lp(5)

# --- Los cuatro filtros como pasa-bajos de fase cero (mismos métodos del ECG) ---
def lp_butter(x, fp, fst):
    N, wn = sig.buttord(fp, fst, gpass_t, gstop_t, fs=fs_temp)
    sos = sig.butter(N, wn, btype='low', output='sos', fs=fs_temp)
    return sig.sosfiltfilt(sos, x)

def lp_cheby(x, fp, fst):
    N, wn = sig.cheb1ord(fp, fst, gpass_t, gstop_t, fs=fs_temp)
    sos = sig.cheby1(N, gpass_t, wn, btype='low', output='sos', fs=fs_temp)
    return sig.sosfiltfilt(sos, x)

def lp_firwin(x, fp, fst):
    nt, beta = sig.kaiserord(gstop_t, (fst - fp) / nyq_temp)
    nt |= 1
    b = sig.firwin(nt, (fp + fst) / 2, window=('kaiser', beta), fs=fs_temp)
    return sig.filtfilt(b, 1.0, x)

def lp_firls(x, fp, fst):
    nt, _ = sig.kaiserord(gstop_t, (fst - fp) / nyq_temp)
    nt |= 1
    # bandas contiguas (transición como rampa) para evitar ripples en el hueco
    bands   = [0, fp, fp, fst, fst, nyq_temp]
    desired = [1, 1, 1,  0,   0,   0]
    b = sig.firls(nt, bands, desired, fs=fs_temp)
    # firls no garantiza ganancia unitaria en continua: deja un error de ~12%
    # que filtfilt (doble pasada) eleva a ~27% y mete un offset de varios °C.
    # Se normaliza a ganancia 1 en DC, como corresponde a un promediador.
    b /= b.sum()
    return sig.filtfilt(b, 1.0, x)

metodos = [
    ('butter', 'Butterworth',          lp_butter),
    ('cheby',  'Chebyshev I',          lp_cheby),
    ('firwin', 'FIR ventanas (Kaiser)', lp_firwin),
    ('firls',  'FIR cuadrados mínimos', lp_firls),
]

# --- Mosaico 2x2: un método por panel, con los cortes de 5 y 30 días ---
fig, axd = plt.subplot_mosaic([['butter', 'cheby'], ['firwin', 'firls']],
                              figsize=(14, 8), constrained_layout=True)
for clave, nombre, func in metodos:
    ax = axd[clave]
    ax.plot(t_temp, temp, lw=0.4, color='0.7', label='Cruda (horaria)')
    ax.plot(t_temp, func(temp, fpass5,  fstop5),  lw=1.5, color='tab:blue',
            label='Promedio ~5 días')
    ax.plot(t_temp, func(temp, fpass30, fstop30), lw=2.2, color='tab:orange',
            label='Promedio ~30 días')
    ax.set_title(nombre)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Temp [°C]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
fig.suptitle('Buenos Aires 2024: pasa-bajos promediadores (4 métodos, cortes 5 y 30 días)')

# --- Tendencia estacional: sólo Butterworth con corte de 30 días ---
temp_butter30 = lp_butter(temp, fpass30, fstop30)
estaciones = [   # HEMISFERIO SUR
    ("2024-01-01", "2024-03-20", "Verano",    "#fff2b3"),
    ("2024-03-20", "2024-06-20", "Otoño",     "#f5d9b8"),
    ("2024-06-20", "2024-09-22", "Invierno",  "#cfe8ff"),
    ("2024-09-22", "2024-12-21", "Primavera", "#d7f5d0"),
    ("2024-12-21", "2025-01-01", "Verano",    "#fff2b3"),
]
plt.figure(figsize=(14, 5))
for ini, fin, nombre, color in estaciones:
    plt.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=color, zorder=0)
linea, = plt.plot(t_temp, temp_butter30, lw=2.4, color='tab:orange',
                  label='Butterworth (corte ~30 días)', zorder=3)
plt.title('Buenos Aires 2024: tendencia estacional (Butterworth, corte 30 días)')
plt.xlabel('Fecha')
plt.ylabel('Temperatura [°C]')
plt.xlim(t_temp.min(), t_temp.max())
parches = [Patch(facecolor=c, label=n)
           for n, c in dict((n, c) for _, _, n, c in estaciones).items()]
plt.legend(handles=[linea] + parches, loc='lower center', ncol=5, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.show()