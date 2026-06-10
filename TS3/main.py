import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs       = 1000
N        = 1000
f0       = fs / N          # f0 = Δf, 1 ciclo exacto en N muestras
A        = np.sqrt(2)      # potencia normalizada P = A²/2 = 1 W
VF       = 2.0             # fondo de escala [V]
B_values = [4, 8, 16]

t  = np.arange(N) / fs
sR = A * np.sin(2 * np.pi * f0 * t)


def quantize(s, B, VF):
    """Cuantizador uniforme midrise de B bits y rango ±VF."""
    q      = 2 * VF / 2**B
    s_clip = np.clip(s, -VF, VF)
    s_q    = q * (np.floor(s_clip / q) + 0.5)
    s_q    = np.clip(s_q, -VF, VF)
    return s_q, q


# Figura 1 – sR vs sQ
fig1, axes1 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig1.suptitle(
    f'Cuantización ADC: sR vs sQ\n'
    f'(VF = {VF} V,  f0 = {f0:.0f} Hz,  P_sR = 1 W)',
    fontsize=13
)

t_ms = t * 1e3

for ax, B in zip(axes1, B_values):
    sQ, q = quantize(sR, B, VF)

    ax.plot(t_ms, sR, color='steelblue', lw=1.5, alpha=0.85, label='sR')
    ax.step(t_ms, sQ, color='orangered', lw=1.2, alpha=0.90,
            where='post', label=f'sQ  (B = {B} bits)')
    ax.axhline( VF, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(-VF, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylim(-VF * 1.15, VF * 1.15)
    ax.set_ylabel('Amplitud [V]')
    ax.set_title(f'B = {B} bits  →  {2**B} niveles,  q = {q:.5f} V', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)

axes1[-1].set_xlabel('Tiempo [ms]')
plt.tight_layout()
plt.savefig('TS3_sR_vs_sQ.png', dpi=150, bbox_inches='tight')
plt.show()


# Figura 2 – Error en tiempo + histograma
fig2, axes2 = plt.subplots(3, 2, figsize=(13, 10))
fig2.suptitle(
    'Error de Cuantización  e = sQ - sR\n'
    '(distribución uniforme, media y varianza)',
    fontsize=13
)

for row, B in enumerate(B_values):
    sQ, q   = quantize(sR, B, VF)
    e       = sQ - sR
    mean_e  = np.mean(e)
    var_e   = np.var(e)
    var_teo = q**2 / 12

    ax_t = axes2[row, 0]
    ax_t.plot(t_ms, e, color='mediumpurple', lw=0.8)
    ax_t.axhline( q / 2, color='tomato', ls='--', lw=1.0, label=f'+q/2 = {q/2:.4f} V')
    ax_t.axhline(-q / 2, color='tomato', ls='--', lw=1.0, label=f'−q/2')
    ax_t.set_ylabel('Error [V]')
    ax_t.set_title(
        f'B = {B}  |  μ = {mean_e:.2e} V  |  σ² = {var_e:.2e}  (teo: {var_teo:.2e})',
        fontsize=9
    )
    ax_t.legend(fontsize=8, loc='upper right')
    ax_t.grid(True, alpha=0.25)

    ax_h = axes2[row, 1]
    ax_h.hist(e, bins=60, density=True, color='mediumpurple', alpha=0.65,
              edgecolor='white', lw=0.4, label='Histograma')
    x_u = np.linspace(-q * 0.8, q * 0.8, 300)
    y_u = np.where(np.abs(x_u) < q / 2, 1.0 / q, 0.0)
    ax_h.plot(x_u, y_u, 'r-', lw=2, label='U(-q/2, q/2)')
    ax_h.set_xlabel('Error [V]')
    ax_h.set_ylabel('Densidad de prob.')
    ax_h.set_title(f'Distribución del error  (B = {B} bits)', fontsize=9)
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.25)

axes2[-1, 0].set_xlabel('Tiempo [ms]')
plt.tight_layout()
plt.savefig('TS3_error_analisis.png', dpi=150, bbox_inches='tight')
plt.show()


# Figura 3 – Autocorrelación del error
MAX_LAG = 60

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
fig3.suptitle(
    'Autocorrelación del Error (verificación de incorrelación)',
    fontsize=13
)

for ax, B in zip(axes3, B_values):
    sQ, q = quantize(sR, B, VF)
    e     = sQ - sR
    e_c   = e - np.mean(e)

    corr  = np.correlate(e_c, e_c, mode='full')
    corr /= corr[N - 1]
    mid   = N - 1
    lags  = np.arange(MAX_LAG + 1)
    acf   = corr[mid : mid + MAX_LAG + 1]

    ml, sl, bl = ax.stem(lags, acf, linefmt='mediumpurple', markerfmt='o', basefmt='k-')
    ml.set_markersize(3)
    sl.set_linewidth(0.7)

    ci = 1.96 / np.sqrt(N)
    ax.axhspan(-ci, ci, alpha=0.15, color='red', label=f'IC 95 %  (±{ci:.3f})')
    ax.set_title(f'ACF del error  (B = {B} bits)', fontsize=10)
    ax.set_xlabel('Lag [muestras]')
    ax.set_ylabel('Re[k] normalizada')
    ax.set_xlim(-1, MAX_LAG)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('TS3_autocorrelacion.png', dpi=150, bbox_inches='tight')
plt.show()


for B in B_values:
    sQ, q  = quantize(sR, B, VF)
    e      = sQ - sR
    var_e  = np.var(e)
    sqnr   = 10 * np.log10(np.mean(sR**2) / var_e)
    sqnr_t = 6.02 * B + 1.76   # teórico para senoidal
    print(f"B={B:>2}  q={q:.5f} V  μ={np.mean(e):.2e}  σ²={var_e:.2e} (teo {q**2/12:.2e})  SQNR={sqnr:.2f} dB (teo {sqnr_t:.2f})")
