import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs  = 1000
N   = 1000
df  = fs / N              # resolución espectral Δf = 1 Hz
A   = np.sqrt(2)          # potencia normalizada P = A²/2 = 1 W

# Desintonías respecto a Δf:  f0 = k0 · Δf
k0_values = [N / 4, N / 4 + 0.25, N / 4 + 0.5]

t = np.arange(N) / fs


def psd(x, nfft):
    """PSD normalizada tal que sum(psd) = potencia media de x (Parseval)."""
    X = np.fft.fft(x, nfft)
    return np.abs(X) ** 2 / (N * nfft)


# a) y b) — Tres senoidales con desintonía
fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8))
fig1.suptitle(
    'PSD de senoidales con desintonía  f0 = k0·Δf   (N = 1000, fs = 1000 Hz)',
    fontsize=13
)

freqs  = np.arange(N) * df
colors = ['steelblue', 'orangered', 'seagreen']

print("Verificación de potencia (sum PSD):")
for k0, c in zip(k0_values, colors):
    f0 = k0 * df
    x  = A * np.sin(2 * np.pi * f0 * t)

    P  = psd(x, N)
    pot_total = np.sum(P)

    # Vista completa (dB) y zoom lineal alrededor de k0
    axes1[0].plot(freqs, 10 * np.log10(P + 1e-15), color=c, lw=1.2,
                  label=f'k0 = {k0:g}  (P = {pot_total:.4f} W)')
    axes1[1].plot(freqs, P, color=c, lw=1.5, marker='o', ms=4,
                  label=f'k0 = {k0:g}')

    print(f"  k0 = {k0:8g}  f0 = {f0:7.2f} Hz  -> P = {pot_total:.6f} W")

axes1[0].set_title('PSD completa [dB]', fontsize=10)
axes1[0].set_xlabel('Frecuencia [Hz]')
axes1[0].set_ylabel('PSD [dB]')
axes1[0].set_xlim(0, fs / 2)
axes1[0].legend(fontsize=9)
axes1[0].grid(True, alpha=0.25)

axes1[1].set_title('Zoom lineal en torno a k0 = N/4 = 250', fontsize=10)
axes1[1].set_xlabel('Frecuencia [Hz]')
axes1[1].set_ylabel('PSD [W]')
axes1[1].set_xlim(244, 256)
axes1[1].legend(fontsize=9)
axes1[1].grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('TS4_psd_desintonia.png', dpi=150, bbox_inches='tight')
plt.show()


# Zero padding (9*N ceros -> nfft = 10*N)
nfft     = 10 * N
df_zp    = fs / nfft
freqs_zp = np.arange(nfft) * df_zp

fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.suptitle(
    f'Zero padding ({9*N} ceros -> nfft = {nfft}, Δf = {df_zp:g} Hz)',
    fontsize=13
)

print("\nZero padding (sum PSD):")
for k0, c in zip(k0_values, colors):
    f0 = k0 * df
    x  = A * np.sin(2 * np.pi * f0 * t)   # señal original de N muestras

    P_zp = psd(x, nfft)                   # fft con relleno de ceros
    pot  = np.sum(P_zp)

    ax2.plot(freqs_zp, P_zp, color=c, lw=1.2,
             label=f'k0 = {k0:g}  (P = {pot:.4f} W)')
    print(f"  k0 = {k0:8g}  f0 = {f0:7.2f} Hz  -> P = {pot:.6f} W")

ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('PSD [W]')
ax2.set_xlim(244, 256)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('TS4_psd_zeropadding.png', dpi=150, bbox_inches='tight')
plt.show()
