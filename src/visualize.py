import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
import seaborn as sns

# 1. Diagrama de flujo general
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.axis('off')

# Coordinates for blocks
blocks = [
    ((0.1, 0.7), "Video BrnoCompSpeed"),
    ((0.4, 0.7), "YOLO11s-seg\n+ BoTSort"),
    ((0.7, 0.7), "Segmentación\nInstancias"),
    ((0.4, 0.4), "Extracción\nde Features"),
    ((0.4, 0.1), "Generación\nde Secuencias"),
    ((0.7, 0.4), "LSTM many-to-one"),
    ((0.7, 0.1), "Salida de\nVelocidad")
]

# Draw blocks
for (x, y), text in blocks:
    rect = FancyBboxPatch((x, y), 0.25, 0.15, boxstyle="round,pad=0.02", linewidth=1, edgecolor='black', facecolor='lightgray')
    ax1.add_patch(rect)
    ax1.text(x+0.125, y+0.075, text, ha='center', va='center', fontsize=8)

# Draw arrows
arrows = [
    ((0.35, 0.775), (0.4, 0.775)),
    ((0.65, 0.775), (0.7, 0.775)),
    ((0.525, 0.7), (0.525, 0.55)),
    ((0.525, 0.4), (0.525, 0.25)),
    ((0.65, 0.55), (0.7, 0.55)),
    ((0.65, 0.25), (0.7, 0.25))
]
for start, end in arrows:
    ax1.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1))

ax1.set_title("Pipeline General", fontsize=10)
plt.tight_layout()

# 2. Evolución de features para un solo vehículo
fig2, axes2 = plt.subplots(5, 1, figsize=(6, 8), sharex=True)
time_steps = np.arange(0, 50)
np.random.seed(0)
features = {
    "Aspect Ratio": np.sin(0.1 * time_steps) + 1 + 0.1*np.random.randn(50),
    "Ángulo Rotado": np.cos(0.1 * time_steps) * 30 + 0.1*np.random.randn(50),
    "Área de Máscara": np.abs(np.sin(0.05 * time_steps) * 500 + 0.1*np.random.randn(50)),
    "Y Bottom": np.linspace(400, 200, 50) + 5*np.random.randn(50),
    "Ancho BBox": np.linspace(50, 150, 50) + 5*np.random.randn(50)
}

for ax, (feat_name, data) in zip(axes2, features.items()):
    ax.plot(time_steps, data, color='tab:blue')
    ax.set_ylabel(feat_name, fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5)
axes2[-1].set_xlabel("Frame Index", fontsize=8)
fig2.suptitle("Evolución de Features (Ejemplo de una Trajectoria)", fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# 3. Arquitectura LSTM many-to-one
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.axis('off')

# Blocks for LSTM architecture
lstm_blocks = [
    ((0.1, 0.4), "Input: B×T×F"),
    ((0.4, 0.4), "pack_padded\nsequence"),
    ((0.55, 0.55), "LSTM Layer 1\n(hidden=64)"),
    ((0.55, 0.35), "LSTM Layer 2\n(hidden=64)"),
    ((0.75, 0.4), "Dropout\n(p=0.0)"),
    ((0.9, 0.4), "FC → ŷ")
]

for (x, y), text in lstm_blocks:
    rect = FancyBboxPatch((x, y), 0.2, 0.15, boxstyle="round,pad=0.02", linewidth=1, edgecolor='black', facecolor='lightgray')
    ax3.add_patch(rect)
    ax3.text(x+0.1, y+0.075, text, ha='center', va='center', fontsize=8)

# Arrows for LSTM
lstm_arrows = [
    ((0.3, 0.475), (0.4, 0.475)),
    ((0.6, 0.625), (0.55, 0.625)),
    ((0.6, 0.475), (0.55, 0.475)),
    ((0.75, 0.475), (0.75, 0.55)),
    ((0.85, 0.475), (0.75, 0.475)),
    ((0.95, 0.475), (0.9, 0.475))
]
for start, end in lstm_arrows:
    ax3.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1))
ax3.set_title("Arquitectura LSTM Many-to-One", fontsize=10)
plt.tight_layout()

# 4. Scatter y histograma de error
# Sample data
np.random.seed(0)
v_real = np.random.uniform(20, 120, 200)
v_pred = v_real + np.random.normal(0, 2, 200)

fig4, (ax4_scatter, ax4_hist) = plt.subplots(1, 2, figsize=(8, 4))

# Scatter plot
ax4_scatter.scatter(v_real, v_pred, color='tab:blue', alpha=0.6, s=10)
ax4_scatter.plot([20, 120], [20, 120], 'r--')
ax4_scatter.set_xlabel("Velocidad Real (km/h)", fontsize=8)
ax4_scatter.set_ylabel("Velocidad Predicha (km/h)", fontsize=8)
ax4_scatter.set_title("Scatter: Real vs Predicha", fontsize=9)
ax4_scatter.grid(True, linestyle='--', linewidth=0.5)

# Histograma del error absoluto
errors = np.abs(v_pred - v_real)
ax4_hist.hist(errors, bins=20, color='tab:green', alpha=0.7, edgecolor='black')
ax4_hist.set_xlabel("Error Absoluto |v_pred - v_real| (km/h)", fontsize=8)
ax4_hist.set_ylabel("Frecuencia", fontsize=8)
ax4_hist.set_title("Histograma del Error Absoluto", fontsize=9)
ax4_hist.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()

# 5. Layout poster completo (esquema)
fig5, ax5 = plt.subplots(figsize=(10, 7))
ax5.axis('off')

# Create rectangles to denote sections
poster_sections = {
    "Título & Autores": (0.1, 0.9, 0.8, 0.08),
    "Introducción\n& Dataset": (0.05, 0.6, 0.4, 0.25),
    "Pipeline General": (0.48, 0.6, 0.47, 0.25),
    "Evolución de Features": (0.05, 0.3, 0.4, 0.25),
    "Arquitectura LSTM": (0.48, 0.3, 0.47, 0.25),
    "Resultados": (0.05, 0.05, 0.4, 0.25),
    "Demo Video (QR)": (0.48, 0.05, 0.23, 0.25),
    "Conclusiones": (0.73, 0.05, 0.22, 0.25)
}

for label, (x, y, w, h) in poster_sections.items():
    rect = Rectangle((x, y), w, h, edgecolor='black', facecolor='none', lw=1.2)
    ax5.add_patch(rect)
    ax5.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)

ax5.set_title("Esquema de Póster Completo", fontsize=12)
plt.tight_layout()

plt.show()
