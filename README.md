# mutualinfo

**Estimación de Información Mutua y Cuantificación de la Incertidumbre**

Este paquete proporciona implementaciones de distintos estimadores de **información mutua (MI)** y métodos para **cuantificar la incertidumbre** de dichas estimaciones, todo desarrollado en Python.

El objetivo principal es facilitar el análisis de relaciones estadísticas entre variables, con herramientas modernas, modulares y científicamente fundamentadas.

---

## 🚀 Características

- Estimadores clásicos:
  - Kraskov (vecinos más cercanos)
  - KDE (Kernel Density Estimation)
  - Histogramas (discretización)
  
- Métodos de cuantificación de incertidumbre:
  - Bootstrap (intervalos de confianza)
  - Conformal Prediction (intervalos de predicción)

- Generación de datos sintéticos y ejemplos visuales.
- Tests unitarios incluidos (`pytest`).

---

## 📦 Instalación

### Clonación del repositorio
```bash
git clone https://github.com/tu_usuario/mutualinfo.git
cd mutualinfo
pip install -e .

