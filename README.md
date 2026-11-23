# reventauto: qué Toyota Yaris comprar hoy para conservar valor

Datos tomados el **23/11/2026**: 928 publicaciones de MercadoLibre y 19 de Kavak (todo el stock de Yaris que tenía Kavak ese día).
Todo está en **USD**. Los precios en pesos se convierten al dólar MEP de **$1.700**.

## Cómo correrlo

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
python scripts/scrape.py     # baja publicaciones -> data/raw_ml.csv, data/raw_kavak.csv
python scripts/clean.py      # limpia y unifica    -> data/yaris_listings.csv
python scripts/analysis.py   # EDA + regresión + proyección -> outputs/
```

Parámetros editables: `MEP` en `clean.py`, y `HORIZON` (años) y `KM_YEAR` (km por año) en `analysis.py`.

## Archivos

| Archivo | Contenido |
|---|---|
| `data/yaris_listings.csv` | Dataset limpio con 838 publicaciones (año, versión, carrocería, transmisión, km, precio, moneda, fuente, url) |
| `data/raw_*.csv` | Datos crudos del scraping |
| `data/historical_reference.csv` | Precios de lista 0 km de 2025 y 2026 (ARS), tomados de notas de prensa |
| `outputs/reporte.txt` | Salida completa: EDA, las dos regresiones, validación cruzada, proyecciones |
| `outputs/coeficientes_regresion.csv` | Cuánto aporta cada variable (USD y %) |
| `outputs/segmentos_proyeccion_4a.csv` | Precio hoy y en 4 años por año, versión, transmisión y carrocería |
| `outputs/ranking_publicaciones.csv` | Cada publicación de Kavak y las XS/XLS de ML, ordenadas por menor pérdida esperada en 4 años |
| `outputs/0*.png` | Gráficos |

## Limpieza

- Se excluyen el GR Yaris (otro segmento) y la generación anterior a 2016.
- Se descartan duplicados y precios absurdos.
- Se sacan 10 publicaciones con residuo mayor a 3 desvíos del modelo. Son casi todas "0 km a USD 12k", es decir anticipos o planes de ahorro. Están listadas en `outputs/publicaciones_descartadas_outliers.csv`.
- **Kavak:** se usa el precio de contado, no el precio "financiando desde 40%".
- **Transmisión:** se infiere del título de MercadoLibre. "CVT" o "AT" cuenta como automático y el resto como manual. S y XLS Pack son siempre CVT.

## Regresión lineal múltiple: cuánto vale cada variable

Errores robustos HC3. R² = 0,89 en USD y 0,90 en logaritmo.

| Variable (el resto constante) | Efecto en USD | Efecto en % |
|---|---|---|
| +1 año de antigüedad | **−800** | −4,3% |
| +10.000 km | **−252** | −1,5% |
| XLS (vs XS) | +1.628 | +9,1% |
| XLS Pack (vs XS) | +2.479 | +14,1% |
| S (vs XS) | +3.383 | +18,2% |
| Manual (vs automático) | −306 | −2,6% |
| Sedán (vs hatchback) | −388 | −2,6% |
| 0 km | +1.227 | +4,3% |
| Publicado en USD (vs pesos al MEP) | +2.251 | +12,8% |
| Publicado en Kavak (vs ML) | +686 (p=0,06) | +3,6% |
| Interior (vs AMBA) | ≈0 (no significativo) | |

Qué dicen los números:

- **Antigüedad y km explican casi todo.** Un año de antigüedad equivale a unos 32.000 km.
- **Automático y hatchback valen más, pero poco: ~2,6% cada uno**, unos USD 300–400.
- **Los manuales se deprecian más lento.** En el modelo predictivo, la interacción `manual × antigüedad` da +1,2% por año (p<0,001). El automático arranca más caro y pierde esa diferencia con los años. En sedán la interacción no es significativa.
- **Las publicaciones en USD piden ~13% más** que las publicadas en pesos al MEP. Ese 13% es el margen de negociación, o el tipo de cambio implícito que usan los vendedores en pesos.
- **Kavak no es más caro que ML** (+3,6%, apenas significativo). Hoy no se paga un sobreprecio grande por comprar ahí.

## Modelo predictivo y proyección a 4 años

Se compararon tres modelos con validación cruzada de 5 particiones (error absoluto medio, MAE):

| Modelo | MAE |
|---|---|
| OLS lineal en USD | USD 950 |
| **OLS en logaritmo, con curvatura en edad y km, e interacciones** | **USD 860** ← elegido |
| Gradient Boosting monótono | USD 915 |

Se eligió el modelo logarítmico porque además de ser el más preciso **extrapola razonablemente**: un 2018 dentro de 4 años tiene 12 años de antigüedad, y en los datos casi no hay autos tan viejos.

Cómo se proyecta: al auto se le suman 4 años de antigüedad y 48.000 km (12.000 por año). Se valora como publicación en pesos de MercadoLibre en AMBA, en USD constantes. Se incluye un intervalo del 80% (p10–p90).

### Pérdida esperada en 4 años según el segmento (km típico de cada año)

| Año que comprás | XS manual | XLS manual | XS CVT | XLS CVT hatch | XLS CVT sedán |
|---|---|---|---|---|---|
| 2026 | 2.936 (84%) | 3.176 (84%) | 3.965 (80%) | 4.289 (80%) | 3.838 (81%) |
| 2024 | 3.044 (81%) | 3.293 (81%) | 3.890 (77%) | 4.208 (77%) | 3.829 (79%) |
| 2022 | 3.115 (79%) | 3.370 (79%) | 3.787 (75%) | 4.096 (75%) | 3.785 (76%) |
| 2020 | 3.256 (76%) | 3.522 (76%) | 3.768 (72%) | 4.076 (72%) | 3.824 (73%) |

La primera cifra es la pérdida en USD. Entre paréntesis, el porcentaje del valor que conserva.

Conclusiones:

1. **La pérdida en dólares es casi la misma cualquiera sea el año: unos USD 3–4,3k en 4 años.** Por eso conviene comprar lo más nuevo que permita el presupuesto: por la misma pérdida tenés un auto mejor, y en porcentaje retiene más (80–84% contra 69–72% de un 2018).
2. **Un XS o XLS manual pierde unos USD 1.000 menos que un automático** en 4 años. Tu intuición de "manual barato con buena reventa" se sostiene.
3. **Hatchback contra sedán:** el sedán es ~2,6% más barato de entrada y pierde algo menos en USD. El hatchback no aporta ventaja de reventa medible.
4. **XS contra XLS:** retienen el mismo porcentaje. El XLS cuesta ~USD 1.600 más y pierde ~USD 250–300 más.

### Publicaciones de Kavak (de menor a mayor pérdida esperada)

| Auto | Km | Precio contado | Valor de mercado hoy | Reventa en 4 años (p10–p90) | Pérdida |
|---|---|---|---|---|---|
| 2024 XLS CVT sedán | 59.855 | 15.653 | 17.226 | 13.694 (12,7k–14,8k) | **1.959** |
| 2021 XLS manual sedán | 48.053 | 14.353 | 15.466 | 11.959 (11,1k–12,9k) | 2.394 |
| **2024 XS manual hatch** | 20.710 | 16.618 | 16.874 | 13.580 (12,6k–14,7k) | **3.037** |
| 2021 XLS CVT sedán | 43.201 | 15.376 | 15.934 | 11.703 | 3.674 |
| 2018 XLS manual hatch | 56.113 | 13.447 | 13.624 | 9.703 | 3.744 |
| 2025 XS CVT hatch | 32.899 | 18.865 | 17.913 | 14.144 | 4.720 |
| 2025 XLS CVT hatch | 3.867 | 22.576 | 20.669 | 16.074 | 6.502 |

La lista completa está en `outputs/ranking_publicaciones.csv`.

**Recomendación:** de lo que hay hoy en Kavak y encaja con lo que buscás (XS/XLS hatchback), el **2024 XS manual con 20.700 km a ~USD 16,6k ($28,25M)** es el que mejor conserva valor. Está a precio de mercado y perdería unos USD 3.000 en 4 años.

Si aceptás un sedán, el **2024 XLS CVT con 60k km** es la mejor compra (9% bajo mercado). Pero tiene mucho uso: 30k km por año.

Ningún XLS CVT hatchback de Kavak está barato hoy. El 2025 está ~9% sobre mercado. En MercadoLibre hay XLS CVT hatch 2024 con ~40k km a ~USD 16,2k, un 13% bajo mercado.

## Limitaciones

- **Son precios de publicación, no de venta.** Si en 4 años le vendés el auto a Kavak, esperá entre 10% y 15% menos que el valor proyectado.
- **Datos históricos escasos.** No pude acceder a Wayback Machine y Kavak/ML no muestran publicaciones viejas. La depreciación se estima con el **corte transversal de hoy**: un 2022 de hoy representa a un 2026 dentro de 4 años. Esto supone que los precios en USD de los usados se mantienen estables en términos reales. En Argentina eso puede moverse mucho con el tipo de cambio y la apertura de importaciones.
- La transmisión se infiere del título en ML, así que algún CVT mal titulado puede figurar como manual.
- La conversión se hizo con MEP $1.700. Si cambia, rehacé `clean.py` con el valor nuevo.
