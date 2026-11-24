"""EDA + regresión lineal múltiple + modelo de reventa a 4 años.
Entrada: data/yaris_listings.csv   Salidas: outputs/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, statsmodels.formula.api as smf
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "outputs"; O.mkdir(exist_ok=True)
HORIZON, KM_YEAR = 4, 12000      # años de tenencia, km anuales supuestos

COL = {"XS": "#2a78d6", "XLS": "#eb6834", "XLS Pack": "#1baf7a", "S": "#eda100", "ND": "#9a9890",
     "automatico": "#2a78d6", "manual": "#eb6834", "hatchback": "#2a78d6", "sedan": "#eb6834",
     "kavak": "#2a78d6", "mercadolibre": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})
usd = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

df = pd.read_csv(ROOT / "data/yaris_listings.csv")
df["km10k"] = df.km / 10000
df["kavak"] = (df.source == "kavak").astype(int)
df["manual"] = (df.transmission == "manual").astype(int)
df["sedan"] = (df.body == "sedan").astype(int)
df["interior"] = (df.region == "Interior").astype(int)
df["log_price"] = np.log(df.price_usd)
rep = []
P = lambda *a: rep.append(" ".join(str(x) for x in a))
_f = smf.ols("log_price ~ age + I(age**2) + km10k + C(version) + manual + sedan + kavak + listed_in_usd + interior + is_new", df).fit()
_z = _f.resid / _f.resid.std()
SUSP = df[_z.abs() > 3].copy()
SUSP.to_csv(O / "publicaciones_descartadas_outliers.csv", index=False)
df = df[_z.abs() <= 3].copy()

# ---------------------------------------------------------------- EDA
P(f"Se descartaron {len(SUSP)} publicaciones con residuo > 3 desvíos (anticipos, planes de ahorro, errores de carga). Ver publicaciones_descartadas_outliers.csv\n")
P("# EDA\n", f"n = {len(df)} publicaciones ({df.kavak.sum()} Kavak, {(1-df.kavak).sum()} MercadoLibre)\n")
P(df[["price_usd", "year", "age", "km"]].describe().round(0).to_string(), "\n")
P("Correlaciones (Pearson) con precio USD:\n",
  df[["price_usd", "age", "km", "manual", "sedan", "kavak", "listed_in_usd", "interior", "is_new"]].corr()["price_usd"].round(2).to_string(), "\n")
for col in ["version", "transmission", "body", "source", "region", "currency"]:
    P(f"Precio por {col}:\n", df.groupby(col).price_usd.agg(["count", "median", "mean"]).round(0).to_string(), "\n")
P("Mediana de precio por año y transmisión (controla el año):\n",
  df.pivot_table("price_usd", "year", "transmission", "median").round(0).to_string(), "\n")
P("Mediana de precio por año y versión:\n", df.pivot_table("price_usd", "year", "version", "median").round(0).to_string(), "\n")
P("Mediana de precio por año y carrocería:\n", df.pivot_table("price_usd", "year", "body", "median").round(0).to_string(), "\n")

# fig 1: precio vs año por versión
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for v in ["XS", "XLS", "XLS Pack", "S", "ND"]:
    s = df[df.version == v]; j = np.random.default_rng(0).uniform(-.25, .25, len(s))
    ax[0].scatter(s.year + j, s.price_usd, s=12, alpha=.55, color=COL[v], label=f"{v} ({len(s)})", linewidths=0)
med = df.groupby("year").price_usd.median()
ax[0].plot(med.index, med.values, color=INK, lw=2, label="mediana")
ax[0].set(title="Precio vs. año del modelo", xlabel="Año", ylabel="Precio (USD)"); ax[0].yaxis.set_major_formatter(usd)
ax[0].legend(fontsize=8, ncol=2)
for v in ["XS", "XLS", "XLS Pack", "S"]:
    s = df[df.version == v]
    ax[1].scatter(s.km / 1000, s.price_usd, s=12, alpha=.55, color=COL[v], label=v, linewidths=0)
ax[1].set(title="Precio vs. kilometraje", xlabel="Kilómetros (miles)", ylabel="Precio (USD)"); ax[1].yaxis.set_major_formatter(usd)
ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(O / "01_precio_vs_anio_km.png", dpi=150); plt.close()

# fig 2: small multiples - mediana por año según cada variable categórica
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, (col, levels) in zip(axs, [("transmission", ["automatico", "manual"]), ("body", ["hatchback", "sedan"]),
                                    ("version", ["XS", "XLS", "XLS Pack", "S"])]):
    for lv in levels:
        g = df[df[col] == lv].groupby("year").price_usd.agg(["median", "count"]); g = g[g["count"] >= 3]
        ax.plot(g.index, g["median"], marker="o", ms=5, lw=2, color=COL[lv], label=lv)
    ax.set(title=f"Mediana por año · {dict(transmission='transmisión', body='carrocería', version='versión')[col]}", xlabel="Año"); ax.yaxis.set_major_formatter(usd); ax.legend(fontsize=8)
axs[0].set_ylabel("Precio mediano (USD)")
fig.tight_layout(); fig.savefig(O / "02_mediana_por_anio_y_variable.png", dpi=150); plt.close()

# fig 3: km por año (uso típico)
fig, ax = plt.subplots(figsize=(6.5, 4))
u = df[df.age >= 1]; kmy = (u.km / u.age); P(f"Km/año mediano en usados: {kmy.median():,.0f}\n")
g = df.groupby("age").km.median()
ax.bar(g.index, g.values / 1000, color=COL["XS"], width=.6)
ax.set(title="Kilometraje mediano por antigüedad", xlabel="Antigüedad (años)", ylabel="Km (miles)")
fig.tight_layout(); fig.savefig(O / "03_km_por_antiguedad.png", dpi=150); plt.close()

# ---------------------------------------------------------------- Regresión lineal múltiple
FORM_LIN = "price_usd ~ age + km10k + C(version, Treatment('XS')) + manual + sedan + kavak + listed_in_usd + interior + is_new"
lin = smf.ols(FORM_LIN, df).fit(cov_type="HC3")
FORM_LOG = FORM_LIN.replace("price_usd", "log_price")
logm = smf.ols(FORM_LOG, df).fit(cov_type="HC3")
P("# Regresión lineal múltiple (USD, errores robustos HC3)\n", lin.summary().as_text(), "\n")
P("# Regresión log-lineal (efectos en %)\n", logm.summary().as_text(), "\n")
names = {"age": "+1 año de antigüedad", "km10k": "+10.000 km", "manual": "Manual (vs automático)",
         "sedan": "Sedán (vs hatchback)", "kavak": "Publicado en Kavak", "listed_in_usd": "Publicado en USD (vs pesos)",
         "interior": "Interior (vs AMBA)", "is_new": "0 km",
         "C(version, Treatment('XS'))[T.XLS]": "XLS (vs XS)", "C(version, Treatment('XS'))[T.XLS Pack]": "XLS Pack (vs XS)",
         "C(version, Treatment('XS'))[T.S]": "S (vs XS)", "C(version, Treatment('XS'))[T.ND]": "Versión sin dato (vs XS)"}
coef = pd.DataFrame({"variable": [names.get(k, k) for k in lin.params.index], "usd": lin.params.values,
                     "ic95_low": lin.conf_int()[0].values, "ic95_high": lin.conf_int()[1].values, "p_value": lin.pvalues.values,
                     "pct_log_model": (np.exp(logm.params.values) - 1) * 100}).iloc[1:]
coef.round(3).to_csv(O / "coeficientes_regresion.csv", index=False)
P("Tabla resumen de coeficientes:\n", coef.round(2).to_string(index=False), "\n")
P(f"R² lineal = {lin.rsquared:.3f} | R² log = {logm.rsquared:.3f}\n")

fig, ax = plt.subplots(figsize=(8, 5))
c = coef.sort_values("usd"); y = np.arange(len(c))
ax.hlines(y, c.ic95_low, c.ic95_high, color=INK2, lw=1.5)
ax.scatter(c.usd, y, color=[COL["XS"] if v > 0 else COL["XLS"] for v in c.usd], s=40, zorder=3)
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(y, c.variable)
for yy, v in zip(y, c.usd): ax.text(v, yy + .28, f"{v:+,.0f}", ha="center", fontsize=8, color=INK2)
ax.set(title="Cuánto aporta cada variable al precio (USD, IC 95%)", xlabel="Efecto sobre el precio, manteniendo el resto constante (USD)")
ax.grid(axis="y", visible=False); fig.tight_layout(); fig.savefig(O / "04_coeficientes_regresion.png", dpi=150); plt.close()

# ---------------------------------------------------------------- Modelo predictivo
FEATS = ["age", "km10k", "manual", "sedan", "kavak", "listed_in_usd", "interior", "is_new"]
FORM_PRED = ("log_price ~ age + I(age**2) + km10k + I(km10k**2) + C(version, Treatment('XS')) + manual + manual:age"
             " + sedan + sedan:age + kavak + listed_in_usd + interior + is_new")
def Xgb(d): return pd.concat([d[FEATS], pd.get_dummies(d.version).reindex(columns=["XLS", "XLS Pack", "S", "ND"], fill_value=0)], axis=1).astype(float)
res = {"OLS lineal (USD)": [], "OLS log + curvatura + interacciones": [], "Gradient Boosting (monótono)": []}
for tr, te in KFold(5, shuffle=True, random_state=42).split(df):
    a, b = df.iloc[tr], df.iloc[te]
    res["OLS lineal (USD)"].append(np.abs(smf.ols(FORM_LIN, a).fit().predict(b) - b.price_usd).mean())
    res["OLS log + curvatura + interacciones"].append(np.abs(np.exp(smf.ols(FORM_PRED, a).fit().predict(b)) - b.price_usd).mean())
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, monotonic_cst=[-1, -1] + [0] * 10, random_state=0)
    gb.fit(Xgb(a), a.log_price); res["Gradient Boosting (monótono)"].append(np.abs(np.exp(gb.predict(Xgb(b))) - b.price_usd).mean())
cv = pd.Series({k: np.mean(v) for k, v in res.items()}).round(0)
P("# Validación cruzada 5-fold (MAE en USD)\n", cv.to_string(), "\n")
pred = smf.ols(FORM_PRED, df).fit(cov_type="HC3")
smear = np.mean(np.exp(pred.resid))  # corrección de retransformación (Duan)
P("# Modelo de proyección elegido: OLS log con curvatura en edad/km e interacciones\n", pred.summary().as_text(), "\n",
  f"Factor de smearing de Duan: {smear:.4f}\n")
resid_sd = np.std(pred.resid)

def predict(d):
    return np.exp(pred.predict(d)) * smear

def project(d):
    """Precio esperado hoy y en HORIZON años (USD constantes, venta particular en ML)."""
    now = d.copy(); now["kavak"] = 0; now["listed_in_usd"] = 0; now["interior"] = 0
    fut = now.copy(); fut["age"] = now.age + HORIZON; fut["km10k"] = now.km10k + HORIZON * KM_YEAR / 1e4; fut["is_new"] = 0
    return predict(now), predict(fut)

# Grilla de segmentos: qué comprar hoy (km típico para su antigüedad)
km_by_age = df[df.is_new == 0].groupby("age").km.median().reindex(range(0, 11)).interpolate().fillna(0)
km_by_age[0] = 5000
rows = []
for year in range(2018, 2027):
    for ver, tr, body in [("XS", "manual", "hatchback"), ("XS", "automatico", "hatchback"), ("XLS", "manual", "hatchback"),
                          ("XLS", "automatico", "hatchback"), ("XLS", "automatico", "sedan"), ("XLS", "manual", "sedan")]:
        age = 2026 - year
        rows.append(dict(year=year, version=ver, transmission=tr, body=body, age=age, km=km_by_age[age]))
g = pd.DataFrame(rows); g["km10k"] = g.km / 1e4; g["manual"] = (g.transmission == "manual").astype(int)
g["sedan"] = (g.body == "sedan").astype(int); g["is_new"] = 0; g["kavak"] = 0; g["listed_in_usd"] = 0; g["interior"] = 0
g["precio_hoy"], g["precio_4a"] = project(g)
g["perdida_usd"] = g.precio_hoy - g.precio_4a; g["retencion_pct"] = g.precio_4a / g.precio_hoy * 100
g["perdida_usd_anual"] = g.perdida_usd / HORIZON
g.round(0).to_csv(O / "segmentos_proyeccion_4a.csv", index=False)
P("# Proyección por segmento (km típico, USD constantes, precio de venta particular)\n",
  g[["year", "version", "transmission", "body", "km", "precio_hoy", "precio_4a", "perdida_usd", "retencion_pct"]].round(0).to_string(index=False), "\n")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for (ver, tr, body), s in g.groupby(["version", "transmission", "body"]):
    if body == "sedan" and tr == "manual": continue
    lab = f"{ver} {tr[:4]}. {body[:5]}"
    col = {"XS manual": "#2a78d6", "XS automatico": "#1baf7a", "XLS manual": "#eb6834", "XLS automatico": "#eda100"}[f"{ver} {tr}"]
    ls = "--" if body == "sedan" else "-"
    axs[0].plot(s.year, s.perdida_usd, marker="o", ms=5, lw=2, color=col, ls=ls, label=lab)
    axs[1].plot(s.year, s.retencion_pct, marker="o", ms=5, lw=2, color=col, ls=ls, label=lab)
axs[0].set(title=f"Pérdida de valor esperada en {HORIZON} años (USD)", xlabel="Año del auto que comprás hoy", ylabel="USD perdidos")
axs[1].set(title=f"% del valor que conserva en {HORIZON} años", xlabel="Año del auto que comprás hoy", ylabel="% retenido")
axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
axs[1].text(.98, .03, "XS y XLS manual retienen casi igual (líneas superpuestas)", transform=axs[1].transAxes, ha="right", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(O / "05_perdida_valor_4a_por_segmento.png", dpi=150); plt.close()

# Ranking de publicaciones concretas (Kavak + XS/XLS de ML)
# los outliers se excluyen del entrenamiento pero las publicaciones de Kavak se siguen evaluando (pueden ser oportunidades reales)
cand = pd.concat([df, SUSP[SUSP.kavak == 1]])
cand = cand[cand.version.isin(["XS", "XLS", "XLS Pack"]) | (cand.kavak == 1)].copy()
cand["precio_justo_hoy"] = predict(cand)                      # con sus propias características de canal
cand["valor_reventa_hoy"], cand["reventa_4a"] = project(cand)
cand["reventa_4a_p10"] = cand.reventa_4a * np.exp(-1.2816 * resid_sd)
cand["reventa_4a_p90"] = cand.reventa_4a * np.exp(1.2816 * resid_sd)
cand["perdida_4a_usd"] = cand.price_usd - cand.reventa_4a
cand["perdida_anual_usd"] = cand.perdida_4a_usd / HORIZON
cand["retencion_pct"] = cand.reventa_4a / cand.price_usd * 100
cand["descuento_vs_mercado_pct"] = (1 - cand.price_usd / cand.valor_reventa_hoy) * 100
cols = ["source", "year", "version", "transmission", "body", "km", "price_usd", "valor_reventa_hoy", "descuento_vs_mercado_pct",
        "reventa_4a", "reventa_4a_p10", "reventa_4a_p90", "perdida_4a_usd", "perdida_anual_usd", "retencion_pct", "location", "url"]
cand = cand.sort_values("perdida_4a_usd")
cand[cols].round(1).to_csv(O / "ranking_publicaciones.csv", index=False)
kv = cand[cand.kavak == 1]
P("# Ranking Kavak (menor pérdida esperada en 4 años primero)\n", kv[cols[:-2]].round(0).to_string(index=False), "\n")
pref = cand[(cand.version.isin(["XS", "XLS"])) & (cand.body == "hatchback")]
P("# Top 15 XS/XLS hatchback (Kavak + ML) por menor pérdida en 4 años\n", pref[cols[:-2] + ["url"]].head(15).round(0).to_string(index=False), "\n")
(O / "reporte.txt").write_text("\n".join(rep))
print("\n".join(rep[-6:]))
print(cv)
