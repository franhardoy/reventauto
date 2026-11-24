"""Limpieza y unificación de publicaciones -> data/yaris_listings.csv"""
import re, numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "data"
MEP = 1700.0          # ARS por USD, dólar MEP 23/11/2026
REF_YEAR = 2026

ml = pd.read_csv(D / "raw_ml.csv", dtype={"id": str})
kv = pd.read_csv(D / "raw_kavak.csv", dtype={"id": str})
kv = kv[kv.url.str.contains("/venta/toyota-yaris")].copy()
df = pd.concat([ml, kv], ignore_index=True)
t = df.title.str.lower()
df["transmission_raw"] = df.get("transmission_raw")

def version(s):
    if re.search(r"\bgr\b", s): return "GR"
    if re.search(r"xls\s*\+|xls plus", s): return "XLS"
    if "xls pack" in s or "xls-pack" in s: return "XLS Pack"
    if re.search(r"\bxls\b", s): return "XLS"
    if re.search(r"\bxs\b", s): return "XS"
    if re.search(r"\bs\b", s.replace("1.5", "")): return "S"
    return "ND"
df["version"] = t.map(version)
df["body"] = np.where(t.str.contains("sedan|sedán|4p|4 p"), "sedan", "hatchback")
auto_title = t.str.contains(r"cvt|\bat\b|autom")
manual_title = t.str.contains(r"6m|m/t|\bmt\b|manual")
df["transmission"] = np.where(df.transmission_raw.fillna("").str.contains("Autom"), "automatico",
                     np.where(df.transmission_raw.fillna("").str.contains("Manual"), "manual",
                     np.where(auto_title & ~manual_title, "automatico", "manual")))
# S y XLS Pack sólo existen con CVT
df.loc[df.version.isin(["S", "XLS Pack"]), "transmission"] = "automatico"

df["price"] = pd.to_numeric(df.price, errors="coerce")
df["year"] = pd.to_numeric(df.year, errors="coerce")
df["km"] = pd.to_numeric(df.km, errors="coerce")
df["currency"] = df.currency.str.replace(r"U$S", "US$", regex=False)
df["listed_in_usd"] = (df.currency == "US$").astype(int)
df["price_usd"] = np.where(df.listed_in_usd == 1, df.price, df.price / MEP)
df["age"] = (REF_YEAR - df.year).clip(lower=0)
loc = df.location.fillna("").str.lower()
df["region"] = np.where(loc.str.contains("capital federal|buenos aires|gba|caba"), "AMBA", "Interior")
df["is_new"] = (df.km <= 100).astype(int)

n0 = len(df)
reasons = {}
def drop(mask, why):
    global df
    reasons[why] = int(mask.sum()); df = df[~mask]
drop(df.version == "GR", "GR Yaris (otro segmento)")
drop(df.year < 2016, "generación anterior (<2016)")
drop(df.price_usd.isna() | (df.price_usd < 8000) | (df.price_usd > 45000), "precio fuera de rango (anticipos/cuotas/typos)")
drop(df.km.isna() | (df.km > 350000), "km inválido")
drop((df.age >= 2) & (df.km < 500), "usado con km=0 (dato faltante)")
# precio absurdo relativo a su año (> 2.2 desvíos robustos de la mediana de su año)
med = df.groupby("year").price_usd.transform("median")
mad = df.groupby("year").price_usd.transform(lambda s: (s - s.median()).abs().median())
drop((df.price_usd - med).abs() > 4 * 1.4826 * mad.clip(lower=500), "outlier de precio dentro de su año")
df = df.drop_duplicates(subset=["source", "year", "km", "price_usd", "version"])
# autos de Kavak re-publicados en ML (mismo año y km): se queda la versión de Kavak
kv_keys = set(zip(df[df.source == "kavak"].year, df[df.source == "kavak"].km))
kavak_dup = (df.source == "mercadolibre") & pd.Series([(y, k) in kv_keys for y, k in zip(df.year, df.km)], index=df.index) & (df.km > 1000)
reasons["duplicado de Kavak en ML"] = int(kavak_dup.sum()); df = df[~kavak_dup]
print(f"{n0} -> {len(df)} filas"); [print(f"  - {k}: {v}") for k, v in reasons.items()]
cols = ["source", "id", "title", "year", "age", "version", "body", "transmission", "km", "is_new",
        "currency", "price", "listed_in_usd", "price_usd", "location", "region", "url", "scraped"]
df[cols].sort_values(["year", "km"]).to_csv(D / "yaris_listings.csv", index=False)
print(df.groupby(["version", "transmission"]).size().unstack(fill_value=0))
print(df.groupby("body").size()); print(df.source.value_counts())
