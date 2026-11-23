"""Scraper de publicaciones de Toyota Yaris en MercadoLibre Argentina y Kavak Argentina.
Uso: python scripts/scrape.py  -> escribe data/raw_ml.csv y data/raw_kavak.csv
"""
import csv, re, time, html, sys, datetime, urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
OUT = Path(__file__).resolve().parent.parent / "data"
TODAY = datetime.date.today().isoformat()

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "es-AR,es;q=0.9"})
    for i in range(3):
        try:
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        except Exception as e:
            print("retry", url, e, file=sys.stderr); time.sleep(2 + 2 * i)
    return ""

def strip(s):
    s = re.sub(r"<[^>]+>", " | ", s)
    return [t.strip() for t in re.split(r"(?:\s*\|\s*)+", html.unescape(s)) if t.strip()]

def scrape_ml():
    rows, seen = [], set()
    base = "https://autos.mercadolibre.com.ar/toyota/yaris/"
    for off in range(1, 2000, 48):
        url = base if off == 1 else f"{base}_Desde_{off}_NoIndex_True"
        h = get(url)
        cards = h.split('class="poly-card')[1:]
        if not cards: break
        new = 0
        for c in cards:
            m = re.search(r'href="(https://(?:auto|articulo)\.mercadolibre\.com\.ar/(MLA-?\d+)[^"]*)"', c)
            mid = m.group(2).replace("-", "") if m else None
            t = re.search(r'poly-component__title[^>]*>([^<]+)<', c)
            if not t or not mid or mid in seen: continue
            seen.add(mid); new += 1
            title = html.unescape(t.group(1)).strip()
            cur = re.search(r'andes-money-amount__currency-symbol[^>]*>([^<]+)<', c)
            frac = re.search(r'andes-money-amount__fraction[^>]*>([^<]+)<', c)
            attrs = re.findall(r'poly-attributes_list__item[^>]*>([^<]+)<', c)
            loc = re.search(r'poly-component__location[^>]*>([^<]+)<', c)
            year = next((a for a in attrs if re.fullmatch(r"\d{4}", a.strip())), "")
            km = next((a for a in attrs if "km" in a.lower()), "")
            rows.append(dict(source="mercadolibre", id=mid, title=title,
                currency=html.unescape(cur.group(1)).strip() if cur else "",
                price=frac.group(1).replace(".", "") if frac else "",
                year=year, km=re.sub(r"\D", "", km), location=html.unescape(loc.group(1)).strip() if loc else "",
                url=m.group(1).split("#")[0], scraped=TODAY))
        print(f"ML offset {off}: {new} nuevas (total {len(rows)})", file=sys.stderr)
        if new == 0: break
        time.sleep(1.5)
    return rows

def scrape_kavak():
    rows, seen = [], set()
    for path in ["toyota/yaris", "yaris", "toyota/yaris/automatico", "toyota/yaris/sedan", "toyota/yaris/hatchback",
                 "toyota/yaris/manual", "toyota/yaris/buenos-aires", "toyota/yaris/cordoba", "toyota/yaris/santa-fe", "toyota/yaris/mendoza", "toyota/yaris/tucuman"]:
        for page in range(0, 10):
            url = f"https://www.kavak.com/ar/usados/{path}" + (f"?page={page}" if page else "")
            h = get(url).replace('\\"', '"')
            items = re.findall(r'"url":"(https://www\.kavak\.com/ar/venta/[^"]+)".{0,1500}?"subtitle":"([^"]+)".{0,400}?"priceTop":"([^"]*)".{0,60}?"mainPrice":"([^"]*)".{0,300}?"footerInfo":"([^"]*)".{0,200}?"car_id":"(\d+)"', h)
            new = 0
            for u, sub, ptop, price, loc, cid in items:
                cash = re.sub(r"\D", "", ptop) or price.replace(".", "")
                if cid in seen: continue
                seen.add(cid); new += 1
                parts = [p.strip() for p in sub.split("•")]
                rows.append(dict(source="kavak", id=cid, title="Toyota Yaris " + " ".join(parts[2:3]) + (" hatchback" if "hatchback" in u else " sedan" if "sedan" in u else ""),
                    currency="$", price=cash, price_financed=price.replace(".", ""), year=parts[0], km=re.sub(r"\D", "", parts[1]),
                    location=loc, url=u, scraped=TODAY, transmission_raw=parts[3] if len(parts) > 3 else ""))
            print(f"Kavak {path} p{page}: {new} nuevas (total {len(rows)})", file=sys.stderr)
            if new == 0: break
            time.sleep(1)
    return rows

def write(rows, name):
    if not rows: return
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} filas", file=sys.stderr)

if __name__ == "__main__":
    write(scrape_kavak(), "raw_kavak.csv")
    write(scrape_ml(), "raw_ml.csv")
