import re, html, time, urllib.request, collections, csv
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
get=lambda u: urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=30).read().decode("utf-8","ignore")
rows=[]; base="https://autos.mercadolibre.com.ar/toyota/yaris/"
for off in range(1,1000,48):
    cs=get(base if off==1 else f"{base}_Desde_{off}_NoIndex_True").split('class="poly-card')[1:]
    if not cs: break
    for c in cs:
        g=lambda p:(lambda m: html.unescape(m.group(1)).strip() if m else "")(re.search(p,c))
        rows.append(dict(seller=g(r'poly-component__seller[^>]*>([^<]+)<'),title=g(r'poly-component__title[^>]*>([^<]+)<'),cur=g(r'andes-money-amount__currency-symbol[^>]*>([^<]+)<'),price=g(r'andes-money-amount__fraction[^>]*>([^<]+)<'),attrs=" | ".join(re.findall(r'poly-attributes_list__item[^>]*>([^<]+)<',c)),loc=g(r'poly-component__location[^>]*>([^<]+)<'),url=g(r'href="(https://(?:auto|articulo)\.mercadolibre\.com\.ar/MLA-?\d+[^"#]*)')))
    time.sleep(1)
w=csv.DictWriter(open("data/ml_vendedores.csv","w"),rows[0].keys()); w.writeheader(); w.writerows(rows)
cnt=collections.Counter(r["seller"] for r in rows); print(len(rows), sum(1 for r in rows if r['seller'])); print(cnt.most_common(40))
