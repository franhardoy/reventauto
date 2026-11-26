import pandas as pd, numpy as np, statsmodels.formula.api as smf
pd.set_option('display.width',260); pd.set_option('display.max_colwidth',95)
s=pd.read_csv('data/ml_vendedores.csv').drop_duplicates('url').rename(columns={'attrs':'atr'})
OF=['Toyota Usados','Toyota KANSAI S.A.','Treos Toyota','Zento Toyota','Toyota Panamericana','TOYOTA DEL PILAR','Hauswagen','Asahi Motors','Kavak']
s=s[s.seller.isin(OF)].copy()
s['year']=s.atr.str.extract(r'(\d{4})')[0].astype(float); s['km']=s.atr.str.extract(r'([\d\.]+) Km')[0].str.replace('.','').astype(float)
s['p']=s.price.str.replace('.','').astype(float); s['usd']=np.where(s.cur.str.contains('US'),s.p,s.p/1700)
t=s.title.str.lower()
s['version']=np.select([t.str.contains(r'\bgr\b'),t.str.contains('cross'),t.str.contains('xls pack'),t.str.contains(r'\bxls'),t.str.contains(r'\bxs\b'),t.str.contains(r'\bs\b')],['GR','Cross','XLS Pack','XLS','XS','S'],'ND')
s['trans']=np.where(t.str.contains(r'cvt|\bat\b|autom'),'automatico','manual'); s['body']=np.where(t.str.contains('sedan|4p|4 p'),'sedan','hatchback')
u=s[(s.km>500)&~s.version.isin(['GR','Cross'])].copy()
d=pd.read_csv('data/yaris_listings.csv'); d['log_price']=np.log(d.price_usd); d['km10k']=d.km/1e4; d['kavak']=(d.source=='kavak').astype(int)
d['manual']=(d.transmission=='manual').astype(int); d['sedan']=(d.body=='sedan').astype(int); d['interior']=(d.region=='Interior').astype(int)
F="log_price ~ age + I(age**2) + km10k + C(version, Treatment('XS')) + manual + manual:age + sedan + sedan:age + kavak + listed_in_usd + interior + is_new"
m=smf.ols(F,d).fit(); z=m.resid/m.resid.std(); m=smf.ols(F,d[z.abs()<=3]).fit(); sm=np.mean(np.exp(m.resid))
X=pd.DataFrame(dict(age=(2026-u.year).clip(lower=0),km10k=u.km/1e4,version=u.version.replace('ND','XLS'),manual=(u.trans=='manual').astype(int),sedan=(u.body=='sedan').astype(int),kavak=0,listed_in_usd=0,interior=0,is_new=0))
u['valor']=np.exp(m.predict(X))*sm; F_=X.copy(); F_['age']+=4; F_['km10k']+=4.8
u['rev4']=np.exp(m.predict(F_))*sm; u['desc']=(1-u.usd/u.valor)*100; u['perdida']=u.usd-u.rev4
print(len(s), len(u), u.seller.value_counts().to_dict())
print(u.sort_values('usd')[['seller','year','version','trans','body','km','cur','price','usd','valor','desc','rev4','perdida','loc']].round(0).to_string(index=False))
u.to_csv('outputs/concesionarios_oficiales.csv',index=False)
