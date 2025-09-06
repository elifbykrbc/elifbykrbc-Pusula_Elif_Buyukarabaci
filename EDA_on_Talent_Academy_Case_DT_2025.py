import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer, LabelEncoder

df = pd.read_excel('Talent_Academy_Case_DT_2025.xlsx')

df['UygulamaSuresi'] = df['UygulamaSuresi'].astype(str).str.replace('Dakika', '', case=False).str.strip()

def convert_to_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        return [int(p) for p in x.split(",") if p.strip().isdigit()]
    if isinstance(x, (list, tuple)):
        return [int(i) for i in x if str(i).isdigit()]
    return []

df['UygulamaSuresi_list'] = df['UygulamaSuresi'].apply(convert_to_list)
df['Seans_Sayisi'] = df['UygulamaSuresi_list'].apply(len)
df['Toplam_Uygulama_Suresi'] = df['UygulamaSuresi_list'].apply(sum)
df['Ortalama_Uygulama_Suresi'] = df['UygulamaSuresi_list'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

df['TedaviSuresi'] = df['TedaviSuresi'].astype(str).str.extract(r'(\d+)')
df['TedaviSuresi'] = pd.to_numeric(df['TedaviSuresi'], errors='coerce')

numeric_cols = ['Yas', 'TedaviSuresi', 'Seans_Sayisi',
                'Toplam_Uygulama_Suresi', 'Ortalama_Uygulama_Suresi']

imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
                         columns=[f"{c}_scaled" for c in numeric_cols],
                         index=df.index)

def split_list(x):
    if pd.isna(x):
        return []
    return [item.strip() for item in str(x).split(',')]

mlb_kronik = MultiLabelBinarizer()
kronik_encoded = mlb_kronik.fit_transform(df['KronikHastalik'].apply(split_list))
df_kronik = pd.DataFrame(kronik_encoded, columns=[f"Kronik_{c}" for c in mlb_kronik.classes_], index=df.index)

df['Alerji_Var'] = df['Alerji'].apply(lambda x: 0 if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else 1)

df['Tanilar'] = df['Tanilar'].fillna('NaN')
df['TedaviAdi'] = df['TedaviAdi'].fillna('NaN')
df['Tanilar_enc'] = LabelEncoder().fit_transform(df['Tanilar'])
df['TedaviAdi_enc'] = LabelEncoder().fit_transform(df['TedaviAdi'])
df['UygulamaYerleri'] = df['UygulamaYerleri'].fillna('NaN')
df['UygulamaYerleri_enc'] = LabelEncoder().fit_transform(df['UygulamaYerleri'].astype(str))

categorical_cols = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum']
df[categorical_cols] = df[categorical_cols].fillna('NaN')

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[categorical_cols])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

df_final = pd.concat([
    df[['HastaNo']],
    df_scaled,
    df_encoded,
    df[['Tanilar_enc', 'TedaviAdi_enc', 'UygulamaYerleri_enc']],
    df_kronik,
    df[['Alerji_Var']]
], axis=1)

df_final.to_excel("hasta_model_ready.xlsx", index=False)
print("Dosya başarıyla 'hasta_model_ready.xlsx' olarak kaydedildi.")

df[numeric_cols].hist(figsize=(10,8), bins=20)
plt.suptitle("Sayısal Değişkenlerin Histogramları")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Yas', y='TedaviSuresi', data=df, hue='Cinsiyet')
plt.title("Yaş ve Tedavi Süresi İlişkisi (Cinsiyet Bazlı)")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Cinsiyet', y='TedaviSuresi', data=df)
plt.title("Cinsiyet ve Tedavi Süresi")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Sayısal Değişkenler Korelasyonu")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Seans_Sayisi', y='TedaviSuresi', hue='Cinsiyet', data=df)
plt.title("Cinsiyet ve Seans Sayısına Göre Tedavi Süresi")
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x='Bolum', y='TedaviSuresi', hue='Cinsiyet', data=df, ci=None)
plt.title("Cinsiyet ve Bölüme Göre Ortalama Tedavi Süresi")
plt.xticks(rotation=45)
plt.show()

df['KronikSayisi'] = df['KronikHastalik'].apply(lambda x: len(split_list(x)))
plt.figure(figsize=(10,6))
sns.boxplot(x='KronikSayisi', y='TedaviSuresi', hue='Cinsiyet', data=df)
plt.title("Kronik Hastalık Sayısı ve Cinsiyet'e Göre Tedavi Süresi")
plt.show()

