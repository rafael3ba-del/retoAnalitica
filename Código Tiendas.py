import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("escenario_tiendas_15.csv", encoding="utf-8-sig")

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)

print("\nValores faltantes por columna\n", df.isna().sum())

desc = df.select_dtypes("number").describe().round(2)
print("\nDescripción numérica:\n", desc.to_string())

var1 = "Ventas_Mensuales"
var2 = "Precio_Promedio"

media_var1 = df[var1].mean(skipna=True)
media_var2 = df[var2].mean(skipna=True)

mediana_var1 = df[var1].median(skipna=True)
mediana_var2 = df[var2].median(skipna=True)

moda_var1 = df[var1].mode(dropna=True)
moda_var2 = df[var2].mode(dropna=True)
moda_var1 = moda_var1.iloc[0] if not moda_var1.empty else None
moda_var2 = moda_var2.iloc[0] if not moda_var2.empty else None

print("\n---Indicadores de Tendencia Central---")
print(f"{var1} -> media: {media_var1:.2f}, mediana: {mediana_var1:.2f}, moda: {moda_var1}")
print(f"{var2} -> media: {media_var2:.2f}, mediana: {mediana_var2:.2f}, moda: {moda_var2}")

plt.figure()
df[var1].plot(kind="box", title=f"Boxplot - {var1}")
plt.tight_layout()
plt.savefig("boxplot_ventas.png", dpi=120)
plt.show()
plt.close()

plt.figure()
df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("hist_ventas.png", dpi=120)
plt.show()
plt.close()

plt.figure()
df[var2].plot(kind="box", title=f"Boxplot - {var2}")
plt.tight_layout()
plt.savefig("boxplot_precio_promedio.png", dpi=120)
plt.show()
plt.close()

num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)
print("\nMatriz de correlación:\n", corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="plasma", fmt=".2f", square=True)
plt.title("Heatmap de correlaciones (variables numéricas)")
plt.tight_layout()
plt.savefig("heatmap_correlaciones.png", dpi=120)
plt.show()
plt.close()

if var1 in corr.columns:
    top_corr = corr[var1].dropna().abs().sort_values(ascending=False).head(6)
    print(f"\nCorrelaciones más fuertes (absolutas) con {var1}:\n", top_corr)
    
print("\nListo.")
print("Imágenes guardadas:")
print(" - boxplot_ventas.png")
print(" - hist_ventas.png")
print(" - boxplot_precio_promedio.png")
print(" - heatmap_correlaciones.png")