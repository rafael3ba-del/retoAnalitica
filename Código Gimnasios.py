import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("escenario_gimnasios_15.csv", encoding="utf-8-sig")

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)

print("\nValores faltantes por columna\n", df.isna().sum())

desc = df.select_dtypes("number").describe().round(2)
print("\nDescripción numérica:\n", desc.to_string())

var1 = "Ingresos_Mensuales"
var2 = "Precio_Membresia"

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
plt.savefig("boxplot_ingresos.png", dpi=120)
plt.show()
plt.close()

plt.figure()
df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("hist_ingresos.png", dpi=120)
plt.show()
plt.close()

plt.figure()
df[var2].plot(kind="box", title=f"Boxplot - {var2}")
plt.tight_layout()
plt.savefig("boxplot_precio_membresia.png", dpi=120)
plt.show()
plt.close()

num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)
print("\nMatriz de correlación:\n", corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="plasma", fmt=".2f", square=True)
plt.title("Heatmap de correlaciones (variables numéricas)")
plt.tight_layout()
plt.savefig("heatmap_correlaciones_gym.png", dpi=120)
plt.show()
plt.close()

if var1 in corr.columns:
    top_corr = corr[var1].dropna().abs().sort_values(ascending=False).head(6)
    print(f"\nCorrelaciones más fuertes (absolutas) con {var1}:\n", top_corr)
    
X = df[[var1, var2]].dropna().values
k = 3
np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

for _ in range(10):
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    labels = distances.argmin(axis=1)
    
    for i in range(k):
        if np.any(labels == i):
            centroids[i] = X[labels == i].mean(axis=0)

df_clusters = df[[var1, var2]].copy()
df_clusters = df_clusters.dropna().reset_index(drop=True)
df_clusters['Cluster'] = labels

print("\n---Primeros 10 registros con Cluster---")
print(df_clusters.head(10))

plt.figure(figsize=(8,6))
colors = ['yellow', 'orange', 'purple']
for i in range(k):
    cluster_points = df_clusters[df_clusters['Cluster'] == i]
    plt.scatter(cluster_points[var1], cluster_points[var2], 
                color=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='X', s=100, label='Centroides')
plt.xlabel(var1)
plt.ylabel(var2)
plt.title("Clusters de Gimnasios (KMeans)")
plt.legend()
plt.tight_layout()
plt.savefig("clusters_gimnasios.png", dpi=120)
plt.show()
plt.close()

print("\nListo.")
print("Imágenes guardadas:")
print(" - boxplot_ingresos.png")
print(" - hist_ingresos.png")
print(" - boxplot_precio_membresia.png")
print(" - heatmap_correlaciones_gym.png")
print(" - clusters_gimnasios.png")