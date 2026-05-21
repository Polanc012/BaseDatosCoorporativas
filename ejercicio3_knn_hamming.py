"""
EJERCICIO 3 — KNN CON DISTANCIA DE HAMMING (Recomendación de Productos)
========================================================================
Cada cliente se representa como un vector binario de 10 posiciones
(1 si compró productos de esa categoría, 0 si no).
Dado un cliente X, encontrar el vecino más similar (K=1)
usando distancia de Hamming y recomendar las categorías que le faltan.

NOTA: Se analizan VENTAS INDIVIDUALES (no el historial completo del cliente)
para tener vectores con más variación entre sí.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine

matplotlib.use('Agg')

engine = create_engine('postgresql+psycopg2://postgres:mi_password@localhost:5432/postgres')

NOMBRES_CORTOS = {
    'Baby Care': 'Baby', 'Bakery, Cakes & Dairy': 'Bakery',
    'Beauty & Hygiene': 'Beauty', 'Beverages': 'Beverages',
    'Cleaning & Household': 'Cleaning', 'Eggs, Meat & Fish': 'Eggs/Meat',
    'Foodgrains, Oil & Masala': 'Foodgrains', 'Fruits & Vegetables': 'Fruits/Veg',
    'Gourmet & World Food': 'Gourmet', 'Snacks & Branded Foods': 'Snacks',
}


def knn_hamming():

    print("=" * 70)
    print("EJERCICIO 3 — KNN CON DISTANCIA DE HAMMING")
    print("Recomendación de categorías a clientes")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # 1. OBTENER VENTAS INDIVIDUALES CON SUS CATEGORÍAS
    #    (Usamos ventas individuales, no el historial completo,
    #     para que haya diferencias reales entre vectores)
    # ═══════════════════════════════════════════════════════════════
    print("\nConsultando base de datos...")

    query = """
        SELECT d.id_venta, a.category
        FROM detalle_ventas d
        JOIN articulos a ON d.id_articulo = a.id_articulo
    """
    df = pd.read_sql(query, engine)

    # Categorías únicas por VENTA (no por cliente)
    venta_cats = df.groupby('id_venta')['category'].apply(
        lambda x: set(x.unique())
    ).reset_index()
    venta_cats.columns = ['id_venta', 'categorias']
    venta_cats['num_cats'] = venta_cats['categorias'].apply(len)

    todas_cats = sorted(df['category'].unique())
    cats_short = [NOMBRES_CORTOS.get(c, c[:8]) for c in todas_cats]

    print(f"Total ventas: {len(venta_cats):,}")
    print(f"Categorías: {len(todas_cats)}")
    print(f"Promedio categorías por venta: {venta_cats['num_cats'].mean():.1f}")

    # ═══════════════════════════════════════════════════════════════
    # 2. BUSCAR VENTAS CON DIFERENCIAS REALES
    #    Elegimos una venta X con 5-6 categorías y vecinos con 6-8
    #    para que haya categorías que recomendar
    # ═══════════════════════════════════════════════════════════════

    def to_vector(cats_set):
        return [1 if cat in cats_set else 0 for cat in todas_cats]

    # Buscar una buena venta X: que tenga 5-6 categorías (no todas)
    candidatos_x = venta_cats[(venta_cats['num_cats'] >= 5) & (venta_cats['num_cats'] <= 6)]

    # Buscar vecinos que tengan 7-9 categorías (más que X, para poder recomendar)
    candidatos_vecinos = venta_cats[(venta_cats['num_cats'] >= 7) & (venta_cats['num_cats'] <= 9)]

    print(f"\nVentas con 5-6 categorías (candidatos X): {len(candidatos_x):,}")
    print(f"Ventas con 7-9 categorías (candidatos vecinos): {len(candidatos_vecinos):,}")

    # Elegir X
    np.random.seed(123)
    venta_x = candidatos_x.sample(1).iloc[0]
    vector_x = to_vector(venta_x['categorias'])

    # Elegir 4 vecinos con diferentes distancias
    # Necesitamos que al menos 1 vecino tenga categorías que X no tiene
    mejores_vecinos = []

    for _, row in candidatos_vecinos.sample(min(500, len(candidatos_vecinos)), random_state=42).iterrows():
        v = to_vector(row['categorias'])
        hamming = sum(a != b for a, b in zip(vector_x, v))
        # Categorías que el vecino tiene y X no
        nuevas = sum(1 for j in range(len(vector_x)) if vector_x[j] == 0 and v[j] == 1)
        if hamming >= 1 and nuevas >= 1:
            mejores_vecinos.append({
                'id_venta': row['id_venta'],
                'categorias': row['categorias'],
                'vector': v,
                'hamming': hamming,
                'nuevas': nuevas,
            })

    # Ordenar: queremos variedad de distancias
    mejores_vecinos.sort(key=lambda x: x['hamming'])

    # Tomar 4 vecinos con distancias variadas
    seleccionados = []
    distancias_vistas = set()
    for v in mejores_vecinos:
        if len(seleccionados) >= 4:
            break
        if v['hamming'] not in distancias_vistas or len(seleccionados) < 2:
            seleccionados.append(v)
            distancias_vistas.add(v['hamming'])

    # Si no hay suficientes, rellenar
    if len(seleccionados) < 4:
        for v in mejores_vecinos:
            if v not in seleccionados:
                seleccionados.append(v)
            if len(seleccionados) >= 4:
                break

    # ═══════════════════════════════════════════════════════════════
    # 3. MOSTRAR EL PROBLEMA
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("DATOS DEL PROBLEMA")
    print(f"{'─' * 70}")
    print(f"\n  Representamos cada venta como vector binario de 10 categorías")
    print(f"  (1 si la venta incluye esa categoría, 0 si no)\n")
    print(f"  Categorías: {', '.join(cats_short)}\n")

    letras = ['A', 'B', 'C', 'D']
    for i, v in enumerate(seleccionados):
        cats_nombres = [cats_short[j] for j in range(len(v['vector'])) if v['vector'][j] == 1]
        print(f"  Venta {letras[i]}: ({','.join(map(str, v['vector']))}) → {{{', '.join(cats_nombres)}}}")

    cats_x = [cats_short[j] for j in range(len(vector_x)) if vector_x[j] == 1]
    cats_faltantes_x = [cats_short[j] for j in range(len(vector_x)) if vector_x[j] == 0]
    print(f"\n  Venta X: ({','.join(map(str, vector_x))}) → {{{', '.join(cats_x)}}}")
    print(f"  Le faltan: {{{', '.join(cats_faltantes_x)}}}")

    print(f"\n  Problema: Usando KNN con K=1 y Distancia de Hamming,")
    print(f"  encontrar la venta más similar a X y recomendar")
    print(f"  las categorías que esa venta tiene pero X no.")

    # ═══════════════════════════════════════════════════════════════
    # 4. CALCULAR DISTANCIA DE HAMMING (paso a paso)
    # ═══════════════════════════════════════════════════════════════
    K = 1
    print(f"\n{'─' * 70}")
    print(f"PASO A PASO — Distancia de Hamming")
    print(f"  (Número de posiciones donde los bits son diferentes)")
    print(f"{'─' * 70}")

    distancias = []
    for i, v in enumerate(seleccionados):
        diferencias = []
        for j in range(len(vector_x)):
            if vector_x[j] != v['vector'][j]:
                diferencias.append(j + 1)

        distancia = len(diferencias)
        distancias.append({
            'cliente': letras[i],
            'vector': v['vector'],
            'categorias': v['categorias'],
            'diferencias': diferencias,
            'distancia': distancia
        })

        print(f"\n  Venta {letras[i]} ({','.join(map(str, v['vector']))})")
        print(f"    Comparamos con X ({','.join(map(str, vector_x))})")
        if diferencias:
            pos_names = [cats_short[p-1] for p in diferencias]
            print(f"    Diferencias en posiciones: {diferencias} ({', '.join(pos_names)})")
        else:
            print(f"    Sin diferencias")
        print(f"    Distancia de Hamming = {distancia}")

    df_dist = pd.DataFrame(distancias).sort_values('distancia')

    # ═══════════════════════════════════════════════════════════════
    # 5. ENCONTRAR VECINO Y RECOMENDAR
    # ═══════════════════════════════════════════════════════════════
    vecino = df_dist.iloc[0]

    print(f"\n{'─' * 70}")
    print(f"RESULTADO — Vecino más cercano: Venta {vecino['cliente']} (distancia = {vecino['distancia']})")
    print(f"{'─' * 70}")

    cats_vecino = [cats_short[j] for j in range(len(vecino['vector'])) if vecino['vector'][j] == 1]
    print(f"\n  Venta X tiene:   {{{', '.join(cats_x)}}}")
    print(f"  Venta {vecino['cliente']} tiene: {{{', '.join(cats_vecino)}}}")

    recomendaciones = []
    for j in range(len(vector_x)):
        if vector_x[j] == 0 and vecino['vector'][j] == 1:
            recomendaciones.append(cats_short[j])

    if recomendaciones:
        print(f"\n  ► RECOMENDACIÓN: Se recomienda al cliente probar:")
        for r in recomendaciones:
            print(f"    ★ {r}")
        print(f"\n  Porque la venta más similar (Venta {vecino['cliente']}) incluye estas categorías")
        print(f"  y la Venta X aún no las tiene.")
    else:
        print(f"\n  ► No hay categorías nuevas para recomendar.")

    # ═══════════════════════════════════════════════════════════════
    # 6. GRÁFICA 1 — Vectores binarios comparados
    # ═══════════════════════════════════════════════════════════════
    print("\nGenerando gráficas...")

    labels = ['X'] + [d['cliente'] for d in distancias]
    vectors = [vector_x] + [d['vector'] for d in distancias]

    fig, ax = plt.subplots(figsize=(14, 5))
    data = np.array(vectors)

    im = ax.imshow(data, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(cats_short)))
    ax.set_xticklabels(cats_short, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f'Venta {l}' for l in labels], fontsize=11)

    for i in range(len(labels)):
        for j in range(len(cats_short)):
            color = 'white' if data[i][j] == 1 else 'gray'
            ax.text(j, i, str(int(data[i][j])), ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

    # Resaltar diferencias entre X y vecino ganador
    vecino_idx = labels.index(vecino['cliente'])
    for j in range(len(vector_x)):
        if vector_x[j] != vectors[vecino_idx][j]:
            rect = plt.Rectangle((j-0.5, -0.5), 1, 1, fill=False,
                                  edgecolor='red', linewidth=2.5)
            ax.add_patch(rect)
            rect2 = plt.Rectangle((j-0.5, vecino_idx-0.5), 1, 1, fill=False,
                                   edgecolor='red', linewidth=2.5)
            ax.add_patch(rect2)

    ax.set_title(f'VECTORES BINARIOS — Diferencias resaltadas en rojo (X vs {vecino["cliente"]})',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('ejercicio3_vectores_binarios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio3_vectores_binarios.png")

    # ═══════════════════════════════════════════════════════════════
    # 7. GRÁFICA 2 — Distancias de Hamming
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))
    clientes_labels = [f'Venta {row["cliente"]}' for _, row in df_dist.iterrows()]
    dists = [row['distancia'] for _, row in df_dist.iterrows()]
    colors = ['#2ECC71' if i == 0 else '#3498DB' for i in range(len(dists))]

    bars = ax.bar(clientes_labels, dists, color=colors, edgecolor='white', linewidth=1.5)
    for bar, d in zip(bars, dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(d), ha='center', fontsize=14, fontweight='bold')

    ax.set_ylabel('Distancia de Hamming', fontsize=12)
    ax.set_title(f'DISTANCIA DE HAMMING — Desde Venta X (K={K}, más cercano en verde)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('ejercicio3_distancias_hamming.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio3_distancias_hamming.png")

    # ═══════════════════════════════════════════════════════════════
    # 8. GRÁFICA 3 — Recomendación visual
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Lo que tiene X
    ax1 = axes[0]
    colors_x = ['#2ECC71' if v == 1 else '#FFB3B3' for v in vector_x]
    ax1.barh(cats_short, vector_x, color=colors_x, edgecolor='white')
    ax1.set_title('Venta X\n(tiene / no tiene)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.5)
    for i, v in enumerate(vector_x):
        ax1.text(v + 0.05, i, '✓ Tiene' if v else '✗ No tiene', va='center', fontsize=9)

    # Panel 2: Lo que tiene el vecino
    ax2 = axes[1]
    vals_v = vecino['vector']
    colors_v = ['#3498DB' if v == 1 else '#FFB3B3' for v in vals_v]
    ax2.barh(cats_short, vals_v, color=colors_v, edgecolor='white')
    ax2.set_title(f'Venta {vecino["cliente"]} (vecino más cercano)\n(tiene / no tiene)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.5)
    for i, v in enumerate(vals_v):
        ax2.text(v + 0.05, i, '✓ Tiene' if v else '✗ No tiene', va='center', fontsize=9)

    # Panel 3: Recomendación
    ax3 = axes[2]
    rec_vals = [1 if cats_short[j] in recomendaciones else 0 for j in range(len(cats_short))]
    colors_r = ['#E74C3C' if v == 1 else '#EEEEEE' for v in rec_vals]
    ax3.barh(cats_short, rec_vals, color=colors_r, edgecolor='white')
    ax3.set_title('► RECOMENDACIÓN\n(categorías sugeridas)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.5)
    for i, v in enumerate(rec_vals):
        if v == 1:
            ax3.text(v + 0.05, i, '★ Recomendar', va='center', fontsize=9, fontweight='bold', color='#E74C3C')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('KNN HAMMING — Sistema de Recomendación por Categorías', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ejercicio3_recomendacion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio3_recomendacion.png")

    print(f"\n{'=' * 70}")
    print("EJERCICIO 3 COMPLETADO")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    knn_hamming()
