"""
EJERCICIO 2 — KNN CON DISTANCIA EUCLIDIANA
============================================
Adaptado del ejercicio de libreta con familias (Gasto, Miembros → Clase).
Ahora usando clientes reales de la BD con total_venta y cantidad_productos.

Problema:
    Dado un nuevo cliente con gasto=$X y cantidad_productos=Y,
    clasificarlo como tipo "Alto" o "Bajo" gasto usando KNN con K=3
    y distancia Euclidiana.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine

matplotlib.use('Agg')

engine = create_engine('postgresql+psycopg2://postgres:mi_password@localhost:5432/postgres')


def knn_euclidiana():

    print("=" * 70)
    print("EJERCICIO 2 — KNN CON DISTANCIA EUCLIDIANA")
    print("Clasificación de clientes por gasto y cantidad de productos")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # 1. OBTENER DATOS DE CLIENTES DESDE LA BD
    # ═══════════════════════════════════════════════════════════════
    print("\nConsultando base de datos...")

    query = """
        SELECT id_cliente,
               AVG(total_venta) AS gasto_promedio,
               AVG(cantidad_productos) AS productos_promedio,
               COUNT(*) AS num_compras
        FROM ventas
        GROUP BY id_cliente
        HAVING COUNT(*) >= 3
        ORDER BY RANDOM()
        LIMIT 200
    """
    df_clientes = pd.read_sql(query, engine)
    print(f"Clientes obtenidos: {len(df_clientes)}")

    # ═══════════════════════════════════════════════════════════════
    # 2. CLASIFICAR CLIENTES (Gasto Alto vs Gasto Bajo)
    # ═══════════════════════════════════════════════════════════════
    mediana_gasto = df_clientes['gasto_promedio'].median()
    df_clientes['clase'] = df_clientes['gasto_promedio'].apply(
        lambda x: 'Gasto Alto' if x >= mediana_gasto else 'Gasto Bajo'
    )

    print(f"\nMediana de gasto: ${mediana_gasto:,.2f}")
    print(f"  Gasto Alto: {(df_clientes['clase'] == 'Gasto Alto').sum()} clientes")
    print(f"  Gasto Bajo: {(df_clientes['clase'] == 'Gasto Bajo').sum()} clientes")

    # ═══════════════════════════════════════════════════════════════
    # 3. SELECCIONAR MUESTRA PARA EJEMPLO (como en la libreta)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("DATOS DEL PROBLEMA (como en la libreta)")
    print(f"{'─' * 70}")

    # Tomar 6 clientes como ejemplo
    muestra = df_clientes.head(6).copy()
    muestra = muestra.reset_index(drop=True)
    muestra.index += 1

    print(f"\n  Clientes conocidos:")
    print(f"  {'#':<4} {'Gasto Prom':>12} {'Productos Prom':>16} {'Clase':<12}")
    print(f"  {'─'*4} {'─'*12} {'─'*16} {'─'*12}")
    for idx, row in muestra.iterrows():
        print(f"  {idx:<4} ${row['gasto_promedio']:>10,.2f} {row['productos_promedio']:>14.1f}   {row['clase']}")

    # Nuevo cliente a clasificar (punto medio)
    nuevo_gasto = muestra['gasto_promedio'].mean()
    nuevo_productos = muestra['productos_promedio'].mean()

    print(f"\n  Nuevo cliente a clasificar:")
    print(f"    Gasto promedio:    ${nuevo_gasto:,.2f}")
    print(f"    Productos promedio: {nuevo_productos:.1f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. CALCULAR DISTANCIA EUCLIDIANA (paso a paso)
    # ═══════════════════════════════════════════════════════════════
    K = 3
    print(f"\n{'─' * 70}")
    print(f"PASO A PASO — KNN con K={K} y Distancia Euclidiana")
    print(f"{'─' * 70}")
    print(f"\n  Fórmula: d = √[(gasto₁ - gasto₂)² + (productos₁ - productos₂)²]")

    distancias = []
    for idx, row in muestra.iterrows():
        diff_gasto = nuevo_gasto - row['gasto_promedio']
        diff_prod = nuevo_productos - row['productos_promedio']
        distancia = np.sqrt(diff_gasto**2 + diff_prod**2)

        distancias.append({
            'cliente': idx,
            'gasto': row['gasto_promedio'],
            'productos': row['productos_promedio'],
            'clase': row['clase'],
            'distancia': distancia
        })

        print(f"\n  Distancia al Cliente {idx} [{row['gasto_promedio']:,.2f}, {row['productos_promedio']:.1f}]:")
        print(f"    d = √[({nuevo_gasto:,.2f} - {row['gasto_promedio']:,.2f})² + ({nuevo_productos:.1f} - {row['productos_promedio']:.1f})²]")
        print(f"    d = √[({diff_gasto:,.2f})² + ({diff_prod:.1f})²]")
        print(f"    d = √[{diff_gasto**2:,.2f} + {diff_prod**2:.2f}]")
        print(f"    d = √[{diff_gasto**2 + diff_prod**2:,.2f}]")
        print(f"    d = {distancia:,.2f}  →  Clase: {row['clase']}")

    df_dist = pd.DataFrame(distancias).sort_values('distancia')

    # ═══════════════════════════════════════════════════════════════
    # 5. SELECCIONAR K VECINOS MÁS CERCANOS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"RESULTADO — Los {K} vecinos más cercanos")
    print(f"{'─' * 70}")

    vecinos = df_dist.head(K)
    print(f"\n  {'Cliente':<10} {'Distancia':>12} {'Clase':<15}")
    print(f"  {'─'*10} {'─'*12} {'─'*15}")
    for _, row in vecinos.iterrows():
        print(f"  Cliente {int(row['cliente']):<5} {row['distancia']:>12,.2f}   {row['clase']}")

    # Votación
    votos = vecinos['clase'].value_counts()
    print(f"\n  Votación:")
    for clase, count in votos.items():
        print(f"    {clase}: {count} voto(s)")

    clase_final = votos.index[0]
    print(f"\n  ► CLASIFICACIÓN: El nuevo cliente pertenece a → {clase_final}")

    # ═══════════════════════════════════════════════════════════════
    # 6. GRÁFICA — Scatter con vecinos resaltados
    # ═══════════════════════════════════════════════════════════════
    print("\nGenerando gráficas...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Todos los clientes de la muestra
    for clase, color, marker in [('Gasto Alto', '#E74C3C', 'o'), ('Gasto Bajo', '#3498DB', 's')]:
        mask = muestra['clase'] == clase
        ax.scatter(muestra.loc[mask, 'gasto_promedio'], muestra.loc[mask, 'productos_promedio'],
                   c=color, marker=marker, s=120, label=clase, edgecolors='white', linewidth=1.5, zorder=3)

    # Nuevo cliente
    ax.scatter(nuevo_gasto, nuevo_productos, c='#2ECC71', marker='*', s=400,
               label=f'Nuevo Cliente → {clase_final}', edgecolors='black', linewidth=2, zorder=5)

    # Líneas a los K vecinos
    for _, row in vecinos.iterrows():
        ax.plot([nuevo_gasto, row['gasto']], [nuevo_productos, row['productos']],
                'k--', alpha=0.4, linewidth=1)
        ax.annotate(f'd={row["distancia"]:,.0f}',
                    xy=((nuevo_gasto + row['gasto'])/2, (nuevo_productos + row['productos'])/2),
                    fontsize=8, ha='center', style='italic', color='gray')

    # Etiquetas de clientes
    for idx, row in muestra.iterrows():
        ax.annotate(f'C{idx}', (row['gasto_promedio'], row['productos_promedio']),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight='bold')

    ax.set_xlabel('Gasto Promedio ($)', fontsize=12)
    ax.set_ylabel('Productos Promedio por Compra', fontsize=12)
    ax.set_title(f'KNN (K={K}) — Clasificación de Nuevo Cliente (Distancia Euclidiana)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ejercicio2_knn_euclidiana.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio2_knn_euclidiana.png")

    # ═══════════════════════════════════════════════════════════════
    # 7. GRÁFICA 2 — KNN con todos los 200 clientes
    # ═══════════════════════════════════════════════════════════════
    # Calcular distancias de todos los 200 clientes
    df_clientes['distancia'] = np.sqrt(
        (df_clientes['gasto_promedio'] - nuevo_gasto)**2 +
        (df_clientes['productos_promedio'] - nuevo_productos)**2
    )
    df_clientes_sorted = df_clientes.sort_values('distancia')
    vecinos_full = df_clientes_sorted.head(K)
    votos_full = vecinos_full['clase'].value_counts()
    clase_full = votos_full.index[0]

    fig, ax = plt.subplots(figsize=(14, 9))
    for clase, color, marker in [('Gasto Alto', '#E74C3C', 'o'), ('Gasto Bajo', '#3498DB', 's')]:
        mask = df_clientes['clase'] == clase
        ax.scatter(df_clientes.loc[mask, 'gasto_promedio'], df_clientes.loc[mask, 'productos_promedio'],
                   c=color, marker=marker, s=40, label=clase, alpha=0.5, edgecolors='white', linewidth=0.5)

    ax.scatter(nuevo_gasto, nuevo_productos, c='#2ECC71', marker='*', s=500,
               label=f'Nuevo → {clase_full}', edgecolors='black', linewidth=2, zorder=5)

    # Círculo de radio = distancia al K-ésimo vecino
    radio = vecinos_full.iloc[-1]['distancia']
    circle = plt.Circle((nuevo_gasto, nuevo_productos), radio, fill=False,
                         linestyle='--', color='green', linewidth=2, alpha=0.7)
    ax.add_patch(circle)

    for _, row in vecinos_full.iterrows():
        ax.plot([nuevo_gasto, row['gasto_promedio']], [nuevo_productos, row['productos_promedio']],
                'g-', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Gasto Promedio ($)', fontsize=12)
    ax.set_ylabel('Productos Promedio', fontsize=12)
    ax.set_title(f'KNN (K={K}) — 200 Clientes Reales de la BD',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('ejercicio2_knn_200_clientes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio2_knn_200_clientes.png")

    print(f"\n{'=' * 70}")
    print("EJERCICIO 2 COMPLETADO")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    knn_euclidiana()
