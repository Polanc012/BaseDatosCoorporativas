import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import combinations
from sqlalchemy import create_engine

matplotlib.use('Agg')

engine = create_engine('postgresql+psycopg2://postgres:mi_password@localhost:5432/postgres')

# Nombres cortos para las categorías (para gráficas 3, 4, 5)
NOMBRES_CORTOS = {
    'Baby Care':                'Baby',
    'Bakery, Cakes & Dairy':    'Bakery',
    'Beauty & Hygiene':         'Beauty',
    'Beverages':                'Beverages',
    'Cleaning & Household':     'Cleaning',
    'Eggs, Meat & Fish':        'Eggs/Meat',
    'Foodgrains, Oil & Masala': 'Foodgrains',
    'Fruits & Vegetables':      'Fruits/Veg',
    'Gourmet & World Food':     'Gourmet',
    'Snacks & Branded Foods':   'Snacks',
}

COLORES_CAT = {
    'Baby Care':                '#FF6B6B',
    'Bakery, Cakes & Dairy':    '#FFA07A',
    'Beauty & Hygiene':         '#FFD93D',
    'Beverages':                '#6BCB77',
    'Cleaning & Household':     '#4D96FF',
    'Eggs, Meat & Fish':        '#9B59B6',
    'Foodgrains, Oil & Masala': '#FF8C00',
    'Fruits & Vegetables':      '#2ECC71',
    'Gourmet & World Food':     '#E74C3C',
    'Snacks & Branded Foods':   '#3498DB',
}


def acortar_nombre(categorias_list):
    """Convierte lista de categorías a etiqueta corta en una sola línea."""
    return ' + '.join([NOMBRES_CORTOS.get(c, c[:8]) for c in categorias_list])


def generar_itemsets_con_graficas():

    print("=" * 70)
    print("ITEMSETS POR CATEGORÍA + GRÁFICAS")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # 1. OBTENER CATEGORÍAS POR VENTA
    # ═══════════════════════════════════════════════════════════════
    print("\nConsultando base de datos...")

    query = """
        SELECT d.id_venta, a.category
        FROM detalle_ventas d
        JOIN articulos a ON d.id_articulo = a.id_articulo
    """
    df = pd.read_sql(query, engine)

    ventas_categorias = df.groupby('id_venta')['category'].apply(
        lambda x: frozenset(x.unique())
    ).reset_index()
    ventas_categorias.columns = ['id_venta', 'categorias']

    total_ventas = len(ventas_categorias)
    todas_categorias = sorted(df['category'].unique())

    print(f"Total ventas: {total_ventas:,}")
    print(f"Categorías: {len(todas_categorias)}")

    # ═══════════════════════════════════════════════════════════════
    # 2. CALCULAR ITEMSETS 1-5
    # ═══════════════════════════════════════════════════════════════
    resultados = {}

    for nivel in range(1, 6):
        print(f"\nCalculando itemset {nivel}...")
        combos = list(combinations(todas_categorias, nivel))
        itemset_data = []

        for combo in combos:
            combo_set = set(combo)
            count = ventas_categorias['categorias'].apply(
                lambda x: combo_set.issubset(x)
            ).sum()
            soporte = count / total_ventas

            itemset_data.append({
                'itemset': ' + '.join(combo),
                'categorias': list(combo),
                'ventas': count,
                'soporte': round(soporte, 4),
                'porcentaje': round(soporte * 100, 2),
            })

        df_itemset = pd.DataFrame(itemset_data).sort_values('ventas', ascending=False)
        resultados[nivel] = df_itemset
        print(f"  {len(combos)} combinaciones — Top: {df_itemset.iloc[0]['itemset']} ({df_itemset.iloc[0]['porcentaje']}%)")

    # ═══════════════════════════════════════════════════════════════
    # 3. GRÁFICA ITEMSET 1 — Barras por categoría individual
    # ═══════════════════════════════════════════════════════════════
    print("\nGenerando gráficas...")

    df1 = resultados[1].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    colores = [COLORES_CAT.get(cat, '#888') for cat in df1['itemset']]
    bars = ax.barh(df1['itemset'], df1['ventas'], color=colores, edgecolor='white', linewidth=0.5)
    for bar, pct in zip(bars, df1['porcentaje']):
        ax.text(bar.get_width() + total_ventas * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Cantidad de Ventas', fontsize=12)
    ax.set_title('ITEMSET 1 — Ventas por Categoría Individual', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, df1['ventas'].max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('itemset_1_grafica.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  itemset_1_grafica.png")

    # ═══════════════════════════════════════════════════════════════
    # 4. GRÁFICA ITEMSET 2 — Top 20 pares
    # ═══════════════════════════════════════════════════════════════
    df2 = resultados[2].head(20).copy()
    df2['label'] = df2['categorias'].apply(acortar_nombre)

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(df2['label'], df2['ventas'], color='#4D96FF', edgecolor='white', linewidth=0.5)
    for bar, pct in zip(bars, df2['porcentaje']):
        ax.text(bar.get_width() + total_ventas * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Cantidad de Ventas', fontsize=12)
    ax.set_title('ITEMSET 2 — Top 20 Pares de Categorías', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, df2['ventas'].max() * 1.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.savefig('itemset_2_grafica.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  itemset_2_grafica.png")

    # ═══════════════════════════════════════════════════════════════
    # 5. GRÁFICA ITEMSET 3 — Top 20 tríos
    # ═══════════════════════════════════════════════════════════════
    df3 = resultados[3].head(20).copy()
    df3['label'] = df3['categorias'].apply(acortar_nombre)

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(df3['label'], df3['ventas'], color='#6BCB77', edgecolor='white', linewidth=0.5, height=0.7)
    for bar, pct in zip(bars, df3['porcentaje']):
        ax.text(bar.get_width() + total_ventas * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Cantidad de Ventas', fontsize=12)
    ax.set_title('ITEMSET 3 — Top 20 Tríos de Categorías', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, df3['ventas'].max() * 1.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig('itemset_3_grafica.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  itemset_3_grafica.png")

    # ═══════════════════════════════════════════════════════════════
    # 6. GRÁFICA ITEMSET 4 — Top 15 cuartetos
    # ═══════════════════════════════════════════════════════════════
    df4 = resultados[4].head(15).copy()
    df4['label'] = df4['categorias'].apply(acortar_nombre)

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(df4['label'], df4['ventas'], color='#FFA07A', edgecolor='white', linewidth=0.5, height=0.7)
    for bar, pct in zip(bars, df4['porcentaje']):
        ax.text(bar.get_width() + total_ventas * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Cantidad de Ventas', fontsize=12)
    ax.set_title('ITEMSET 4 — Top 15 Cuartetos de Categorías', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, df4['ventas'].max() * 1.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig('itemset_4_grafica.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  itemset_4_grafica.png")

    # ═══════════════════════════════════════════════════════════════
    # 7. GRÁFICA ITEMSET 5 — Top 15 quintetos
    # ═══════════════════════════════════════════════════════════════
    df5 = resultados[5].head(15).copy()
    df5['label'] = df5['categorias'].apply(acortar_nombre)

    fig, ax = plt.subplots(figsize=(18, 10))
    bars = ax.barh(df5['label'], df5['ventas'], color='#9B59B6', edgecolor='white', linewidth=0.5, height=0.7)
    for bar, pct in zip(bars, df5['porcentaje']):
        ax.text(bar.get_width() + total_ventas * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Cantidad de Ventas', fontsize=12)
    ax.set_title('ITEMSET 5 — Top 15 Quintetos de Categorías', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, df5['ventas'].max() * 1.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig('itemset_5_grafica.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  itemset_5_grafica.png")

    # ═══════════════════════════════════════════════════════════════
    # 8. GRÁFICA RESUMEN — Soporte por nivel
    # ═══════════════════════════════════════════════════════════════
    niveles = [1, 2, 3, 4, 5]
    soporte_prom = [resultados[n]['soporte'].mean() * 100 for n in niveles]
    soporte_max = [resultados[n]['soporte'].max() * 100 for n in niveles]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(niveles))
    width = 0.35
    bars1 = ax.bar(x - width/2, soporte_max, width, label='Soporte Máximo', color='#4D96FF', edgecolor='white')
    bars2 = ax.bar(x + width/2, soporte_prom, width, label='Soporte Promedio', color='#FFD93D', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Nivel de Itemset', fontsize=12)
    ax.set_ylabel('Soporte (%)', fontsize=12)
    ax.set_title('RESUMEN — Soporte por Nivel de Itemset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Itemset {n}' for n in niveles])
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('itemsets_resumen_grafica.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  itemsets_resumen_grafica.png")

    # ═══════════════════════════════════════════════════════════════
    # 9. GUARDAR CSVs
    # ═══════════════════════════════════════════════════════════════
    print(f"\nGuardando CSVs...")

    all_itemsets = []
    for nivel, df_itemset in resultados.items():
        filename = f'itemset_{nivel}.csv'
        df_export = df_itemset[['itemset', 'ventas', 'soporte', 'porcentaje']].copy()
        df_export.to_csv(filename, index=False)
        print(f"  {filename}")

        df_export.insert(0, 'nivel', nivel)
        all_itemsets.append(df_export)

    df_all = pd.concat(all_itemsets, ignore_index=True)
    df_all.to_csv('itemsets_completo.csv', index=False)
    print(f"  itemsets_completo.csv")

    # ═══════════════════════════════════════════════════════════════
    # 10. RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("RESUMEN FINAL")
    print(f"{'=' * 70}")
    print(f"\n  Ventas analizadas: {total_ventas:,}")

    for nivel, df_itemset in resultados.items():
        top = df_itemset.iloc[0]
        print(f"\n  Itemset {nivel}:")
        print(f"    Combinaciones: {len(df_itemset)}")
        print(f"    Top: {top['itemset']}")
        print(f"    Frecuencia: {top['ventas']:,} ({top['porcentaje']}%)")

    print(f"\n  Archivos generados:")
    print(f"    6 CSVs (itemset_1.csv a itemset_5.csv + itemsets_completo.csv)")
    print(f"    6 PNGs (itemset_1_grafica.png a itemset_5_grafica.png + resumen)")

    return resultados


if __name__ == "__main__":
    resultados = generar_itemsets_con_graficas()
