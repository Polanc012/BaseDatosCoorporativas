"""
EJERCICIO 1 — REGLAS DE ASOCIACIÓN (Soporte, Confianza, Lift)
==============================================================
Adaptado del ejercicio de libreta con Leche, Huevo, Tortillas...
Ahora usando las 400K ventas reales y las 10 categorías de canasta básica.

Problema:
    Dadas las ventas en la BD, calcular soporte, confianza y lift
    para reglas de asociación entre categorías.
    Ejemplo: Si un cliente compra Fruits & Vegetables (X),
    ¿qué tan probable es que también compre Bakery, Cakes & Dairy (Y)?
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import combinations
from sqlalchemy import create_engine

matplotlib.use('Agg')

engine = create_engine('postgresql+psycopg2://postgres:mi_password@localhost:5432/postgres')

# Nombres cortos para gráficas
NOMBRES_CORTOS = {
    'Baby Care': 'Baby', 'Bakery, Cakes & Dairy': 'Bakery',
    'Beauty & Hygiene': 'Beauty', 'Beverages': 'Beverages',
    'Cleaning & Household': 'Cleaning', 'Eggs, Meat & Fish': 'Eggs/Meat',
    'Foodgrains, Oil & Masala': 'Foodgrains', 'Fruits & Vegetables': 'Fruits/Veg',
    'Gourmet & World Food': 'Gourmet', 'Snacks & Branded Foods': 'Snacks',
}


def reglas_de_asociacion():

    print("=" * 70)
    print("EJERCICIO 1 — REGLAS DE ASOCIACIÓN")
    print("Soporte, Confianza y Lift entre Categorías")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # 1. OBTENER DATOS DE LA BD
    # ═══════════════════════════════════════════════════════════════
    print("\nConsultando base de datos...")

    query = """
        SELECT d.id_venta, a.category
        FROM detalle_ventas d
        JOIN articulos a ON d.id_articulo = a.id_articulo
    """
    df = pd.read_sql(query, engine)

    # Categorías únicas por venta
    ventas_cats = df.groupby('id_venta')['category'].apply(
        lambda x: frozenset(x.unique())
    ).reset_index()
    ventas_cats.columns = ['id_venta', 'categorias']

    total_ventas = len(ventas_cats)
    todas_cats = sorted(df['category'].unique())

    print(f"Total ventas: {total_ventas:,}")
    print(f"Categorías: {len(todas_cats)}")

    # ═══════════════════════════════════════════════════════════════
    # 2. CALCULAR SOPORTE INDIVIDUAL (como en la libreta)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("PASO 1: SOPORTE INDIVIDUAL — ¿En cuántas ventas aparece cada categoría?")
    print(f"{'─' * 70}")

    soporte_individual = {}
    for cat in todas_cats:
        count = ventas_cats['categorias'].apply(lambda x: cat in x).sum()
        soporte = count / total_ventas
        soporte_individual[cat] = {'count': count, 'soporte': soporte}
        print(f"  {cat:<35} aparece en {count:>7,} ventas  →  Soporte = {soporte:.4f} ({soporte*100:.2f}%)")

    # ═══════════════════════════════════════════════════════════════
    # 3. CALCULAR REGLAS DE ASOCIACIÓN PARA TODOS LOS PARES
    #    (Igual que en la libreta: X→Y con Soporte, Confianza, Lift)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("PASO 2: REGLAS DE ASOCIACIÓN X → Y")
    print("  Soporte(X∪Y) = ventas con X y Y / total ventas")
    print("  Confianza    = Soporte(X∪Y) / Soporte(X)")
    print("  Lift         = Confianza / Soporte(Y)")
    print(f"{'─' * 70}")

    reglas = []

    for x, y in combinations(todas_cats, 2):
        # Contar ventas donde aparecen AMBAS categorías (X ∪ Y)
        count_xy = ventas_cats['categorias'].apply(
            lambda cats: x in cats and y in cats
        ).sum()

        soporte_xy = count_xy / total_ventas
        soporte_x = soporte_individual[x]['soporte']
        soporte_y = soporte_individual[y]['soporte']

        # Regla X → Y
        confianza_xy = soporte_xy / soporte_x if soporte_x > 0 else 0
        lift_xy = confianza_xy / soporte_y if soporte_y > 0 else 0

        reglas.append({
            'antecedente': x,
            'consecuente': y,
            'ant_short': NOMBRES_CORTOS.get(x, x[:10]),
            'cons_short': NOMBRES_CORTOS.get(y, y[:10]),
            'soporte_x': round(soporte_x, 4),
            'soporte_y': round(soporte_y, 4),
            'soporte_xy': round(soporte_xy, 4),
            'confianza': round(confianza_xy, 4),
            'lift': round(lift_xy, 4),
        })

        # Regla Y → X (la dirección importa para confianza)
        confianza_yx = soporte_xy / soporte_y if soporte_y > 0 else 0
        lift_yx = confianza_yx / soporte_x if soporte_x > 0 else 0

        reglas.append({
            'antecedente': y,
            'consecuente': x,
            'ant_short': NOMBRES_CORTOS.get(y, y[:10]),
            'cons_short': NOMBRES_CORTOS.get(x, x[:10]),
            'soporte_x': round(soporte_y, 4),
            'soporte_y': round(soporte_x, 4),
            'soporte_xy': round(soporte_xy, 4),
            'confianza': round(confianza_yx, 4),
            'lift': round(lift_yx, 4),
        })

    df_reglas = pd.DataFrame(reglas).sort_values('lift', ascending=False)

    # ═══════════════════════════════════════════════════════════════
    # 4. EJEMPLO DETALLADO (como en la libreta)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("EJEMPLO DETALLADO — Paso a paso (como en la libreta)")
    print(f"{'─' * 70}")

    # Tomar la regla con mayor lift
    top = df_reglas.iloc[0]
    x_name = top['antecedente']
    y_name = top['consecuente']

    count_x = soporte_individual[x_name]['count']
    count_y = soporte_individual[y_name]['count']
    count_xy = int(top['soporte_xy'] * total_ventas)

    print(f"\n  Regla: {x_name} → {y_name}")
    print(f"\n  Datos:")
    print(f"    Antecedente (X) = {x_name}")
    print(f"      → Aparece en {count_x:,} de {total_ventas:,} ventas")
    print(f"    Consecuente (Y) = {y_name}")
    print(f"      → Aparece en {count_y:,} de {total_ventas:,} ventas")
    print(f"    Unión (X ∪ Y)")
    print(f"      → Ambos aparecen en {count_xy:,} ventas")

    print(f"\n  Cálculos:")
    print(f"    Soporte(X∪Y) = {count_xy:,} / {total_ventas:,} = {top['soporte_xy']:.4f} ({top['soporte_xy']*100:.2f}%)")
    print(f"    Confianza    = Soporte(X∪Y) / Soporte(X) = {top['soporte_xy']:.4f} / {top['soporte_x']:.4f} = {top['confianza']:.4f} ({top['confianza']*100:.2f}%)")
    print(f"    Lift         = Confianza / Soporte(Y) = {top['confianza']:.4f} / {top['soporte_y']:.4f} = {top['lift']:.4f}")

    print(f"\n  Interpretación:")
    if top['lift'] > 1:
        print(f"    Lift = {top['lift']:.4f} > 1 → Las categorías se compran juntas MÁS de lo esperado.")
        print(f"    Hay una asociación POSITIVA entre {x_name} y {y_name}.")
    elif top['lift'] == 1:
        print(f"    Lift = {top['lift']:.4f} = 1 → Las categorías son INDEPENDIENTES.")
    else:
        print(f"    Lift = {top['lift']:.4f} < 1 → Las categorías se compran juntas MENOS de lo esperado.")

    # ═══════════════════════════════════════════════════════════════
    # 5. TOP 20 REGLAS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("TOP 20 REGLAS POR LIFT")
    print(f"{'─' * 70}")
    print(f"  {'Regla':<50} {'Soporte':>10} {'Confianza':>12} {'Lift':>8}")
    print(f"  {'─'*50} {'─'*10} {'─'*12} {'─'*8}")

    for _, row in df_reglas.head(20).iterrows():
        regla_str = f"{row['ant_short']} → {row['cons_short']}"
        print(f"  {regla_str:<50} {row['soporte_xy']:>10.4f} {row['confianza']:>12.4f} {row['lift']:>8.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 6. GRÁFICA 1 — Top 20 reglas por Lift
    # ═══════════════════════════════════════════════════════════════
    print("\nGenerando gráficas...")

    top20 = df_reglas.head(20).copy()
    top20['regla'] = top20['ant_short'] + ' → ' + top20['cons_short']

    fig, ax = plt.subplots(figsize=(14, 9))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top20)))
    bars = ax.barh(top20['regla'], top20['lift'], color=colors, edgecolor='white', linewidth=0.5)
    for bar, conf in zip(bars, top20['confianza']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'Conf: {conf:.2%}', va='center', fontsize=8, fontweight='bold')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Lift = 1 (independencia)')
    ax.set_xlabel('Lift', fontsize=12)
    ax.set_title('REGLAS DE ASOCIACIÓN — Top 20 por Lift', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('ejercicio1_reglas_lift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio1_reglas_lift.png")

    # ═══════════════════════════════════════════════════════════════
    # 7. GRÁFICA 2 — Heatmap de Confianza
    # ═══════════════════════════════════════════════════════════════
    cats_short = [NOMBRES_CORTOS.get(c, c[:10]) for c in todas_cats]
    matrix_conf = np.zeros((len(todas_cats), len(todas_cats)))

    for _, row in df_reglas.iterrows():
        i = cats_short.index(row['ant_short'])
        j = cats_short.index(row['cons_short'])
        matrix_conf[i][j] = row['confianza']

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix_conf, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(cats_short)))
    ax.set_yticks(range(len(cats_short)))
    ax.set_xticklabels(cats_short, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cats_short, fontsize=9)

    for i in range(len(cats_short)):
        for j in range(len(cats_short)):
            if matrix_conf[i][j] > 0:
                ax.text(j, i, f'{matrix_conf[i][j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if matrix_conf[i][j] > 0.5 else 'black')

    ax.set_xlabel('Consecuente (Y)', fontsize=12)
    ax.set_ylabel('Antecedente (X)', fontsize=12)
    ax.set_title('HEATMAP DE CONFIANZA — Regla: X → Y', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Confianza')
    plt.tight_layout()
    plt.savefig('ejercicio1_heatmap_confianza.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio1_heatmap_confianza.png")

    # ═══════════════════════════════════════════════════════════════
    # 8. GRÁFICA 3 — Scatter: Soporte vs Confianza (tamaño = Lift)
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df_reglas['soporte_xy'], df_reglas['confianza'],
        s=df_reglas['lift'] * 300, alpha=0.6,
        c=df_reglas['lift'], cmap='coolwarm', edgecolors='gray', linewidth=0.5
    )
    for _, row in df_reglas.head(10).iterrows():
        ax.annotate(f"{row['ant_short']}→{row['cons_short']}",
                    (row['soporte_xy'], row['confianza']),
                    fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('Soporte (X∪Y)', fontsize=12)
    ax.set_ylabel('Confianza', fontsize=12)
    ax.set_title('SOPORTE vs CONFIANZA — Tamaño = Lift', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(scatter, ax=ax, label='Lift')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('ejercicio1_scatter_soporte_confianza.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ejercicio1_scatter_soporte_confianza.png")

    # Guardar CSV
    df_reglas.to_csv('ejercicio1_reglas_asociacion.csv', index=False)
    print("  ejercicio1_reglas_asociacion.csv")

    print(f"\n{'=' * 70}")
    print("EJERCICIO 1 COMPLETADO")
    print(f"{'=' * 70}")

    return df_reglas


if __name__ == "__main__":
    reglas = reglas_de_asociacion()
