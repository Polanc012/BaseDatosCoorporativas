import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg2://postgres:mi_password@localhost:5432/postgres')


def generar_transacciones():
    """
    Genera 400,000 transacciones.
    - ventas:          1 registro por ticket (resumen)
    - detalle_ventas:  10-25 registros por ticket (cada producto)
    - transacciones:   1 registro por ticket (movimiento financiero)

    ventas y transacciones son 1:1 (400,000 cada una)
    detalle_ventas tiene millones (10-25 productos por venta)
    """

    print("=" * 60)
    print("GENERADOR DE 400,000 TRANSACCIONES")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 1. LEER ARTÍCULOS
    # ═══════════════════════════════════════════════════════════════
    df_art = pd.read_sql("SELECT * FROM articulos", engine)
    print(f"\nArtículos disponibles: {len(df_art):,}")

    ids = df_art['id_articulo'].values
    precios = df_art['sale_price'].values
    categorias = df_art['category'].values
    categorias_unicas = df_art['category'].unique()

    indices_por_cat = {}
    for cat in categorias_unicas:
        indices_por_cat[cat] = np.where(categorias == cat)[0]

    # ═══════════════════════════════════════════════════════════════
    # 2. CONFIGURACIÓN
    # ═══════════════════════════════════════════════════════════════
    TOTAL_VENTAS = 400_000        # 400K ventas = 400K transacciones
    NUM_CLIENTES = 50_000
    ITEMS_MIN = 10
    ITEMS_MAX = 25
    ITEMS_PROMEDIO = 17
    MIN_CATEGORIAS = 5

    np.random.seed(42)

    peso_categoria = {
        'Fruits & Vegetables':      5.0,
        'Bakery, Cakes & Dairy':    4.5,
        'Eggs, Meat & Fish':        4.0,
        'Foodgrains, Oil & Masala': 3.5,
        'Beverages':                3.0,
        'Snacks & Branded Foods':   2.5,
        'Cleaning & Household':     2.0,
        'Beauty & Hygiene':         1.5,
        'Gourmet & World Food':     1.0,
        'Baby Care':                0.8,
    }

    pesos_cat = np.array([peso_categoria.get(cat, 1.0) for cat in categorias_unicas])
    pesos_cat = pesos_cat / pesos_cat.sum()

    horas_disponibles = np.arange(7, 23)
    peso_horas = np.array([
        2, 4, 6, 8, 8, 7,
        5, 4, 4, 5, 7, 8,
        7, 5, 3, 1
    ], dtype=float)
    peso_horas = peso_horas / peso_horas.sum()

    fechas = pd.date_range('2024-01-01', '2024-12-31')

    # ═══════════════════════════════════════════════════════════════
    # 3. GENERAR LAS 400K VENTAS
    # ═══════════════════════════════════════════════════════════════
    print(f"\nGenerando {TOTAL_VENTAS:,} ventas con 10-25 productos cada una...")

    ventas_data = []
    detalle_data = []
    transacciones_data = []

    total_detalles = 0

    for id_venta in range(1, TOTAL_VENTAS + 1):

        # Datos del ticket
        cliente = np.random.randint(1, NUM_CLIENTES + 1)
        fecha = np.random.choice(fechas)
        hora = int(np.random.choice(horas_disponibles, p=peso_horas))
        n_items = int(np.clip(np.random.normal(ITEMS_PROMEDIO, 3), ITEMS_MIN, ITEMS_MAX))

        # --- Selección diversa de artículos (mín 5 categorías) ---
        n_cats = min(np.random.randint(MIN_CATEGORIAS, len(categorias_unicas) + 1), n_items)
        cats_elegidas = np.random.choice(
            len(categorias_unicas), size=n_cats, replace=False, p=pesos_cat
        )

        items_por_cat = np.ones(n_cats, dtype=int)
        items_restantes = n_items - n_cats
        if items_restantes > 0:
            pesos_elegidas = pesos_cat[cats_elegidas]
            pesos_elegidas = pesos_elegidas / pesos_elegidas.sum()
            extra = np.random.multinomial(items_restantes, pesos_elegidas)
            items_por_cat += extra

        articulos_elegidos = []
        for cat_idx, n_from_cat in zip(cats_elegidas, items_por_cat):
            cat_name = categorias_unicas[cat_idx]
            pool = indices_por_cat[cat_name]
            n_pick = min(n_from_cat, len(pool))
            picks = np.random.choice(pool, size=n_pick, replace=False)
            articulos_elegidos.extend(picks)

        # --- Generar detalle de esta venta ---
        total_venta = 0.0

        for art_idx in articulos_elegidos:
            cantidad = np.random.choice([1, 1, 1, 1, 2, 2, 3])
            precio_unit = precios[art_idx]
            descuento = np.random.uniform(0, 0.10)
            precio_final = round(precio_unit * (1 - descuento), 2)
            subtotal = round(precio_final * cantidad, 2)
            total_venta += subtotal

            # detalle_ventas: 1 registro por producto en el ticket
            detalle_data.append((
                id_venta,
                int(ids[art_idx]),
                cantidad,
                precio_final,
                subtotal,
            ))

        total_venta = round(total_venta, 2)

        # ventas: 1 registro por ticket (resumen)
        ventas_data.append((
            cliente,
            fecha,
            hora,
            len(articulos_elegidos),
            total_venta,
        ))

        # transacciones: 1 registro por ticket (movimiento financiero)
        transacciones_data.append((
            id_venta,
            cliente,
            fecha,
            hora,
            len(articulos_elegidos),
            total_venta,
        ))

        total_detalles += len(articulos_elegidos)

        # Progreso cada 50K
        if id_venta % 50000 == 0:
            print(f"  {id_venta:,} ventas generadas ({total_detalles:,} detalles)...")

    # ═══════════════════════════════════════════════════════════════
    # 4. CREAR DATAFRAMES
    # ═══════════════════════════════════════════════════════════════
    print("\nCreando DataFrames...")

    df_ventas = pd.DataFrame(ventas_data, columns=[
        'id_cliente', 'fecha', 'hora', 'cantidad_productos', 'total_venta'
    ])

    df_detalle = pd.DataFrame(detalle_data, columns=[
        'id_venta', 'id_articulo', 'cantidad', 'precio_unit', 'subtotal'
    ])

    df_transacciones = pd.DataFrame(transacciones_data, columns=[
        'id_venta', 'id_cliente', 'fecha', 'hora', 'cantidad_productos', 'total_pagado'
    ])

    print(f"  ventas:          {len(df_ventas):>10,} registros  (1 por ticket)")
    print(f"  detalle_ventas:  {len(df_detalle):>10,} registros  (productos por ticket)")
    print(f"  transacciones:   {len(df_transacciones):>10,} registros  (1 por ticket)")
    print(f"  promedio items/ticket: {len(df_detalle) / len(df_ventas):.1f}")

    # ═══════════════════════════════════════════════════════════════
    # 5. CARGAR A POSTGRESQL
    # ═══════════════════════════════════════════════════════════════
    print("\nCargando a PostgreSQL...")

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE transacciones RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE detalle_ventas RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE ventas RESTART IDENTITY CASCADE;"))
        conn.commit()

    print("  Cargando ventas (400K)...")
    df_ventas.to_sql('ventas', engine, if_exists='append', index=False,
                     chunksize=10000, method='multi')

    print("  Cargando detalle_ventas (~6.8M)...")
    df_detalle.to_sql('detalle_ventas', engine, if_exists='append', index=False,
                      chunksize=10000, method='multi')

    print("  Cargando transacciones (400K)...")
    df_transacciones.to_sql('transacciones', engine, if_exists='append', index=False,
                            chunksize=10000, method='multi')

    print("\n¡Carga completada!")

    # ═══════════════════════════════════════════════════════════════
    # 6. VERIFICACIÓN
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("VERIFICACIÓN EN BASE DE DATOS")
    print(f"{'=' * 60}")

    with engine.connect() as conn:
        v = conn.execute(text("SELECT COUNT(*) FROM ventas")).scalar()
        d = conn.execute(text("SELECT COUNT(*) FROM detalle_ventas")).scalar()
        t = conn.execute(text("SELECT COUNT(*) FROM transacciones")).scalar()
        c = conn.execute(text("SELECT COUNT(DISTINCT id_cliente) FROM ventas")).scalar()

        print(f"\n  ventas:          {v:>10,}  (tickets)")
        print(f"  detalle_ventas:  {d:>10,}  (líneas de producto)")
        print(f"  transacciones:   {t:>10,}  (movimientos)")
        print(f"  clientes únicos: {c:>10,}")

        # Top 10 productos
        print(f"\n--- Top 10 productos más vendidos ---")
        top = conn.execute(text("""
            SELECT a.product, a.category, COUNT(*) as veces, SUM(d.subtotal) as ingreso
            FROM detalle_ventas d
            JOIN articulos a ON d.id_articulo = a.id_articulo
            GROUP BY a.product, a.category
            ORDER BY veces DESC
            LIMIT 10
        """))
        for row in top:
            print(f"  {row[0][:40]:<40} | {row[2]:>6} ventas | ${row[3]:>10,.2f}")

        # Categorías por ticket (para validar itemsets)
        print(f"\n--- Categorías por ticket (para itemsets) ---")
        cats_stats = conn.execute(text("""
            SELECT
                MIN(cats) as min_cats,
                MAX(cats) as max_cats,
                ROUND(AVG(cats), 1) as avg_cats,
                COUNT(*) FILTER (WHERE cats >= 5) as tickets_5_plus
            FROM (
                SELECT d.id_venta, COUNT(DISTINCT a.category) as cats
                FROM detalle_ventas d
                JOIN articulos a ON d.id_articulo = a.id_articulo
                GROUP BY d.id_venta
            ) sub
        """)).fetchone()
        print(f"  Mínimo:  {cats_stats[0]} categorías")
        print(f"  Máximo:  {cats_stats[1]} categorías")
        print(f"  Promedio: {cats_stats[2]} categorías")
        print(f"  Tickets con 5+ categorías: {cats_stats[3]:,}")

        # Ingreso total
        ingreso = conn.execute(text("SELECT SUM(total_venta) FROM ventas")).scalar()
        print(f"\nIngreso total: ${ingreso:,.2f}")


if __name__ == "__main__":
    generar_transacciones()
