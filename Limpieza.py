import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg2://postgres:mi_password@localhost:5432/postgres')


def limpieza_canasta_basica():

    # ═══════════════════════════════════════════════════════════════
    # 1. CARGA DEL DATASET
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("LIMPIEZA DE DATASET - CANASTA BÁSICA")
    print("=" * 60)

    df = pd.read_csv('BigBasket Products.csv', low_memory=False)
    print(f"\nRegistros originales: {len(df):,}")
    print(f"Columnas: {list(df.columns)}")

    for col in ['product', 'category', 'sub_category', 'brand', 'type']:
        df[col] = df[col].astype(str).str.strip()

    # ═══════════════════════════════════════════════════════════════
    # 2. FILTRO POR CATEGORÍAS DE CANASTA BÁSICA
    # ═══════════════════════════════════════════════════════════════
    categorias_completas = [
        'Foodgrains, Oil & Masala',
        'Fruits & Vegetables',
        'Eggs, Meat & Fish',
        'Bakery, Cakes & Dairy',
        'Beverages',
        'Snacks & Branded Foods',
    ]

    higiene_esencial = [
        'Bath & Hand Wash',
        'Oral Care',
        'Feminine Hygiene',
        'Health & Medicine',
    ]

    limpieza_esencial = [
        'All Purpose Cleaners',
        'Detergents & Dishwash',
        'Disposables, Garbage Bag',
        'Fresheners & Repellents',
        'Mops, Brushes & Scrubs',
    ]

    baby_esencial = [
        'Diapers & Wipes',
        'Baby Food & Formula',
        'Baby Bath & Hygiene',
    ]

    gourmet_basico = [
        'Cooking & Baking Needs',
        'Sauces, Spreads & Dips',
        'Dairy & Cheese',
        'Pasta, Soup & Noodles',
        'Oils & Vinegar',
        'Cereals & Breakfast',
        'Tinned & Processed Food',
    ]

    mask_completas = df['category'].isin(categorias_completas)
    mask_higiene = (df['category'] == 'Beauty & Hygiene') & (df['sub_category'].isin(higiene_esencial))
    mask_limpieza = (df['category'] == 'Cleaning & Household') & (df['sub_category'].isin(limpieza_esencial))
    mask_baby = (df['category'] == 'Baby Care') & (df['sub_category'].isin(baby_esencial))
    mask_gourmet = (df['category'] == 'Gourmet & World Food') & (df['sub_category'].isin(gourmet_basico))

    df_canasta = df[mask_completas | mask_higiene | mask_limpieza | mask_baby | mask_gourmet].copy()
    print(f"\nDespués de filtro por categoría: {len(df_canasta):,}")

    # ═══════════════════════════════════════════════════════════════
    # 3. FILTRO POR PRECIO
    # ═══════════════════════════════════════════════════════════════
    df_canasta['sale_price'] = pd.to_numeric(df_canasta['sale_price'], errors='coerce')
    df_canasta['market_price'] = pd.to_numeric(df_canasta['market_price'], errors='coerce')

    df_canasta = df_canasta[
        (df_canasta['sale_price'] > 0) &
        (df_canasta['sale_price'] <= 2000)
    ]
    print(f"Después de filtro por precio (≤2000): {len(df_canasta):,}")

    # ═══════════════════════════════════════════════════════════════
    # 4. LIMPIEZA DE DATOS
    # ═══════════════════════════════════════════════════════════════
    df_canasta = df_canasta[df_canasta['product'].notna() & (df_canasta['product'] != 'nan')]
    df_canasta = df_canasta[df_canasta['brand'].notna() & (df_canasta['brand'] != 'nan')]
    df_canasta['rating'] = pd.to_numeric(df_canasta['rating'], errors='coerce').fillna(0)
    df_canasta = df_canasta.drop_duplicates(subset=['product', 'brand', 'sale_price'])
    print(f"Después de eliminar duplicados: {len(df_canasta):,}")

    # ═══════════════════════════════════════════════════════════════
    # 5. RECORTE FINAL A 10,000 ARTÍCULOS
    # ═══════════════════════════════════════════════════════════════
    if len(df_canasta) > 10000:
        total_antes = len(df_canasta)
        sampled = []
        for cat, group in df_canasta.groupby('category'):
            n_sample = max(1, int(10000 * len(group) / total_antes))
            n_sample = min(n_sample, len(group))
            sampled.append(group.sample(n=n_sample, random_state=42))
        df_canasta = pd.concat(sampled, ignore_index=True)
        if len(df_canasta) > 10000:
            df_canasta = df_canasta.sample(n=10000, random_state=42)
        print(f"Después de muestreo estratificado: {len(df_canasta):,}")

    # ═══════════════════════════════════════════════════════════════
    # 6. AGREGAR ID Y COLUMNAS FINALES
    # ═══════════════════════════════════════════════════════════════
    df_canasta = df_canasta.reset_index(drop=True)
    df_canasta.index += 1
    df_canasta.index.name = 'id_articulo'

    columnas_finales = [
        'product', 'category', 'sub_category', 'brand',
        'sale_price', 'market_price', 'type', 'rating',
    ]

    df_final = df_canasta[columnas_finales]

    # ═══════════════════════════════════════════════════════════════
    # 7. GUARDAR CSV LIMPIO
    # ═══════════════════════════════════════════════════════════════
    df_final.to_csv('canasta_basica_limpia.csv')

    print(f"\n{'=' * 60}")
    print(f"RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(f"Artículos limpios: {len(df_final):,}")
    print(f"Archivo guardado: canasta_basica_limpia.csv")

    print(f"\n--- Distribución por categoría ---")
    resumen = df_final.groupby('category').agg(
        productos=('product', 'count'),
        precio_min=('sale_price', 'min'),
        precio_max=('sale_price', 'max'),
        precio_prom=('sale_price', 'mean'),
        marcas=('brand', 'nunique')
    ).sort_values('productos', ascending=False)
    print(resumen.to_string())

    print(f"\n--- Resumen general ---")
    print(f"Categorías: {df_final['category'].nunique()}")
    print(f"Subcategorías: {df_final['sub_category'].nunique()}")
    print(f"Marcas únicas: {df_final['brand'].nunique()}")
    print(f"Rango de precios: {df_final['sale_price'].min():.2f} - {df_final['sale_price'].max():.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 8. CARGA A POSTGRESQL
    # ═══════════════════════════════════════════════════════════════
    print("\nCargando a PostgreSQL...")

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS transacciones CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS detalle_ventas CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS ventas CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS articulos CASCADE;"))
        conn.commit()

    # Tabla articulos
    df_final.to_sql('articulos', engine, if_exists='replace', index=True)
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE articulos ADD PRIMARY KEY (id_articulo);"))
        conn.commit()
    print("Tabla 'articulos' creada.")

    # Tabla ventas (1 registro = 1 ticket, resumen)
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE ventas (
                id_venta           SERIAL PRIMARY KEY,
                id_cliente         INT NOT NULL,
                fecha              DATE NOT NULL,
                hora               INT NOT NULL,
                cantidad_productos INT NOT NULL,
                total_venta        FLOAT NOT NULL
            );
        '''))
        conn.commit()
    print("Tabla 'ventas' creada.")

    # Tabla detalle_ventas (N registros por ticket, 1 por producto)
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE detalle_ventas (
                id_detalle   SERIAL PRIMARY KEY,
                id_venta     INT NOT NULL REFERENCES ventas(id_venta),
                id_articulo  INT NOT NULL REFERENCES articulos(id_articulo),
                cantidad     INT NOT NULL,
                precio_unit  FLOAT NOT NULL,
                subtotal     FLOAT NOT NULL
            );
        '''))
        conn.commit()
    print("Tabla 'detalle_ventas' creada.")

    # Tabla transacciones (1 registro = 1 ticket, movimiento financiero)
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE transacciones (
                id_transaccion     SERIAL PRIMARY KEY,
                id_venta           INT NOT NULL REFERENCES ventas(id_venta),
                id_cliente         INT NOT NULL,
                fecha              DATE NOT NULL,
                hora               INT NOT NULL,
                cantidad_productos INT NOT NULL,
                total_pagado       FLOAT NOT NULL
            );
        '''))
        conn.commit()
    print("Tabla 'transacciones' creada.")

    print("\n¡Todas las tablas creadas en PostgreSQL!")

    return df_final


if __name__ == "__main__":
    df = limpieza_canasta_basica()
