import duckdb
import os

DB_PATH = "data/ecommerce.db"
RAW     = "data/raw/"

TABLE_MAP = {
    "orders":      "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers":   "olist_customers_dataset.csv",
    "products":    "olist_products_dataset.csv",
    "sellers":     "olist_sellers_dataset.csv",
    "payments":    "olist_order_payments_dataset.csv",
    "reviews":     "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "categories":  "product_category_name_translation.csv",
}

def load_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    for table, fname in TABLE_MAP.items():
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv_auto('{path}', header=true)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  [OK] {table:<15} {n:>10,} rows")

def deduplicate_geolocation(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE geolocation_clean AS
        SELECT
            geolocation_zip_code_prefix AS zip,
            AVG(geolocation_lat)        AS lat,
            AVG(geolocation_lng)        AS lng,
            MAX(geolocation_city)       AS city,
            MAX(geolocation_state)      AS state
        FROM geolocation
        GROUP BY geolocation_zip_code_prefix
    """)
    n = con.execute("SELECT COUNT(*) FROM geolocation_clean").fetchone()[0]
    print(f"  [OK] geolocation_clean  {n:>10,} rows (deduped)")

def create_typed_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW orders_clean AS
        SELECT
            order_id,
            customer_id,
            order_status,
            CAST(order_purchase_timestamp       AS TIMESTAMP) AS purchased_at,
            CAST(order_approved_at              AS TIMESTAMP) AS approved_at,
            CAST(order_delivered_carrier_date   AS TIMESTAMP) AS dispatched_at,
            CAST(order_delivered_customer_date  AS TIMESTAMP) AS delivered_at,
            CAST(order_estimated_delivery_date  AS TIMESTAMP) AS estimated_at,
            DATE_DIFF('day',
                CAST(order_purchase_timestamp      AS TIMESTAMP),
                CAST(order_delivered_customer_date AS TIMESTAMP)
            ) AS actual_delivery_days,
            DATE_DIFF('day',
                CAST(order_delivered_customer_date AS TIMESTAMP),
                CAST(order_estimated_delivery_date AS TIMESTAMP)
            ) AS delivery_delta_days
        FROM orders
    """)
    print("  [OK] orders_clean view created")

    con.execute("""
        CREATE OR REPLACE VIEW order_items_clean AS
        SELECT
            order_id,
            order_item_id,
            product_id,
            seller_id,
            CAST(shipping_limit_date AS TIMESTAMP) AS shipping_limit_at,
            CAST(price         AS DOUBLE) AS price,
            CAST(freight_value AS DOUBLE) AS freight_value,
            CAST(price AS DOUBLE) + CAST(freight_value AS DOUBLE) AS total_item_value
        FROM order_items
    """)
    print("  [OK] order_items_clean view created")

    con.execute("""
        CREATE OR REPLACE VIEW products_clean AS
        SELECT
            p.product_id,
            COALESCE(c.product_category_name_english, 'unknown') AS category_en,
            CAST(p.product_weight_g        AS INTEGER) AS weight_g,
            CAST(p.product_length_cm       AS INTEGER) AS length_cm,
            CAST(p.product_height_cm       AS INTEGER) AS height_cm,
            CAST(p.product_width_cm        AS INTEGER) AS width_cm,
            CAST(p.product_photos_qty      AS INTEGER) AS photos_qty,
            CAST(p.product_description_lenght AS INTEGER) AS desc_length
        FROM products p
        LEFT JOIN categories c
            ON p.product_category_name = c.product_category_name
    """)
    print("  [OK] products_clean view created")

if __name__ == "__main__":
    print(f"\nConnecting to {DB_PATH}...")
    con = duckdb.connect(DB_PATH)

    print("\nLoading raw tables...")
    load_raw_tables(con)

    print("\nDeduplicating geolocation...")
    deduplicate_geolocation(con)

    print("\nCreating typed views...")
    create_typed_views(con)

    print(con.execute("""
        SELECT
            MIN(purchased_at) AS earliest_order,
            MAX(purchased_at) AS latest_order,
            COUNT(*)          AS total_orders
        FROM orders_clean
    """).df())

    print("\nDone.\n")
    con.close()
