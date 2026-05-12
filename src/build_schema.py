import duckdb

con = duckdb.connect("data/ecommerce.db")

con.execute("""
    CREATE OR REPLACE TABLE dim_customer AS
    SELECT
        customer_id,
        customer_unique_id,
        customer_zip_code_prefix  AS zip,
        customer_city             AS city,
        customer_state            AS state
    FROM customers
""")
print("dim_customer:", con.execute("SELECT COUNT(*) FROM dim_customer").fetchone()[0])

con.execute("""
    CREATE OR REPLACE TABLE dim_product AS
    SELECT
        p.product_id,
        COALESCE(c.product_category_name_english, 'unknown') AS category_en,
        CAST(p.product_weight_g           AS INTEGER) AS weight_g,
        CAST(p.product_length_cm          AS INTEGER) AS length_cm,
        CAST(p.product_height_cm          AS INTEGER) AS height_cm,
        CAST(p.product_width_cm           AS INTEGER) AS width_cm,
        CAST(p.product_photos_qty         AS INTEGER) AS photos_qty,
        CAST(p.product_description_lenght AS INTEGER) AS desc_length,
        (CAST(p.product_length_cm AS INTEGER) *
         CAST(p.product_height_cm AS INTEGER) *
         CAST(p.product_width_cm  AS INTEGER)) AS volume_cm3
    FROM products p
    LEFT JOIN categories c
        ON p.product_category_name = c.product_category_name
""")
print("dim_product:", con.execute("SELECT COUNT(*) FROM dim_product").fetchone()[0])

con.execute("""
    CREATE OR REPLACE TABLE dim_seller AS
    SELECT
        seller_id,
        seller_zip_code_prefix AS zip,
        seller_city            AS city,
        seller_state           AS state
    FROM sellers
""")
print("dim_seller:", con.execute("SELECT COUNT(*) FROM dim_seller").fetchone()[0])

con.execute("""
    CREATE OR REPLACE TABLE dim_date AS
    WITH date_spine AS (
        SELECT
            CAST(range AS DATE) AS date
        FROM range(
            (SELECT MIN(purchased_at::DATE) FROM orders_clean),
            (SELECT MAX(purchased_at::DATE) + INTERVAL '1 day' FROM orders_clean),
            INTERVAL '1 day'
        )
    )
    SELECT
        date,
        EXTRACT(year    FROM date)::INTEGER  AS year,
        EXTRACT(month   FROM date)::INTEGER  AS month,
        EXTRACT(quarter FROM date)::INTEGER  AS quarter,
        EXTRACT(week    FROM date)::INTEGER  AS week_of_year,
        EXTRACT(dow     FROM date)::INTEGER  AS day_of_week,   -- 0=Sun, 6=Sat
        STRFTIME(date, '%B')                 AS month_name,
        STRFTIME(date, '%A')                 AS day_name,
        CASE WHEN EXTRACT(dow FROM date) IN (0, 6)
             THEN true ELSE false END        AS is_weekend,
        CASE WHEN EXTRACT(month FROM date) IN (11, 12, 1)
             THEN true ELSE false END        AS is_peak_season
    FROM date_spine
""")
print("dim_date:", con.execute("SELECT COUNT(*) FROM dim_date").fetchone()[0])

con.execute("""
    CREATE OR REPLACE TABLE orders_fact AS
    SELECT
        o.order_id,
        o.customer_id,
        o.order_status,
        o.purchased_at::DATE                AS order_date,
        o.delivered_at::DATE                AS delivered_date,
        o.estimated_at::DATE                AS estimated_date,
        o.actual_delivery_days,
        o.delivery_delta_days,

        -- aggregated item-level measures (one row per order)
        COUNT(oi.order_item_id)             AS item_count,
        SUM(oi.price)                       AS revenue,
        SUM(oi.freight_value)               AS freight_revenue,
        SUM(oi.total_item_value)            AS gmv,
        AVG(oi.price)                       AS avg_item_price,

        -- review
        r.review_score,

        -- payment
        p.payment_type,
        p.payment_installments

    FROM orders_clean o
    LEFT JOIN order_items_clean oi USING (order_id)
    LEFT JOIN reviews r            USING (order_id)
    LEFT JOIN payments p           USING (order_id)
    GROUP BY ALL
""")
print("orders_fact:", con.execute("SELECT COUNT(*) FROM orders_fact").fetchone()[0])

con.execute("""
    CREATE OR REPLACE TABLE bridge_order_items AS
    SELECT
        oi.order_id,
        oi.order_item_id,
        oi.product_id,
        oi.seller_id,
        o.customer_id,
        o.purchased_at::DATE  AS order_date,
        oi.price,
        oi.freight_value,
        oi.total_item_value
    FROM order_items_clean oi
    JOIN orders_clean o USING (order_id)
""")
print("bridge_order_items:", con.execute("SELECT COUNT(*) FROM bridge_order_items").fetchone()[0])

