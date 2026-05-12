import duckdb

con = duckdb.connect("data/ecommerce.db")

checks = {
    "Orphan order_items (no parent order)": """
        SELECT COUNT(*) FROM order_items oi
        LEFT JOIN orders o USING (order_id)
        WHERE o.order_id IS NULL
    """,
    "Orphan orders (no customer)": """
        SELECT COUNT(*) FROM orders o
        LEFT JOIN customers c USING (customer_id)
        WHERE c.customer_id IS NULL
    """,
    "Orphan order_items (no product)": """
        SELECT COUNT(*) FROM order_items oi
        LEFT JOIN products p USING (product_id)
        WHERE p.product_id IS NULL
    """,
    "Orders with null order_id": """
        SELECT COUNT(*) FROM orders WHERE order_id IS NULL
    """,
    "Orders with null purchased_at": """
        SELECT COUNT(*) FROM orders_clean WHERE purchased_at IS NULL
    """,
    "Negative prices in order_items": """
        SELECT COUNT(*) FROM order_items_clean WHERE price < 0
    """,
    "Delivered before purchased (data error)": """
        SELECT COUNT(*) FROM orders_clean
        WHERE delivered_at IS NOT NULL
          AND delivered_at < purchased_at
    """,
}

print("\nData Quality Report — Day 1\n" + "="*45)
all_pass = True
for check, query in checks.items():
    result = con.execute(query).fetchone()[0]
    status = "PASS" if result == 0 else "WARN"
    if result != 0:
        all_pass = False
    print(f"  [{status}] {check}: {result:,}")

print("\n" + ("All checks passed." if all_pass else "Review warnings above — document in README."))
con.close()
