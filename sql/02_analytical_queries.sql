-- Analytical queries against gold. Window functions are the point of this file.
-- Engine: DuckDB over gold parquet views.
-- Grain is month unless noted. Filter predicates go on fact date_key.

-- QUERY: mom_change_by_product
-- MoM complaint volume by product. LAG of a monthly grain avoids scanning
-- every fact row twice; the expensive part is the grouped scan of fact_complaints.
WITH monthly AS (
    SELECT
        d.year,
        d.month,
        p.product,
        COUNT(*) AS complaint_count
    FROM fact_complaints AS f
    JOIN dim_date AS d ON d.date_key = f.date_key
    JOIN dim_product AS p ON p.product_key = f.product_key
    GROUP BY d.year, d.month, p.product
)
SELECT
    year,
    month,
    product,
    complaint_count,
    LAG(complaint_count) OVER (
        PARTITION BY product
        ORDER BY year, month
    ) AS prev_month_count,
    ROUND(
        100.0 * (
            complaint_count - LAG(complaint_count) OVER (
                PARTITION BY product
                ORDER BY year, month
            )
        ) / NULLIF(
            LAG(complaint_count) OVER (
                PARTITION BY product
                ORDER BY year, month
            ),
            0
        ),
        1
    ) AS mom_pct_change
FROM monthly
ORDER BY product, year, month;


-- QUERY: company_rank_within_product_quarter
-- RANK companies inside each product-quarter. Narrow the fact scan first.
WITH quarterly AS (
    SELECT
        d.year,
        d.quarter,
        p.product,
        c.company_name,
        COUNT(*) AS complaint_count,
        AVG(CAST(f.timely_response AS INTEGER)) AS timely_rate
    FROM fact_complaints AS f
    JOIN dim_date AS d ON d.date_key = f.date_key
    JOIN dim_product AS p ON p.product_key = f.product_key
    JOIN dim_company AS c ON c.company_key = f.company_key
    GROUP BY d.year, d.quarter, p.product, c.company_name
)
SELECT
    year,
    quarter,
    product,
    company_name,
    complaint_count,
    ROUND(timely_rate, 3) AS timely_rate,
    RANK() OVER (
        PARTITION BY year, quarter, product
        ORDER BY complaint_count DESC
    ) AS volume_rank,
    ROUND(
        complaint_count * 1.0 / SUM(complaint_count) OVER (
            PARTITION BY year, quarter, product
        ),
        3
    ) AS product_share
FROM quarterly
QUALIFY volume_rank <= 5
ORDER BY year, quarter, product, volume_rank;


-- QUERY: rolling_timely_rate_by_company
-- Running timely-response rate by company over a 3-month window.
WITH monthly AS (
    SELECT
        c.company_name,
        d.year,
        d.month,
        COUNT(*) AS complaints,
        SUM(CAST(f.timely_response AS INTEGER)) AS timely_yes
    FROM fact_complaints AS f
    JOIN dim_company AS c ON c.company_key = f.company_key
    JOIN dim_date AS d ON d.date_key = f.date_key
    GROUP BY c.company_name, d.year, d.month
)
SELECT
    company_name,
    year,
    month,
    complaints,
    ROUND(
        SUM(timely_yes) OVER w * 1.0 / NULLIF(SUM(complaints) OVER w, 0),
        3
    ) AS timely_rate_3mo
FROM monthly
WINDOW w AS (
    PARTITION BY company_name
    ORDER BY year, month
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
)
ORDER BY company_name, year, month;


-- QUERY: issue_share_within_product
-- Which issues dominate each product? SUM() OVER replaces a self-join.
SELECT
    p.product,
    i.issue,
    COUNT(*) AS complaint_count,
    ROUND(
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY p.product),
        3
    ) AS issue_share,
    DENSE_RANK() OVER (
        PARTITION BY p.product
        ORDER BY COUNT(*) DESC
    ) AS issue_rank
FROM fact_complaints AS f
JOIN dim_product AS p ON p.product_key = f.product_key
JOIN dim_issue AS i ON i.issue_key = f.issue_key
GROUP BY p.product, i.issue
QUALIFY issue_rank <= 5
ORDER BY p.product, issue_rank;


-- QUERY: state_volume_wow
-- Week-over-week volume by state. LAG detects spikes without a correlated subquery.
WITH weekly AS (
    SELECT
        g.state,
        d.year,
        d.week,
        COUNT(*) AS complaint_count
    FROM fact_complaints AS f
    JOIN dim_geo AS g ON g.geo_key = f.geo_key
    JOIN dim_date AS d ON d.date_key = f.date_key
    WHERE g.state IS NOT NULL AND g.state <> ''
    GROUP BY g.state, d.year, d.week
)
SELECT
    state,
    year,
    week,
    complaint_count,
    LAG(complaint_count) OVER (
        PARTITION BY state
        ORDER BY year, week
    ) AS prev_week_count,
    complaint_count - LAG(complaint_count) OVER (
        PARTITION BY state
        ORDER BY year, week
    ) AS wow_delta
FROM weekly
ORDER BY wow_delta DESC NULLS LAST
LIMIT 25;


-- QUERY: narrative_coverage_by_channel
-- RAG prerequisite: where do long-form narratives actually exist?
SELECT
    ch.channel_name,
    COUNT(*) AS complaints,
    SUM(CAST(f.has_narrative AS INTEGER)) AS with_narrative,
    ROUND(
        AVG(CAST(f.has_narrative AS INTEGER)),
        3
    ) AS narrative_rate,
    ROUND(AVG(f.days_to_company), 2) AS avg_days_to_company
FROM fact_complaints AS f
JOIN dim_channel AS ch ON ch.channel_key = f.channel_key
GROUP BY ch.channel_name
ORDER BY complaints DESC;
