-- Gold star schema for CFPB consumer complaints.
-- Engine: DuckDB over parquet (views registered by etl/run_queries.py).

CREATE TABLE IF NOT EXISTS dim_date (
    date_key INT NOT NULL,
    date TIMESTAMP NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    month INT NOT NULL,
    month_name STRING,
    week INT,
    day_of_week STRING,
    is_weekend BOOLEAN,
    CONSTRAINT pk_dim_date PRIMARY KEY (date_key)
);

CREATE TABLE IF NOT EXISTS dim_company (
    company_key STRING NOT NULL,
    company_name STRING NOT NULL,
    CONSTRAINT pk_dim_company PRIMARY KEY (company_key)
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_key STRING NOT NULL,
    product STRING NOT NULL,
    sub_product STRING,
    CONSTRAINT pk_dim_product PRIMARY KEY (product_key)
);

CREATE TABLE IF NOT EXISTS dim_issue (
    issue_key STRING NOT NULL,
    issue STRING,
    sub_issue STRING,
    CONSTRAINT pk_dim_issue PRIMARY KEY (issue_key)
);

CREATE TABLE IF NOT EXISTS dim_geo (
    geo_key STRING NOT NULL,
    state STRING,
    zip_code STRING,
    CONSTRAINT pk_dim_geo PRIMARY KEY (geo_key)
);

CREATE TABLE IF NOT EXISTS dim_channel (
    channel_key STRING NOT NULL,
    channel_name STRING NOT NULL,
    CONSTRAINT pk_dim_channel PRIMARY KEY (channel_key)
);

CREATE TABLE IF NOT EXISTS fact_complaints (
    complaint_id STRING NOT NULL,
    company_key STRING NOT NULL,
    product_key STRING NOT NULL,
    issue_key STRING NOT NULL,
    geo_key STRING,
    channel_key STRING,
    date_key INT NOT NULL,
    sent_date_key INT,
    date_received TIMESTAMP NOT NULL,
    date_sent_to_company TIMESTAMP,
    days_to_company DOUBLE,
    timely_response BOOLEAN,
    consumer_disputed BOOLEAN,
    has_narrative BOOLEAN,
    company_response STRING,
    consumer_consent STRING,
    CONSTRAINT pk_fact_complaints PRIMARY KEY (complaint_id)
);
