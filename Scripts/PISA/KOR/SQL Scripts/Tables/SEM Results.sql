DROP TABLE IF EXISTS pisa.pls_sem_indirect_effects CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_standardized_paths CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_r_squared CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_total_effects CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_reliability CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_fornell_larcker CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_htmt CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_indicator_loadings CASCADE;
DROP TABLE IF EXISTS pisa.pls_sem_latent_correlations CASCADE;


DROP TABLE IF EXISTS pisa.cb_sem_fit_measures CASCADE;
DROP TABLE IF EXISTS pisa.cb_sem_r_squared CASCADE;
DROP TABLE IF EXISTS pisa.cb_sem_latent_correlations CASCADE;



CREATE TABLE pisa.pls_sem_indirect_effects (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    iv               TEXT      NOT NULL,
    mediator         TEXT      NOT NULL,
    dv               TEXT      NOT NULL,

    indirect_b       NUMERIC,
    indirect_std     NUMERIC,
    bootstrap_mean   NUMERIC,
    bootstrap_sd     NUMERIC,
    t_stat           NUMERIC,
    p_value          NUMERIC,
    ci_95_lower      NUMERIC,
    ci_95_upper      NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_standardized_paths (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    dv               TEXT      NOT NULL,
    iv               TEXT      NOT NULL,
    path             TEXT      NOT NULL,   -- "SPI -> SMS"

    original_est     NUMERIC,
    bootstrap_sd     NUMERIC,

    std_b            NUMERIC,
    std_se           NUMERIC,
    std_t            NUMERIC,
    std_p            NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_r_squared (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    construct        TEXT      NOT NULL,
    r2               NUMERIC,
    adj_r2           NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_total_effects (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    iv               TEXT      NOT NULL,
    dv               TEXT      NOT NULL,

    total_effect     NUMERIC,
    bootstrap_mean   NUMERIC,
    bootstrap_sd     NUMERIC,
    t_stat           NUMERIC,
    ci_95_lower      NUMERIC,
    ci_95_upper      NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_reliability (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    construct        TEXT      NOT NULL,
    cronbach_alpha   NUMERIC,
    rho_c            NUMERIC,
    ave              NUMERIC,
    rho_a            NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_fornell_larcker (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    construct_row    TEXT      NOT NULL,
    construct_col    TEXT      NOT NULL,
    correlation      NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_htmt (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    construct_row    TEXT      NOT NULL,
    construct_col    TEXT      NOT NULL,
    htmt_value       NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_indicator_loadings (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    construct        TEXT      NOT NULL,
    indicator        TEXT      NOT NULL,

    loading          NUMERIC,
    indicator_r2     NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.pls_sem_latent_correlations (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    latent_1         TEXT      NOT NULL,
    latent_2         TEXT      NOT NULL,
    correlation      NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.cb_sem_fit_measures (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    metric           TEXT      NOT NULL,   -- CFI, TLI, RMSEA, SRMR, etc.
    value            NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.cb_sem_r_squared (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    latent           TEXT      NOT NULL,
    r2               NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);


CREATE TABLE pisa.cb_sem_latent_correlations (
    data_source      TEXT      NOT NULL,
    country_code     CHAR(3)   NOT NULL,

    latent_1         TEXT      NOT NULL,
    latent_2         TEXT      NOT NULL,
    correlation      NUMERIC,

    created_at       TIMESTAMP DEFAULT now()
);
