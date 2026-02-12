CREATE OR REPLACE VIEW pisa.real_sem_SGP AS
WITH stats AS (
    SELECT
        -- Continuous: medians
        percentile_cont(0.5) WITHIN GROUP (ORDER BY s.AGE)      AS med_age,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY s.ESCS)     AS med_escs,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY sch.MCLSIZE) AS med_mclsize,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY sch.SCHSIZE) AS med_schsize,

        -- Categorical / ordinal: modes
        mode() WITHIN GROUP (ORDER BY s.ST001D01T) AS mode_grade,
        mode() WITHIN GROUP (ORDER BY s.ST004D01T) AS mode_gender,
        mode() WITHIN GROUP (ORDER BY s.MISCED)    AS mode_misced,
        mode() WITHIN GROUP (ORDER BY s.IMMIG)     AS mode_immig
    FROM pisa.students s
    JOIN pisa.schools sch
      ON s.CNT = sch.CNT
     AND s.CNTSCHID = sch.CNTSCHID
    WHERE s.CNT = 'SGP'
)

SELECT
    s.CNT,
    s.CNTSCHID,
    s.CNTSTUID,

    -- student covariates (with imputation)
    COALESCE(s.ST001D01T, st.mode_grade)   AS ST001D01T,
    COALESCE(s.ST004D01T, st.mode_gender)  AS ST004D01T,
    COALESCE(s.MISCED,    st.mode_misced)  AS MISCED,
    COALESCE(s.IMMIG,     st.mode_immig)   AS IMMIG,
    COALESCE(s.AGE,   st.med_age::numeric(4,2)) AS AGE,
    COALESCE(
	    s.ESCS,
	    st.med_escs::numeric(7,4)
	) AS ESCS,


    -- SMS (no imputation; enforced complete)
    s.ST268Q01JA,
    s.ST268Q04JA,
    s.ST268Q07JA,

    -- SMP (no imputation; enforced complete)
    s.PV1MATH, s.PV2MATH, s.PV3MATH, s.PV4MATH, s.PV5MATH,
    s.PV6MATH, s.PV7MATH, s.PV8MATH, s.PV9MATH, s.PV10MATH,

    -- SPI (no imputation; enforced complete)
    sch.SC064Q01TA,
    sch.SC064Q02TA,
    sch.SC064Q03TA,
    sch.SC064Q05WA,
    sch.SC064Q06WA,
    sch.SC064Q07WA,
    sch.SC064Q04NA,

    -- school covariates (with imputation)
    COALESCE(
	    sch.MCLSIZE,
	    st.med_mclsize::smallint
	) AS MCLSIZE
	,
    COALESCE(
	    sch.SCHSIZE,
	    st.med_schsize::integer
	) AS SCHSIZE


FROM pisa.students s
JOIN pisa.schools sch
  ON s.CNT = sch.CNT
 AND s.CNTSCHID = sch.CNTSCHID
CROSS JOIN stats st

WHERE s.CNT = 'SGP'
AND
    -- SPI complete
    sch.SC064Q01TA IS NOT NULL
    AND sch.SC064Q02TA IS NOT NULL
    AND sch.SC064Q03TA IS NOT NULL
    AND sch.SC064Q04NA IS NOT NULL
    AND sch.SC064Q05WA IS NOT NULL
    AND sch.SC064Q06WA IS NOT NULL
    AND sch.SC064Q07WA IS NOT NULL

    -- SMS complete
    AND s.ST268Q01JA IS NOT NULL
    AND s.ST268Q04JA IS NOT NULL
    AND s.ST268Q07JA IS NOT NULL

    -- SMP complete
    AND s.PV1MATH IS NOT NULL
    AND s.PV2MATH IS NOT NULL
    AND s.PV3MATH IS NOT NULL
    AND s.PV4MATH IS NOT NULL
    AND s.PV5MATH IS NOT NULL
    AND s.PV6MATH IS NOT NULL
    AND s.PV7MATH IS NOT NULL
    AND s.PV8MATH IS NOT NULL
    AND s.PV9MATH IS NOT NULL
    AND s.PV10MATH IS NOT NULL;
