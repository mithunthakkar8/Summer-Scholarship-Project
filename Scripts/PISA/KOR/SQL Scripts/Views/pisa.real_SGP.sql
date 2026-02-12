CREATE OR REPLACE VIEW pisa.real_SGP AS
SELECT
    s.CNT,
    s.CNTSCHID,
    s.CNTSTUID,

    -- student covariates
    s.ST001D01T,
    s.ST004D01T,
    s.MISCED,
    s.IMMIG,
    s.AGE,
    s.ESCS,

    -- SMS
    s.ST268Q01JA,
    s.ST268Q04JA,
    s.ST268Q07JA,

    -- SMP
    s.PV1MATH, s.PV2MATH, s.PV3MATH, s.PV4MATH, s.PV5MATH,
    s.PV6MATH, s.PV7MATH, s.PV8MATH, s.PV9MATH, s.PV10MATH,

    -- SPI (recode WA)
    sch.SC064Q01TA,
    sch.SC064Q02TA,
    sch.SC064Q03TA,
    sch.SC064Q05WA,
    sch.SC064Q06WA,
    sch.SC064Q07WA,
    sch.SC064Q04NA,

    sch.MCLSIZE,
    sch.SCHSIZE

FROM pisa.students s
JOIN pisa.schools sch
  ON s.CNT = sch.CNT
 AND s.CNTSCHID = sch.CNTSCHID
WHERE s.CNT = 'SGP';
