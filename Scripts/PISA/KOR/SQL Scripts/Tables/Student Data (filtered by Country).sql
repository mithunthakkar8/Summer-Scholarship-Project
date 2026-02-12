SELECT *
FROM pisa_students_keys k
LEFT JOIN pisa_students_part1 p1 USING (CNT, CNTSCHID, CNTSTUID)
LEFT JOIN pisa_students_part2 p2 USING (CNT, CNTSCHID, CNTSTUID)
LEFT JOIN pisa_students_part3 p3 USING (CNT, CNTSCHID, CNTSTUID)
LEFT JOIN pisa_students_part4 p4 USING (CNT, CNTSCHID, CNTSTUID)
LEFT JOIN pisa_students_part5 p5 USING (CNT, CNTSCHID, CNTSTUID)
where CNT in ('JPN', 'KOR', 'SGP', 'TAP', 'HKG', 'MAC')