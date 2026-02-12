DROP TABLE IF EXISTS pisa.variable_mapping;
CREATE TABLE pisa.variable_mapping (
    variable_code        VARCHAR(32)  NOT NULL,   -- e.g. ST268Q01JA
    full_name            TEXT         NOT NULL,   -- human-readable description
    dataset_level        VARCHAR(16)  NOT NULL,   -- student | school | teacher | etc.
    canonical_name           TEXT         NOT NULL,   -- snake_case analytical name

    value_type           VARCHAR(32),             -- numeric, categorical, index, plausible_value

    -- Governance
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- Constraints
    CONSTRAINT pk_pisa_variable_mapping
        PRIMARY KEY (variable_code, dataset_level),

    CONSTRAINT uq_pisa_variable_mapping_canonical
        UNIQUE (canonical_name, dataset_level),

    CONSTRAINT chk_dataset_level
        CHECK (dataset_level IN ('student', 'school'))
);
