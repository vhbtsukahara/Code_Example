# Analytic Dataset (Anonymized) — Equipment-Year Panel

**File:** `analytic\_dataset\_anonymized\_equipment\_year.csv`



## Unit of analysis

One row per **equipment-year** (only rows where an equipment identifier exists in the source).



## Anonymization

* `equip\_hash`, `hospital\_hash`, `brand\_hash`, `model\_hash` are salted SHA-256 hashes (stable within this dataset; not reversible without the salt).
* Direct identifiers (hospital name, CNES, municipality/UF, equipment ID/serial, acquisition year) are **not included**.
* Hospital-size covariates from CNES are provided only as **year-wise z-scores of log(1+x)** to reduce re-identification risk while keeping signal.

> Note: This is anonymization, not a formal de-identification guarantee. Very large/specific hospitals might still be indirectly inferable via patterns.



## Columns



### Identifiers (hashed)

* `equip\_hash`: anonymized equipment ID
* `hospital\_hash`: anonymized hospital ID
* `brand\_hash`: anonymized manufacturer/brand
* `model\_hash`: anonymized model
* 

### Time

* `year`: calendar year (int)
* 

### Equipment features (observed)

* `age\_at\_year`: age in years in that year
* `cm\_count`: number of corrective maintenance events in that year
* `total\_downtime\_hours`: total downtime hours in that year
* `positive\_downtime\_count`: count of downtime episodes > 0
* `max\_downtime`: maximum downtime episode (hours)
* `any\_downtime`: 1 if any downtime in year, else 0



### Hospital context (CNES-derived, normalized)

Each of these is a z-score computed **within the same year** of `log(1 + raw\_count)`:

* `LEITOS\_EXISTENTES\_z`
* `LEITOS\_SUS\_z`
* `UTI\_TOTAL\_EXIST\_z`
* `UTI\_TOTAL\_SUS\_z`
* `UTI\_ADULTO\_EXIST\_z`
* `UTI\_ADULTO\_SUS\_z`
* `UTI\_PEDIATRICO\_EXIST\_z`
* `UTI\_PEDIATRICO\_SUS\_z`
* `UTI\_NEONATAL\_EXIST\_z`
* `UTI\_NEONATAL\_SUS\_z`

