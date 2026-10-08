# V6.4-C.2 preference-conditioned diffusion warm start

Fixed-budget pilot using real trained weights and unchanged C.1 control/safety execution.

| Endpoint | Preference | Full Task /4 | Five gates /4 | Full +30mm /4 | NO_PLAN |
|---|---|---:|---:|---:|---:|
|R8|A|2|2|0|2|
|R8|B|2|2|2|2|
|R12|A|4|4|0|0|
|R12|B|2|2|2|2|
|N8|A|4|4|0|0|
|N8|B|2|2|2|2|
|D8|A|4|4|0|0|
|D8|B|2|2|2|2|

Actual failures and NO_PLAN stay in the denominator. Four-slot curves are prediction only.
Teacher records distinguish validated execution from complete prediction. Family near-optimal labels are not global optima.

Tables A–D and machine summary retain raw metrics, costs, first hits, censoring and lineage.
Eight versus twelve slots is a configured quota reduction; it alone is no measured speedup.
Learning benefit is NOT_ESTABLISHED pending paired quality/cost interpretation; default remains C.1 rule.
Deployment NOT_MET; continuous-time and hardware safety NOT_ESTABLISHED.
