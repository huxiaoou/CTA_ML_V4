# README

## data structure

### test_returns

+ test_returns_by_instru
    + opn
        + `I[0],I[1],...I[k-1]`
    + cls
        + `I[0],I[1],...I[k-1]`

+ test_returns_avlb_raw(avlb)
    + opn.db
    + cls.db

+ test_returns_avlb_neu(avlb + neutralize)
    + opn.db
    + cls.db

### factors

make sure the algorithm of each factor is symmetrical about 0

+ factors_by_instru
    + F1
        + `I[0],I[1],...I[k-1]`
    + F2
        + `I[0],I[1],...I[k-1]`
    + F3
        + `I[0],I[1],...I[k-1]`
    + ...

+ factors_avlb_raw(avlb + winsorize + normalize)
    + F1.db
    + F2.db
    + F3.db
    + ...

+ factors_avlb_neu(avlb + neutralize)
    + F1.db
    + F2.db
    + F3.db
    + ...
