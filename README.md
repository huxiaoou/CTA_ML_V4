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

### ic-tests

+ data
    + F1_RAW_Cls001L1_RAW.db
    + F1_NEU_Cls001L1_NEU.db
    + ...
+ plot
    + F1_RAW_Cls001L1_RAW.pdf
    + F1_NEU_Cls001L1_NEU.pdf
    + ...

## Steps to add new factor

config.yaml

+ update args

typedef.py

+ update TFactorClass
+ new class CCfgFactorGrpNewFactor
+ update CCfgFactors

config.py

+ update import
+ update cfg_factors

factorAlg.py

+ from typedef import CCfgFactorGrpNewFactor
+ new class CFactorNewFactor
+ update pick_factor

run_all.ps1
+ factor
+ ic

## Steps to add new models

config.yaml

+ update test_models

typedef.py

+ update TModelType

mclrn.py

+ update import
+ give model definition

mclrn_parser.py

+ update import
+ update parse_config_to_mclrn_test
