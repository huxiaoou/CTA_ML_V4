import argparse
from typedef import CCfgFactors, TRets


def parse_args(cfg_facs: CCfgFactors, rets: TRets):
    arg_parser = argparse.ArgumentParser(description="To calculate data, such as macro and forex")
    arg_parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true",
                            help="not using multiprocess, for debug. Works only when switch in (factor,)")
    arg_parser.add_argument("--processes", type=int, default=None,
                            help="number of processes to be called, effective only when nomp = False")
    arg_parser.add_argument("--verbose", default=False, action="store_true",
                            help="whether to print more details, effective only when sub function = (feature_selection,)")

    arg_parser_subs = arg_parser.add_subparsers(
        title="Position argument to call sub functions",
        dest="switch",
        description="use this position argument to call different functions of this project. "
                    "For example: 'python main.py --bgn 20120104 --stp 20240826 available'",
        required=True,
    )

    # switch: available
    arg_parser_subs.add_parser(name="available", help="Calculate available universe")

    # switch: market
    arg_parser_subs.add_parser(name="market", help="Calculate market universe")

    # switch: test return
    arg_parser_subs.add_parser(name="test_return", help="Calculate test returns")

    # switch: factor
    arg_parser_sub = arg_parser_subs.add_parser(name="factor", help="Calculate factor")
    arg_parser_sub.add_argument(
        "--fclass", type=str,
        help="factor class to run",
        required=True, choices=cfg_facs.classes,
    )

    # switch: test return
    arg_parser_sub = arg_parser_subs.add_parser(name="ic_test", help="Calculate ic_tests")
    arg_parser_sub.add_argument(
        "--fclass", type=str,
        help="factor class to test",
        required=True, choices=cfg_factors.classes,
    )
    arg_parser_sub.add_argument(
        "--ret", type=str,
        help="return to test",
        required=True, choices=[r.ret_name for r in rets],
    )
    arg_parser_sub.add_argument(
        "--neu", default=False, action="store_true",
        help="use this to set fac type and ret as 'neu'",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    import sys
    from loguru import logger
    from config import proj_cfg, db_struct_cfg, cfg_factors
    from husfort.qlog import define_logger
    from husfort.qcalendar import CCalendar
    from solutions.shared import get_avlb_db, get_market_db
    from typedef import CRet, TFacRetType

    define_logger()

    calendar = CCalendar(proj_cfg.calendar_path)
    args = parse_args(cfg_facs=cfg_factors, rets=proj_cfg.all_rets)
    bgn_date, stp_date = args.bgn, args.stp or calendar.get_next_date(args.bgn, shift=1)

    if args.switch == "available":
        from solutions.available import main_available

        main_available(
            bgn_date=bgn_date, stp_date=stp_date,
            universe=proj_cfg.universe,
            cfg_avlb_unvrs=proj_cfg.avlb_unvrs,
            db_struct_preprocess=db_struct_cfg.preprocess,
            db_struct_avlb=get_avlb_db(proj_cfg.available_dir),
            calendar=calendar,
        )
    elif args.switch == "market":
        from solutions.market import main_market

        main_market(
            bgn_date=bgn_date, stp_date=stp_date,
            calendar=calendar,
            db_struct_avlb=get_avlb_db(proj_cfg.available_dir),
            db_struct_mkt=get_market_db(proj_cfg.market_dir),
            path_mkt_idx_data=proj_cfg.market_index_path,
            mkt_idxes=proj_cfg.mkt_idxes.idxes,
            sectors=proj_cfg.const.SECTORS,
        )
    elif args.switch == "test_return":
        from solutions.test_return import CTestReturnsByInstru, CTestReturnsAvlb

        for ret in proj_cfg.all_rets:
            test_returns_by_instru = CTestReturnsByInstru(
                ret=ret, universe=proj_cfg.universe,
                test_returns_by_instru_dir=proj_cfg.test_returns_by_instru_dir,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
            test_returns_by_instru.main(bgn_date, stp_date, calendar)
            test_returns_avlb = CTestReturnsAvlb(
                ret=ret, universe=proj_cfg.universe,
                test_returns_by_instru_dir=proj_cfg.test_returns_by_instru_dir,
                test_returns_avlb_raw_dir=proj_cfg.test_returns_avlb_raw_dir,
                test_returns_avlb_neu_dir=proj_cfg.test_returns_avlb_neu_dir,
                db_struct_avlb=get_avlb_db(proj_cfg.available_dir),
            )
            test_returns_avlb.main(bgn_date, stp_date, calendar)
    elif args.switch == "factor":
        from solutions.factor import CFactorsAvlb

        cfg = getattr(cfg_factors, args.fclass)
        if cfg is None:
            logger.warning(f"No cfg for {args.fclass}")
            sys.exit(0)
        if args.fclass == "MTM":
            from solutions.factorAlg import CFactorMTM

            fac = CFactorMTM(
                cfg=cfg,
                factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                universe=proj_cfg.universe,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
        elif args.fclass == "SKEW":
            from solutions.factorAlg import CFactorSKEW

            fac = CFactorSKEW(
                cfg=cfg,
                factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                universe=proj_cfg.universe,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
        elif args.fclass == "KURT":
            from solutions.factorAlg import CFactorKURT

            fac = CFactorKURT(
                cfg=cfg,
                factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                universe=proj_cfg.universe,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
        elif args.fclass == "RS":
            from solutions.factorAlg import CFactorRS

            fac = CFactorRS(
                cfg=cfg,
                factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                universe=proj_cfg.universe,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
        else:
            raise NotImplementedError(f"fclass = {args.fclass}")

        fac.main(
            bgn_date=bgn_date, stp_date=stp_date, calendar=calendar,
            call_multiprocess=not args.nomp, processes=args.processes,
        )
        fac_avlb = CFactorsAvlb(
            factor_grp=cfg,
            universe=proj_cfg.universe,
            factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
            factors_avlb_raw_dir=proj_cfg.factors_avlb_raw_dir,
            factors_avlb_neu_dir=proj_cfg.factors_avlb_neu_dir,
            db_struct_avlb=get_avlb_db(proj_cfg.available_dir),
        )
        fac_avlb.main(bgn_date, stp_date, calendar)
    elif args.switch == "ic_test":
        from solutions.ic_tests import CICTest

        factor_grp = getattr(cfg_factors, args.fclass)
        ret = CRet.parse_from_name(args.ret)
        if args.neu:
            factor_type = ret_type = TFacRetType.NEU
            factors_avlb_dir = proj_cfg.factors_avlb_neu_dir
            test_returns_avlb_dir = proj_cfg.test_returns_avlb_neu_dir
        else:
            factor_type = ret_type = TFacRetType.RAW
            factors_avlb_dir = proj_cfg.factors_avlb_raw_dir
            test_returns_avlb_dir = proj_cfg.test_returns_avlb_raw_dir

        print(f"factor_type = {factor_type}, ret_type = {ret_type}")

        ic_test = CICTest(
            factor_grp=factor_grp,
            factor_type=factor_type,
            ret=ret,
            ret_type=ret_type,
            factors_avlb_dir=factors_avlb_dir,
            test_returns_avlb_dir=test_returns_avlb_dir,
            ic_tests_dir=proj_cfg.ic_tests_dir,
        )
        ic_test.main(bgn_date, stp_date, calendar)
        ic_test.main_summary(bgn_date, stp_date, calendar)
