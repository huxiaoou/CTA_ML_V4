import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qcalendar import CCalendar
from husfort.qinstruments import CInstruMgr
from husfort.qsimulation import CMgrMktData, CMgrMajContract, CSignal, CSimulation
from husfort.qsimulation import TExePriceType
from husfort.qsqlite import CDbStruct
from husfort.qutility import error_handler
from solutions.mclrn import CTestMclrn
from solutions.signals import gen_sig_db
from typedef import TReturnClass


def covert_tests_to_sims(tests: list[CTestMclrn], signals_dir: str) -> list[tuple[CSignal, TExePriceType]]:
    sims: list[tuple[CSignal, TExePriceType]] = []
    for test in tests:
        signal_db_struct = gen_sig_db(save_dir=signals_dir, save_id=test.save_id)
        signal = CSignal(sid=test.save_id, signal_db_struct=signal_db_struct)
        if test.test_data.ret.ret_class == TReturnClass.OPN:
            sims.append((signal, TExePriceType.OPEN))
        else:
            sims.append((signal, TExePriceType.CLOSE))
    return sims


def process_for_sim(
        signal: CSignal,
        init_cash: float,
        cost_rate: float,
        exe_price_type: TExePriceType,
        mgr_instru: CInstruMgr,
        mgr_maj_contract: CMgrMajContract,
        mgr_mkt_data: CMgrMktData,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        sim_save_dir: str,
):
    sim = CSimulation(
        signal=signal,
        init_cash=init_cash,
        cost_rate=cost_rate,
        exe_price_type=exe_price_type,
        mgr_instru=mgr_instru,
        mgr_maj_contract=mgr_maj_contract,
        mgr_mkt_data=mgr_mkt_data,
        sim_save_dir=sim_save_dir,
    )
    sim.main(bgn_date=bgn_date, stp_date=stp_date, calendar=calendar)
    return 0


def main_sims(
        tests: list[CTestMclrn],
        signals_dir: str,
        init_cash: float,
        cost_rate: float,
        instru_info_path: str,
        universe: list[str],
        preprocess: CDbStruct,
        fmd: CDbStruct,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        sim_save_dir: str,
        call_multiprocess: bool,
        processes: int,
):
    sims = covert_tests_to_sims(tests, signals_dir)
    mgr_instru = CInstruMgr(instru_info_path, key="tushareId")
    mgr_maj_contract = CMgrMajContract(universe, preprocess)
    mgr_mkt_data = CMgrMktData(fmd)
    desc = "Do simulations for signals"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for signal, exe_price_type in sims:
                    pool.apply_async(
                        process_for_sim,
                        kwds={
                            "signal": signal,
                            "init_cash": init_cash,
                            "cost_rate": cost_rate,
                            "exe_price_type": exe_price_type,
                            "mgr_instru": mgr_instru,
                            "mgr_maj_contract": mgr_maj_contract,
                            "mgr_mkt_data": mgr_mkt_data,
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                            "sim_save_dir": sim_save_dir,
                        },
                        callback=lambda _: pb.update(task_id=main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for signal, exe_price_type in track(sims, description=desc):
            process_for_sim(
                signal=signal,
                init_cash=init_cash,
                cost_rate=cost_rate,
                exe_price_type=exe_price_type,
                mgr_instru=mgr_instru,
                mgr_maj_contract=mgr_maj_contract,
                mgr_mkt_data=mgr_mkt_data,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                sim_save_dir=sim_save_dir,
            )
    return 0
