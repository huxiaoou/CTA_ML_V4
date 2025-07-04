import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qcalendar import CCalendar
from husfort.qutility import qtimer, check_and_makedirs, error_handler
from husfort.qsimquick import CSimQuick
from typedefs.typedefReturns import CRet, TReturnClass
from solutions.test_return import CTestReturnLoader
from solutions.mclrn import CTestMclrn
from solutions.signals import CSignalsLoader

TSimQuickArgs = tuple[CSignalsLoader, CTestReturnLoader]


def covert_tests_to_sims_quick_args(
        tests: list[CTestMclrn],
        signals_dir: str,
        test_returns_avlb_raw_dir: str,
) -> list[TSimQuickArgs]:
    sim_quick_args: list[TSimQuickArgs] = []
    for test in tests:
        signal = CSignalsLoader(signals_dir=signals_dir, signal_id=test.save_id)
        if test.test_data.ret.ret_class == TReturnClass.OPN:
            ret = CRet.parse_from_name("Opn001L1")
        else:
            ret = CRet.parse_from_name("Cls001L1")
        test_return_loader = CTestReturnLoader(ret, test_returns_avlb_raw_dir)
        sim_quick_args.append((signal, test_return_loader))
    return sim_quick_args


@qtimer
def main_sims_quick(
        tests: list[CTestMclrn],
        signals_dir: str,
        test_returns_avlb_raw_dir: str,
        cost_rate: float,
        sims_quick_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    check_and_makedirs(sims_quick_dir)
    sim_quick_args = covert_tests_to_sims_quick_args(tests, signals_dir, test_returns_avlb_raw_dir)
    desc = "Do quick simulations"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(sim_quick_args))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for signals_loader, test_return_loader in sim_quick_args:
                    sim_quick = CSimQuick(signals_loader, test_return_loader, cost_rate, sims_quick_dir)
                    pool.apply_async(
                        sim_quick.main,
                        kwds={
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(task_id=main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for signals_loader, test_return_loader in track(sim_quick_args, description=desc):
            sim_quick = CSimQuick(signals_loader, test_return_loader, cost_rate, sims_quick_dir)
            sim_quick.main(bgn_date, stp_date, calendar)
    return 0
