"""Explicit bootstrap for isolated Ascend reference-loss scoring servers."""
import os
if os.environ.get('MOQE_ASCEND_INT4_ADAPTER') == '1':
    try:
        from adapter_patch import install
        install()
        if os.environ.get('MOQE_SCORE_CPU_SAMPLER') == '1':
            from ascend_scoring_cpu_sampler import install as install_sampler
            install_sampler()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
