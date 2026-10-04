"""Load the converted Ascend INT4 checkpoint adapter; keep vLLM sampling native."""
import os
if os.environ.get('MOQE_ASCEND_INT4_ADAPTER') == '1':
    try:
        from adapter_patch import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
