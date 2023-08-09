"""
这个文件主要是写了一个多进程计算框架，利用该框架可以加速因子的计算过程
"""
__author__ = 'xhs@我是天空飘着的雾霾'

import multiprocessing
from tqdm import tqdm
from collections import defaultdict


class Accelerator:
    def __init__(self, core=None):
        if core is None:
            self.core = multiprocessing.cpu_count()
        else:
            self.core = core

    def execute(self, functions_and_parameters, timeout=None):
        pool = multiprocessing.Pool(processes=self.core)
        async_dict = defaultdict(list)
        total_tasks = sum(len(parameters) for _, parameters in functions_and_parameters)
        def handle_result(result):
            yield result
        try:
            for func, parameters in functions_and_parameters:
                for param_set in parameters:
                    async_result = pool.apply_async(func, args=param_set, callback=handle_result)
                    param_set2 = tuple(x if isinstance(x, str) or
                                            isinstance(x, int) or
                                            isinstance(x, float) else 'Unhashable_Type' for x in param_set)
                    async_dict[(func.__name__, param_set2)].append(async_result)
            with tqdm(total=total_tasks, desc="Progress") as pbar:
                for func_name, async_results in async_dict.items():
                    for async_result in async_results:
                        try:
                            res = async_result.get(timeout=timeout)
                        except multiprocessing.TimeoutError:
                            res = "Timeout"
                        except Exception as e:
                            res = e
                        finally:
                            yield func_name, res
                            pbar.update(1)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
