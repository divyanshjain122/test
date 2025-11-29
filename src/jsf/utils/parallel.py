"""Parallel processing utilities for JSF-Core."""

from typing import Callable, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp


def parallel_map(
    func: Callable,
    items: List[Any],
    n_jobs: Optional[int] = None,
    use_threads: bool = False,
    show_progress: bool = False,
) -> List[Any]:
    """
    Apply function to items in parallel.

    Args:
        func: Function to apply
        items: List of items to process
        n_jobs: Number of parallel jobs (None = use all CPUs)
        use_threads: Use threads instead of processes
        show_progress: Show progress bar (requires tqdm)

    Returns:
        List of results
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    if n_jobs == 1:
        # Sequential execution
        return [func(item) for item in items]
    
    # Parallel execution
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=n_jobs) as executor:
        futures = [executor.submit(func, item) for item in items]
        
        if show_progress:
            try:
                from tqdm import tqdm
                futures_iter = tqdm(as_completed(futures), total=len(futures))
            except ImportError:
                futures_iter = as_completed(futures)
        else:
            futures_iter = as_completed(futures)
        
        results = []
        for future in futures_iter:
            results.append(future.result())
    
    return results


def parallel_starmap(
    func: Callable,
    arg_list: List[tuple],
    n_jobs: Optional[int] = None,
    use_threads: bool = False,
) -> List[Any]:
    """
    Apply function to multiple arguments in parallel.

    Args:
        func: Function to apply
        arg_list: List of argument tuples
        n_jobs: Number of parallel jobs (None = use all CPUs)
        use_threads: Use threads instead of processes

    Returns:
        List of results
    """
    wrapper = lambda args: func(*args)
    return parallel_map(wrapper, arg_list, n_jobs=n_jobs, use_threads=use_threads)


def get_optimal_n_jobs(n_tasks: int, max_workers: Optional[int] = None) -> int:
    """
    Determine optimal number of parallel jobs.

    Args:
        n_tasks: Number of tasks to execute
        max_workers: Maximum number of workers (None = use all CPUs)

    Returns:
        Optimal number of jobs
    """
    n_cpus = mp.cpu_count()
    if max_workers is None:
        max_workers = n_cpus
    return min(n_tasks, max_workers, n_cpus)
