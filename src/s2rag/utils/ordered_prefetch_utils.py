from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, Optional


class OrderedBoundedPrefetch(Iterator[Any]):
    """Run a bounded number of calls concurrently and yield in input order."""

    def __init__(
        self,
        function: Callable[[Any], Any],
        values: Iterable[Any],
        *,
        max_workers: int,
        prefetch_window: Optional[int] = None,
    ) -> None:
        self._function = function
        self._values = iter(values)
        self._max_workers = max(1, int(max_workers))
        self._prefetch_window = max(
            self._max_workers,
            int(prefetch_window or self._max_workers * 2),
        )
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._futures: Dict[int, Future[Any]] = {}
        self._next_submit = 0
        self._next_yield = 0
        self._source_exhausted = False
        self._closed = False
        self._fill_window()

    def _fill_window(self) -> None:
        while (
            not self._source_exhausted
            and len(self._futures) < self._prefetch_window
        ):
            try:
                value = next(self._values)
            except StopIteration:
                self._source_exhausted = True
                break
            index = self._next_submit
            self._next_submit += 1
            self._futures[index] = self._executor.submit(self._function, value)

    def __iter__(self) -> "OrderedBoundedPrefetch":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        future = self._futures.get(self._next_yield)
        if future is None:
            self.close()
            raise StopIteration
        try:
            result = future.result()
        except BaseException:
            self.close(wait=False, cancel_futures=True)
            raise
        del self._futures[self._next_yield]
        self._next_yield += 1
        self._fill_window()
        return result

    def close(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = False,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(
            wait=wait,
            cancel_futures=cancel_futures,
        )

    def __enter__(self) -> "OrderedBoundedPrefetch":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(
            wait=exc_type is None,
            cancel_futures=exc_type is not None,
        )
