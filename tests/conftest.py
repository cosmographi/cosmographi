from typing import Callable

import pytest

from pytest_codspeed import BenchmarkFixture


def pedanticWrapper(benchmark_fixture):

    def pedanticWrapperInner(target: Callable, *args, **kwargs):
        """Wrap the JAX function in a lambda with args and kwargs and
        run with warmup and async dispatch check.
        See https://docs.jax.dev/en/latest/benchmarking.html

        Parameters
           ----------
           target : Callable
               The function to benchmark.
           args : List
               The args to ``target``.
           kwargs : Dict
               The kwargs to ``target``.
        """
        _target = lambda: target(*args, **kwargs).block_until_ready()

        return benchmark_fixture.pedantic(target=_target, rounds=1, warmup_rounds=5, iterations=50)

    return pedanticWrapperInner


@pytest.fixture(scope="function")
# we need the benchmark fixture for setup
def benchmark_jax(request: pytest.FixtureRequest, benchmark) -> Callable:
    return pedanticWrapper(BenchmarkFixture(request))
