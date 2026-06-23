from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

from numba import njit as _numba_njit  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from collections.abc import Callable


@overload
def njit[**P, R](func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def njit[**P, R](
    **kwargs: object,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def njit[**P, R](
    func: Callable[P, R] | None = None,
    /,
    **kwargs: object,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    if func is not None:
        return cast(
            "Callable[P, R]",
            _numba_njit(func, **kwargs),
        )

    def decorator(inner: Callable[P, R]) -> Callable[P, R]:
        return cast(
            "Callable[P, R]",
            _numba_njit(inner, **kwargs),
        )

    return decorator


__all__ = ["njit"]
