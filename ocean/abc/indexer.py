from collections.abc import Callable


class Indexer[K, V]:
    def __init__(self, getter: Callable[[K], V]) -> None:
        self._getter = getter

    def __getitem__(self, key: K) -> V:
        return self._getter(key)
