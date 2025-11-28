class GarbageManager:
    type GarbageObject = int

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[GarbageObject]

    def __init__(self) -> None:
        self._garbage = []

    def add_garbage(self, *args: GarbageObject) -> None:
        self._garbage.extend(args)

    def remove_garbage(self) -> None:
        self._garbage.clear()

    def garbage_list(self) -> list[GarbageObject]:
        return self._garbage
