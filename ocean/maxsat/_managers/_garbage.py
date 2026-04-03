class GarbageManager:
    """Track temporary MaxSAT clause identifiers tied to one query."""

    type GarbageObject = int

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[GarbageObject]

    def __init__(self) -> None:
        """Initialize the container used for query-specific clause ids."""
        self._garbage = []

    def add_garbage(self, *args: GarbageObject) -> None:
        """Register temporary clause identifiers created for one query."""
        self._garbage.extend(args)

    def remove_garbage(self) -> None:
        """Forget all registered temporary clause identifiers."""
        self._garbage.clear()

    def garbage_list(self) -> list[GarbageObject]:
        """
        Return the registered clause identifiers in insertion order.

        Returns
        -------
        list[GarbageObject]
            Registered clause identifiers.

        """
        return self._garbage
