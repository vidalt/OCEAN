from collections.abc import Iterator

from .node import Node


class Tree:
    root: Node
    _shape: tuple[int, ...]

    def __init__(self, root: Node) -> None:
        self.root = root
        self._shape = root.leaves[0].value.shape

    @property
    def n_nodes(self) -> int:
        return self.root.size

    @property
    def max_depth(self) -> int:
        return self.root.height

    @property
    def leaves(self) -> tuple[Node, *tuple[Node, ...]]:
        return self.root.leaves

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def nodes_at(self, depth: int) -> Iterator[Node]:
        if depth < 0:
            msg = "The depth must be non-negative."
            raise ValueError(msg)
        return self._nodes_at(self.root, depth=depth)

    def _nodes_at(self, node: Node, *, depth: int) -> Iterator[Node]:
        if depth == 0:
            yield node
        elif depth > 0:
            for child in node.children:
                yield from self._nodes_at(child, depth=depth - 1)
