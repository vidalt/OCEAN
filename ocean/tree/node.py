from collections.abc import Hashable

import numpy as np
from anytree import NodeMixin


class Node(NodeMixin):
    _feature: Hashable | None
    _value: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None
    _threshold: float | None
    _code: Hashable | None
    _id: int

    __left: "Node | None" = None
    __right: "Node | None" = None

    def __init__(
        self,
        node_id: int,
        *,
        feature: Hashable | None = None,
        value: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
        parent: "Node | None" = None,
        threshold: float | None = None,
        code: Hashable | None = None,
        left: "Node | None" = None,
        right: "Node | None" = None,
    ) -> None:
        super().__init__()

        self._feature = feature
        self._value = value
        self._threshold = threshold
        self._code = code
        self._id = node_id

        self.parent = parent

        if left is not None:
            self.left = left
        if right is not None:
            self.right = right

    @property
    def feature(self) -> Hashable:
        if self.is_leaf:
            msg = "The feature is only available for non-leaf nodes."
            raise AttributeError(msg)
        return self._feature

    @property
    def value(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        if not self.is_leaf:
            msg = "The value is only available for leaf nodes."
            raise AttributeError(msg)
        if self._value is None:
            msg = "The value has not been set."
            raise AttributeError(msg)
        return self._value

    @property
    def threshold(self) -> float:
        if self.is_leaf:
            msg = "The threshold is only available for non-leaf nodes."
            raise AttributeError(msg)
        if self._threshold is None:
            msg = "The threshold has not been set."
            raise AttributeError(msg)
        return self._threshold

    @property
    def code(self) -> Hashable:
        if self.is_leaf:
            msg = "The code is only available for non-leaf nodes."
            raise AttributeError(msg)
        if self._code is None:
            msg = "The code has not been set."
            raise AttributeError(msg)
        return self._code

    @property
    def node_id(self) -> int:
        return self._id

    @property
    def left(self) -> "Node":
        if self.__left is None:
            msg = "The left child has not been set."
            raise AttributeError(msg)
        return self.__left

    @left.setter
    def left(self, node: "Node") -> None:
        node.parent = self
        if self.__left is not None:
            self.__left.parent = None
        self.__left = node

    @property
    def right(self) -> "Node":
        if self.__right is None:
            msg = "The right child has not been set."
            raise AttributeError(msg)
        return self.__right

    @right.setter
    def right(self, node: "Node") -> None:
        node.parent = self
        if self.__right is not None:
            self.__right.parent = None
        self.__right = node
