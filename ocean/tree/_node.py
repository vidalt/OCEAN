from anytree import NodeMixin

from ..typing import Array, Key, NonNegativeInt, NonNegativeNumber
from ._utils import average_length


class Node(NodeMixin):
    _feature: Key | None
    _value: Array | None
    _threshold: float | None
    _code: Key | None
    _id: NonNegativeInt
    _n_samples: NonNegativeInt

    __left: "Node | None" = None
    __right: "Node | None" = None

    def __init__(
        self,
        node_id: NonNegativeInt,
        *,
        feature: Key | None = None,
        value: Array | None = None,
        parent: "Node | None" = None,
        threshold: float | None = None,
        code: Key | None = None,
        n_samples: NonNegativeInt = 0,
        left: "Node | None" = None,
        right: "Node | None" = None,
    ) -> None:
        super().__init__()

        self._feature = feature
        self._value = value
        self._threshold = threshold
        self._code = code
        self._id = node_id
        self._n_samples = n_samples

        self.parent = parent

        if left is not None:
            self.left = left
        if right is not None:
            self.right = right

    @property
    def node_id(self) -> NonNegativeInt:
        return self._id

    @property
    def feature(self) -> Key:
        if self.is_leaf:
            msg = "The feature is only available for non-leaf nodes."
            raise AttributeError(msg)
        if self._feature is None:
            msg = (
                "Internal node does not have a feature. The tree is corrupted."
            )
            raise AttributeError(msg)
        return self._feature

    @property
    def value(self) -> Array:
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
    def code(self) -> Key:
        if self.is_leaf:
            msg = "The code is only available for non-leaf nodes."
            raise AttributeError(msg)
        if self._code is None:
            msg = "The code has not been set."
            raise AttributeError(msg)
        return self._code

    @property
    def n_samples(self) -> NonNegativeInt:
        return self._n_samples

    @property
    def length(self) -> NonNegativeNumber:
        return self.depth + average_length(self.n_samples)

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
