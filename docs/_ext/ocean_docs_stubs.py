# ruff: noqa

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


class _GenericAliasMixin:
    @classmethod
    def __class_getitem__(cls, _item: object) -> type[_GenericAliasMixin]:
        return cls


class _Constraint(_GenericAliasMixin):
    def OnlyEnforceIf(self, *_args: object, **_kwargs: object) -> _Constraint:
        return self


class _IntVar(_GenericAliasMixin):
    def Proto(self) -> SimpleNamespace:
        return SimpleNamespace(domain=[])


class _LinearExpr(_GenericAliasMixin):
    @staticmethod
    def WeightedSum(*_args: object, **_kwargs: object) -> int:
        return 0


class _Domain(_GenericAliasMixin):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._intervals: list[int] = []

    @classmethod
    def FromValues(cls, values: list[int]) -> _Domain:
        domain = cls()
        domain._intervals = list(values)
        return domain

    def FlattenedIntervals(self) -> list[int]:
        return list(self._intervals)


class _PBResult:
    clauses: list[list[int]] = []


def _ensure_module(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    return module


def _install_gurobi_stub() -> None:
    try:
        import gurobipy  # noqa: F401
    except Exception:
        module = _ensure_module("gurobipy")

        class Env(_GenericAliasMixin):
            pass

        class Var(_GenericAliasMixin):
            pass

        class Constr(_GenericAliasMixin):
            pass

        class LinExpr(_GenericAliasMixin):
            pass

        class QuadExpr(_GenericAliasMixin):
            pass

        class MVar(_GenericAliasMixin):
            pass

        class MLinExpr(_GenericAliasMixin):
            @staticmethod
            def zeros(*_args: object, **_kwargs: object) -> int:
                return 0

        class tupledict(dict, _GenericAliasMixin):
            pass

        class Model(_GenericAliasMixin):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

        class _Callback:
            MIPSOL = 1
            MIPSOL_OBJ = 2

        class GRB:
            BINARY = "B"
            CONTINUOUS = "C"
            INFINITY = float("inf")
            MINIMIZE = 1
            Callback = _Callback

        module.Env = Env
        module.Var = Var
        module.Constr = Constr
        module.LinExpr = LinExpr
        module.QuadExpr = QuadExpr
        module.MVar = MVar
        module.MLinExpr = MLinExpr
        module.Model = Model
        module.tupledict = tupledict
        module.GRB = GRB


def _install_ortools_stub() -> None:
    try:
        from ortools.sat.python import cp_model
    except Exception:
        ortools = _ensure_module("ortools")
        sat = _ensure_module("ortools.sat")
        python = _ensure_module("ortools.sat.python")
        cp_model = _ensure_module("ortools.sat.python.cp_model")

        class CpModel(_GenericAliasMixin):
            def Add(self, *_args: object, **_kwargs: object) -> _Constraint:
                return _Constraint()

            def AddAbsEquality(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> _Constraint:
                return _Constraint()

            def AddElement(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> _Constraint:
                return _Constraint()

            def AddExactlyOne(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> _Constraint:
                return _Constraint()

            def Minimize(self, *_args: object, **_kwargs: object) -> None:
                return None

            def NewBoolVar(self, *_args: object, **_kwargs: object) -> _IntVar:
                return _IntVar()

            def NewIntVar(self, *_args: object, **_kwargs: object) -> _IntVar:
                return _IntVar()

            def NewIntVarFromDomain(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> _IntVar:
                return _IntVar()

        class CpSolver(_GenericAliasMixin):
            def __init__(self) -> None:
                self.parameters = SimpleNamespace(
                    log_search_progress=False,
                    max_time_in_seconds=0,
                    random_seed=0,
                    num_workers=0,
                )

            def BestObjectiveBound(self) -> int:
                return 0

            def ObjectiveValue(self) -> int:
                return 0

            def Solve(self, *_args: object, **_kwargs: object) -> int:
                return 0

            def Value(self, *_args: object, **_kwargs: object) -> int:
                return 0

            def status_name(self) -> str:
                return "UNKNOWN"

        class CpSolverSolutionCallback(_GenericAliasMixin):
            def __init__(self) -> None:
                pass

            def ObjectiveValue(self) -> int:
                return 0

        cp_model.Constraint = _Constraint
        cp_model.CpModel = CpModel
        cp_model.CpSolver = CpSolver
        cp_model.CpSolverSolutionCallback = CpSolverSolutionCallback
        cp_model.Domain = _Domain
        cp_model.IntVar = _IntVar
        cp_model.LinearExpr = _LinearExpr
        cp_model.ObjLinearExprT = int

        ortools.sat = sat
        sat.python = python
        python.cp_model = cp_model


def _install_pysat_stub() -> None:
    try:
        import pysat.examples.rc2
        import pysat.formula
        import pysat.pb
    except Exception:
        pysat = _ensure_module("pysat")
        formula = _ensure_module("pysat.formula")
        pb = _ensure_module("pysat.pb")
        examples = _ensure_module("pysat.examples")
        rc2 = _ensure_module("pysat.examples.rc2")

        class WCNF(_GenericAliasMixin):
            def __init__(self) -> None:
                self.hard: list[list[int]] = []
                self.soft: list[list[int]] = []
                self.wght: list[int] = []
                self.topw = 1

            def append(
                self,
                lits: list[int],
                weight: int | None = None,
            ) -> None:
                if weight is None:
                    self.hard.append(list(lits))
                else:
                    self.soft.append(list(lits))
                    self.wght.append(weight)

        class IDPool(_GenericAliasMixin):
            def __init__(self) -> None:
                self.obj2id: dict[str, int] = {}

            def id(self, name: str) -> int:
                if name not in self.obj2id:
                    self.obj2id[name] = len(self.obj2id) + 1
                return self.obj2id[name]

        class PBEnc:
            @staticmethod
            def atleast(*_args: object, **_kwargs: object) -> _PBResult:
                return _PBResult()

        class RC2(_GenericAliasMixin):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cost = 0

            def __enter__(self) -> RC2:
                return self

            def __exit__(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> bool:
                return False

            def compute(self) -> list[int]:
                return []

        formula.WCNF = WCNF
        formula.IDPool = IDPool
        pb.PBEnc = PBEnc
        rc2.RC2 = RC2

        pysat.formula = formula
        pysat.pb = pb
        pysat.examples = examples
        examples.rc2 = rc2


def install() -> None:
    _install_gurobi_stub()
    _install_ortools_stub()
    _install_pysat_stub()
