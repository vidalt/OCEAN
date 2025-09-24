import gurobipy as gp

type Objective = gp.LinExpr | gp.QuadExpr
