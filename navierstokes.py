# Copyright (C) 2017 Giancarlo Antonucci
#
# This file accompanies the paper "Galerkin Least Squares Finite Element
# Method for the Incompressible Navier-Stokes Equations with FEniCS", by
# Giancarlo Antonucci, submitted as part of the MSc in Mathematical 
# Modelling and Scientific Computing at The University of Oxford.

from dolfin import *
from mshr import *

class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -0.5) and between(x[1], (0.95, 1.05)))

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (-0.5, 0.0)) and near(x[1], 1.05)) or \
	       (near(x[0], 0.0) and between(x[1], (1.05, 2.0))) or \
               (between(x[0], (0.0, 0.9)) and near(x[1], 2.0)) or \
	       (near(x[0], 0.9) and between(x[1], (1.0, 2.0))) or \
               (between(x[0], (0.9, 1.1)) and near(x[1], 1.0)) or \
	       (near(x[0], 1.1) and between(x[1], (1.0, 2.0))) or \
               (between(x[0], (1.1, 2.9)) and near(x[1], 2.0)) or \
	       (near(x[0], 2.9) and between(x[1], (1.0, 2.0))) or \
               (between(x[0], (2.9, 3.1)) and near(x[1], 1.0)) or \
	       (near(x[0], 3.1) and between(x[1], (1.0, 2.0))) or \
               (between(x[0], (3.1, 5.0)) and near(x[1], 2.0)) or \
	       (near(x[0], 5.0) and between(x[1], (1.05, 2.0))) or \
               (between(x[0], (5.0, 5.5)) and near(x[1], 1.05)) or \
               (between(x[0], (-0.5, 0.0)) and near(x[1], 0.95)) or \
	       (near(x[0], 0.0) and between(x[1], (0.0, 0.95))) or \
               (between(x[0], (0.0, 1.9)) and near(x[1], 0.0)) or \
	       (near(x[0], 1.9) and between(x[1], (0.0, 1.0))) or \
               (between(x[0], (1.9, 2.1)) and near(x[1], 1.0)) or \
	       (near(x[0], 2.1) and between(x[1], (0.0, 1.0))) or \
               (between(x[0], (2.1, 3.9)) and near(x[1], 0.0)) or \
	       (near(x[0], 3.9) and between(x[1], (0.0, 1.0))) or \
               (between(x[0], (3.9, 4.1)) and near(x[1], 1.0)) or \
	       (near(x[0], 4.1) and between(x[1], (0.0, 1.0))) or \
               (between(x[0], (4.1, 5.0)) and near(x[1], 0.0)) or \
	       (near(x[0], 5.0) and between(x[1], (0.0, 0.95))) or \
               (between(x[0], (5.0, 5.5)) and near(x[1], 0.95))

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 5.5) and between(x[1], (0.95, 1.05)))

# Domain
channel = Rectangle(Point(0, 0), Point(5.0, 2.0))
wall_1 = Rectangle(Point(0.9, 1.0), Point(1.1, 2.0))
wall_2 = Rectangle(Point(1.9, 0.0), Point(2.1, 1.0))
wall_3 = Rectangle(Point(2.9, 1.0), Point(3.1, 2.0))
wall_4 = Rectangle(Point(3.9, 0.0), Point(4.1, 1.0))
tube_in = Rectangle(Point(-0.5, 0.95), Point(0.0, 1.05))
tube_out = Rectangle(Point(5.0, 0.95), Point(5.5, 1.05))
domain = channel - wall_1 - wall_2 - wall_3 - wall_4 + tube_in + tube_out

# Mesh
mesh = generate_mesh(domain, 50)

# Initialize mesh function for boundaries
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
Inflow().mark(boundaries, 1)
Noslip().mark(boundaries, 2)
Outflow().mark(boundaries, 3)

# Define new measures associated with the boundaries
# ds = ds(subdomain_data = boundaries)

# Save mesh and boundaries
file = File('results/mesh.pvd')
file << mesh
file << boundaries

# Define function spaces
V = VectorElement("CG", mesh.ufl_cell(), 2)	# velocity
Q = FiniteElement("CG", mesh.ufl_cell(), 1)	# pressure
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define unknowns and test functions
w = Function(W)
(u, p) = split(w)
y = TestFunction(W)
(v, q) = split(y)

# Define functions for solutions at previous time steps
w0 = Function(W)
(u0, p0) = split(w0)

# Define boundary conditions
bc_inflow = DirichletBC(W.sub(0), Constant((1, 0)), boundaries, 1) # inflow profile
bc_walls = DirichletBC(W.sub(0), Constant((0, 0)), boundaries, 2)  # zero on walls
bc_inlet = DirichletBC(W.sub(1), Constant(20.0), boundaries, 1)    # inlet pressure
bc_outlet = DirichletBC(W.sub(1), Constant(0.0), boundaries, 3)    # outlet pressure
bcs = [bc_inflow, bc_walls, bc_inlet, bc_outlet]

# Define parameters and coefficients
T = 10.0        # end time
dt = 0.01	# time step size
nu = 0.0001	# viscosity

k = Constant(dt)
f = Constant((0, 0))	# body force
h = CellSize(mesh)	# mesh size

# Define variational problem with GLS stabilisation
F = (
        (1/k)*inner(u - u0, v)*dx
      + inner(grad(u)*u, v)*dx
      + nu*inner(grad(u), grad(v))*dx
      - inner(p, div(v))*dx
      + inner(div(u), q)*dx
      - inner(f, v)*dx
      )

F += h*inner(grad(q) + grad(v)*u - nu*div(grad(v)), grad(p) + grad(u)*u - nu*div(grad(u)))*dx

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = 0
while True:

    # Update current time
    t += dt
    print "t =", t

    # Compute solution
    solve(F == 0, w, bcs)

    # Update solution
    w0.assign(w)

    # Save to files
    (u, p) = w.split()
    ufile << u
    pfile << p

    # Check
    if t >= T: break
