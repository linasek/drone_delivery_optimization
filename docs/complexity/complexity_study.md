# Complexity Study

## 1. Introduction

The Drone-Based Delivery Optimization Problem consists of assigning customers
to a fleet of drones and determining the order in which each drone visits its
assigned customers. Each drone must start at the depot, serve its assigned
customers, and return to the depot while respecting payload capacity, battery
capacity, and no-fly-zone restrictions. The objective is to minimize total
energy consumption.

This problem belongs to the family of combinatorial optimization problems. Its
difficulty comes from two coupled decisions:

- partitioning customers among drones,
- sequencing customers within each drone route.

The first decision is an assignment problem, while the second decision is a
routing problem. When these decisions are combined with resource constraints
such as payload and battery capacity, the search space grows extremely quickly.
As a result, exact optimization methods become computationally expensive for
large instances, and heuristics or metaheuristics become necessary.

## 2. Relation to the Traveling Salesman Problem

The Traveling Salesman Problem (TSP) asks for the shortest tour that starts at
a depot, visits every customer exactly once, and returns to the depot. TSP is a
classical NP-hard problem.

The drone delivery problem contains TSP as a special case. Consider a simplified
version of the drone problem with the following assumptions:

- there is only one drone,
- the drone has unlimited payload capacity,
- the drone has unlimited battery capacity,
- there are no no-fly zones,
- energy consumption is proportional only to distance,
- all customers must be served by the single drone.

Under these assumptions, the drone problem becomes exactly the TSP: the only
remaining decision is to determine the order in which the single drone visits
all customers and returns to the depot. Therefore, any algorithm capable of
solving the general drone delivery problem optimally would also be capable of
solving TSP optimally.

Since TSP is NP-hard, and TSP is a special case of the drone delivery problem,
the drone delivery problem is at least as hard as TSP.

## 3. Relation to the Vehicle Routing Problem

The Vehicle Routing Problem (VRP) generalizes TSP by considering multiple
vehicles instead of a single vehicle. In the capacitated VRP, each customer has
a demand, each vehicle has a capacity, and the objective is usually to minimize
the total travel distance while ensuring that every customer is served exactly
once.

The drone delivery problem is closely related to VRP because:

- drones correspond to vehicles,
- customers correspond to delivery points,
- package demands correspond to customer demands,
- payload capacities correspond to vehicle capacities,
- each route starts and ends at the depot,
- each customer must be visited exactly once.

However, the drone delivery problem is more constrained than the classical VRP.
It includes additional drone-specific restrictions:

- battery capacity limits the total energy consumed by each route,
- energy depends on both distance and carried payload,
- no-fly zones may remove or penalize certain movements,
- optional collision avoidance may couple the routes of different drones in
  time and space.

Thus, the drone delivery problem can be viewed as an enriched VRP with
energy-aware and airspace-aware constraints. Because classical VRP is NP-hard,
and the drone problem contains VRP-like decisions as a core component, the
drone problem is also computationally difficult.

## 4. NP-Hardness Proof Intuition

To show that the drone delivery problem is NP-hard, it is sufficient to show
that a known NP-hard problem can be reduced to it in polynomial time. The TSP
provides a direct reduction.

Given any TSP instance with one depot and \(n\) customers, construct a drone
delivery instance as follows:

1. Use the same depot and customer coordinates.
2. Use exactly one drone.
3. Set the drone payload capacity large enough to carry all packages.
4. Set the drone battery capacity large enough to complete any possible tour.
5. Set all customer demands to a small positive value.
6. Remove all no-fly zones.
7. Use an energy model where energy is proportional to distance.

In this constructed instance, every feasible drone solution corresponds to a
TSP tour, and every TSP tour corresponds to a feasible drone solution. The
minimum-energy drone route is therefore the same as the shortest TSP tour.

If the drone delivery problem could be solved in polynomial time, then TSP
could also be solved in polynomial time by applying the above transformation.
Since TSP is NP-hard, the drone delivery problem is NP-hard.

This proof intuition is important because it uses a simplified version of the
drone problem. Adding payload limits, battery constraints, no-fly zones, and
multiple drones does not make the problem easier; these features only add more
combinatorial decisions and feasibility restrictions.

## 5. Size of the Search Space

The number of possible solutions grows very quickly with the number of
customers and drones. If there are \(n\) customers and \(m\) drones, a solution
must decide:

- which customers are assigned to each drone,
- the order of customers within each drone route,
- whether each route satisfies payload and battery constraints,
- whether route segments avoid no-fly zones.

Ignoring feasibility constraints, each customer can be assigned to one of
\(m\) drones, which gives approximately:

\[
m^n
\]

possible customer-to-drone assignments. For each assignment, the customers
assigned to a drone can be ordered in many ways. In the extreme case with one
drone, there are:

\[
n!
\]

possible customer sequences. With multiple drones, the number of route
partitions and permutations remains exponential or factorial in nature.

This growth explains why brute-force enumeration is impractical. Even moderate
instances become too large to evaluate exhaustively.

## 6. Complexity of Exact Methods

Exact methods, such as Branch and Bound, Branch and Cut, dynamic programming,
or MILP solvers, guarantee optimality if allowed to run to completion. However,
their worst-case complexity is exponential for this class of problems.

In a Branch and Bound method, the algorithm explores a search tree where each
branch represents a partial routing or assignment decision. The theoretical
worst-case number of nodes in the tree can grow exponentially with the number
of customers. In the worst case, the method may need to examine a large
fraction of all possible route assignments and customer permutations before
proving optimality.

For a MILP formulation, the number of binary variables also grows quickly. In a
node-arc formulation with:

- \(n\) customers,
- \(m\) drones,
- \(O(n^2)\) possible arcs,

the routing variables alone have approximate size:

\[
O(mn^2).
\]

Additional variables are required for customer assignment, payload flow,
battery modeling, subtour elimination, and optional time-based collision
avoidance. The number of constraints also grows polynomially in the model size,
but solving the resulting integer program remains exponential in the worst
case because of the binary routing decisions.

Exact methods are therefore appropriate for:

- small benchmark instances,
- validating the quality of heuristic solutions,
- producing optimal reference values,
- studying the structure of the problem.

They are usually not suitable as the only solution approach for large-scale
instances.

## 7. Effect of Drone-Specific Constraints on Complexity

The drone delivery problem is harder than a basic distance-minimizing routing
problem because the feasibility of a route depends on multiple resources.

### 7.1 Payload Capacity

Payload capacity restricts which customers can be grouped together on the same
route. This creates a bin-packing-like component: customers must be partitioned
among drones so that each route respects the drone's capacity.

### 7.2 Battery Capacity

Battery capacity imposes a route-length and energy limit. Unlike ordinary
distance limits, energy may depend on carried payload. Therefore, the same
route distance can have different energy values depending on the order of
deliveries, because early arcs may carry more weight than later arcs.

### 7.3 No-Fly Zones

No-fly zones restrict the feasible movement graph. Some direct arcs between
customers may become forbidden, and alternative routes may be required. This
can reduce feasibility and increase the difficulty of finding connected routes
that serve all customers.

### 7.4 Collision Avoidance

If collision avoidance is included, the routes of different drones are no
longer independent after customer assignment. The model must consider timing,
spatial separation, and possible conflicts between drones. This adds a
scheduling dimension to the routing problem and significantly increases
computational difficulty.

## 8. Scalability Analysis

Scalability refers to how the computational effort increases as the problem
size grows. For this project, the main scalability factors are:

- number of customers,
- number of drones,
- density and size of no-fly zones,
- tightness of payload and battery capacities,
- complexity of the energy model,
- inclusion or exclusion of collision avoidance.

As the number of customers increases, the number of possible customer
permutations grows factorially. As the number of drones increases, the number
of possible customer partitions also grows. Tight payload and battery
constraints may reduce the feasible search space, but they also make it harder
to construct valid routes. No-fly zones can disconnect parts of the movement
graph, which may force longer detours or make some assignments infeasible.

In practice, exact methods may solve small instances with a limited number of
customers, but their runtime can increase sharply as instance size grows.
Metaheuristics do not guarantee optimality, but they can explore large search
spaces more quickly and produce good feasible solutions within limited
runtime.

## 9. Why Heuristics and Metaheuristics Are Needed

Because the drone delivery problem is NP-hard, no polynomial-time algorithm is
known for solving all instances optimally. Exact methods may become too slow
for realistic or large benchmark instances. This motivates the use of
heuristics and metaheuristics.

Heuristics are useful because they can rapidly construct feasible solutions.
For example, a greedy insertion heuristic can assign customers to drones while
respecting payload and battery limits. However, simple heuristics may become
trapped in poor-quality solutions because they make locally attractive
decisions.

Metaheuristics provide a more flexible search framework. In this project:

- A Genetic Algorithm explores multiple solutions at the same time using
  selection, crossover, mutation, and repair operators.
- Simulated Annealing explores neighboring solutions and can temporarily accept
  worse moves to escape local optima.

These methods are appropriate because they can be adapted to the structure of
drone routing. Custom encodings, neighborhood moves, crossover operators, and
repair mechanisms can enforce or restore feasibility with respect to payload,
battery, and customer coverage constraints.

Metaheuristics are especially important for the experimental study because
they allow comparison between:

- solution quality,
- runtime,
- scalability,
- robustness across random benchmark instances.

## 10. Conclusion

The Drone-Based Delivery Optimization Problem is NP-hard because it contains
TSP as a special case and is closely related to the capacitated VRP. Its
complexity is increased by drone-specific constraints such as limited battery
capacity, payload-dependent energy consumption, no-fly zones, and optional
collision avoidance.

Exact methods are valuable for small instances and for producing optimal
benchmarks, but their exponential worst-case behavior limits their scalability.
For larger instances, heuristic and metaheuristic approaches are necessary to
obtain high-quality feasible solutions within reasonable computational time.
