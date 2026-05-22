# Classical MILP Formulation

## 1. Modeling Perspective

This formulation represents the drone delivery problem at the level of
individual travel arcs between nodes. The depot and customers form a complete
directed graph, and the model decides which drone uses which arc. This is the
classical vehicle-routing abstraction: routes are not selected directly, but
are created implicitly by binary arc decisions and flow-conservation
constraints.

The formulation minimizes total energy consumption. Energy is modeled as a
function of both distance and carried payload, which is appropriate for drone
delivery because a drone consumes more energy when it flies with heavier
packages.

No-fly zones are handled by preprocessing the graph: any straight-line arc that
intersects a restricted zone is removed from the feasible arc set. This keeps
the optimization model linear while still enforcing airspace restrictions.

## 2. Sets

Let:

- \(C = \{1, 2, ..., n\}\): set of customers.
- \(0\): depot node.
- \(N = C \cup \{0\}\): set of all nodes.
- \(K = \{1, 2, ..., m\}\): set of drones.
- \(Z\): set of no-fly zones.
- \(A \subseteq N \times N\): set of feasible directed arcs.

An arc \((i,j)\) belongs to \(A\) only if:

- \(i \neq j\),
- the straight segment from node \(i\) to node \(j\) does not intersect any
  no-fly zone,
- the arc is physically allowed in the operating area.

## 3. Parameters

For each customer \(i \in C\):

- \(q_i\): package demand of customer \(i\).

For each drone \(k \in K\):

- \(Q_k\): payload capacity of drone \(k\).
- \(B_k\): battery capacity of drone \(k\).

For each feasible arc \((i,j) \in A\):

- \(d_{ij}\): Euclidean distance between node \(i\) and node \(j\).

Energy parameters:

- \(\alpha\): base energy consumption per distance unit for an empty drone.
- \(\beta\): additional energy consumption per distance and payload unit.

The energy consumed on arc \((i,j)\) by a drone carrying payload \(p\) is:

\[
E_{ij}(p) = d_{ij}(\alpha + \beta p).
\]

Large constant:

- \(M\): sufficiently large number used in logical constraints.

## 4. Decision Variables

Binary routing variable:

\[
x_{ijk} =
\begin{cases}
1, & \text{if drone } k \text{ travels directly from node } i \text{ to node } j,\\
0, & \text{otherwise.}
\end{cases}
\]

Customer assignment variable:

\[
y_{ik} =
\begin{cases}
1, & \text{if customer } i \text{ is served by drone } k,\\
0, & \text{otherwise.}
\end{cases}
\]

Drone usage variable:

\[
z_k =
\begin{cases}
1, & \text{if drone } k \text{ is used},\\
0, & \text{otherwise.}
\end{cases}
\]

Payload-flow variable:

\[
f_{ijk} \geq 0
\]

where \(f_{ijk}\) is the payload carried by drone \(k\) while flying on arc
\((i,j)\). This variable allows the payload-dependent energy expression to
remain linear.

Subtour-ordering variable:

\[
u_{ik} \geq 0
\]

where \(u_{ik}\) represents the visit order of customer \(i\) in the route of
drone \(k\).

## 5. Objective Function

The objective is to minimize total energy consumption over all drones and all
selected arcs:

\[
\min
\sum_{k \in K}
\sum_{(i,j) \in A}
\left(
\alpha d_{ij} x_{ijk}
+ \beta d_{ij} f_{ijk}
\right).
\]

The first term represents distance-based energy consumption of an empty drone.
The second term represents the additional energy caused by the payload carried
on each arc.

## 6. Constraints

### 6.1 Customer Service Constraint

Each customer must be served exactly once:

\[
\sum_{k \in K} y_{ik} = 1
\qquad \forall i \in C.
\]

This ensures that no customer is omitted and no customer is assigned to more
than one drone.

### 6.2 Route Continuity at Customer Nodes

If customer \(i\) is assigned to drone \(k\), the drone must enter and leave
that customer exactly once:

\[
\sum_{j \in N : (j,i) \in A} x_{jik} = y_{ik}
\qquad \forall i \in C,\ \forall k \in K,
\]

\[
\sum_{j \in N : (i,j) \in A} x_{ijk} = y_{ik}
\qquad \forall i \in C,\ \forall k \in K.
\]

These equations create continuous routes. A customer cannot be served without
an incoming and outgoing arc.

### 6.3 Depot Departure and Return

Each used drone must leave the depot once and return to the depot once:

\[
\sum_{j \in C : (0,j) \in A} x_{0jk} = z_k
\qquad \forall k \in K,
\]

\[
\sum_{i \in C : (i,0) \in A} x_{i0k} = z_k
\qquad \forall k \in K.
\]

If \(z_k = 0\), drone \(k\) is unused and no depot arc is selected. If
\(z_k = 1\), the route starts and ends at the depot.

### 6.4 Linking Assignment and Drone Usage

A drone can serve a customer only if it is used:

\[
y_{ik} \leq z_k
\qquad \forall i \in C,\ \forall k \in K.
\]

### 6.5 Payload Capacity Constraint

The total demand assigned to a drone cannot exceed its payload capacity:

\[
\sum_{i \in C} q_i y_{ik} \leq Q_k z_k
\qquad \forall k \in K.
\]

This constraint models the fact that the drone leaves the depot carrying all
packages for the customers on its route.

### 6.6 Payload Flow Conservation

The outgoing payload from the depot equals the total demand assigned to the
drone:

\[
\sum_{j \in C : (0,j) \in A} f_{0jk}
=
\sum_{i \in C} q_i y_{ik}
\qquad \forall k \in K.
\]

At every customer, the payload decreases by the delivered demand:

\[
\sum_{j \in N : (j,i) \in A} f_{jik}
-
\sum_{j \in N : (i,j) \in A} f_{ijk}
= q_i y_{ik}
\qquad \forall i \in C,\ \forall k \in K.
\]

The return arc to the depot carries no undelivered package:

\[
f_{i0k} = 0
\qquad \forall i \in C,\ \forall k \in K : (i,0) \in A.
\]

Payload can be positive only on selected arcs:

\[
0 \leq f_{ijk} \leq Q_k x_{ijk}
\qquad \forall (i,j) \in A,\ \forall k \in K.
\]

These constraints make the energy calculation route-dependent: early arcs tend
to carry more payload than later arcs.

### 6.7 Battery Capacity Constraint

The total energy consumed by each drone cannot exceed its battery capacity:

\[
\sum_{(i,j) \in A}
\left(
\alpha d_{ij} x_{ijk}
+ \beta d_{ij} f_{ijk}
\right)
\leq B_k z_k
\qquad \forall k \in K.
\]

This is the main drone-specific constraint distinguishing the problem from a
standard capacitated vehicle routing problem.

### 6.8 Subtour Elimination

For each drone, subtours among customers are eliminated using MTZ ordering
constraints:

\[
u_{ik} - u_{jk} + n x_{ijk} \leq n - 1
\qquad \forall i,j \in C,\ i \neq j,\ \forall k \in K.
\]

The order variables are active only for customers assigned to a drone:

\[
y_{ik} \leq u_{ik} \leq n y_{ik}
\qquad \forall i \in C,\ \forall k \in K.
\]

These constraints prevent disconnected cycles that do not include the depot.

### 6.9 No-Fly Zone Feasibility

Let \(a_{ij}\) be equal to 1 if the segment from node \(i\) to node \(j\) does
not intersect any no-fly zone, and 0 otherwise. The feasible arc set is:

\[
A = \{(i,j) \in N \times N : i \neq j,\ a_{ij}=1\}.
\]

The model defines \(x_{ijk}\) only for \((i,j) \in A\). Equivalently, for
forbidden arcs:

\[
x_{ijk} = 0
\qquad \forall (i,j) \notin A,\ \forall k \in K.
\]

This enforces no-fly restrictions without introducing nonlinear geometric
constraints into the optimization model.

## 7. Optional Collision-Avoidance Extension

If collision avoidance must be included, arrival-time variables can be added:

\[
t_{ik} \geq 0
\]

where \(t_{ik}\) is the arrival time of drone \(k\) at customer \(i\). For arcs
selected by drone \(k\):

\[
t_{jk} \geq t_{ik} + \tau_{ij} - M(1 - x_{ijk})
\qquad \forall (i,j) \in A,\ \forall k \in K,
\]

where \(\tau_{ij}\) is travel time. If two drones may occupy conflicting arcs
or nearby nodes, binary precedence variables can impose a minimum time
separation. This extension increases the model size significantly and is often
treated as optional in academic experiments.

## 8. Modeling Logic Summary

The MILP uses a detailed node-arc representation. The binary variables
\(x_{ijk}\) construct routes arc by arc, while \(y_{ik}\) assigns customers to
drones. Flow conservation ensures route continuity, depot constraints ensure
that each used drone starts and ends at the depot, and MTZ constraints remove
subtours. Payload-flow variables \(f_{ijk}\) make it possible to model
payload-dependent energy consumption linearly. Battery and payload capacities
then restrict which routes are feasible. No-fly zones are incorporated by
removing unsafe arcs before optimization.

This formulation is suitable for exact optimization on small and medium
instances, but it becomes computationally difficult as the number of customers,
drones, and feasible arcs increases.
