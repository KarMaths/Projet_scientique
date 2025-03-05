# Projet_scientique

 Pathfinding Algorithms in Julia
 This repo is the code used in [my github page videos](https://github.com/KarMaths) who served as foundation of this repo.
 Many changes will be made after ...

# Description
    This project implements several pathfinding algorithms in Julia to find the shortest path between two points on a given map. 
    The algorithms included are:

        _Dijkstra's Algorithm
        _Breadth-First Search (BFS)
        _A* Algorithm
        _Greedy Best-First Search
    The map is read from a file, and movement costs are assigned based on different terrain types.

# Depencies 
    Make sure you have Julia installed and the following package:
        using DataStructures
        using Printf
        
    They can be installed by running 
        import Pkg
        Pkg.add(["DataStructures"]
        Pkg.add(["Printf"])
        
# Usage

1. Reading a Map File
    The program reads a map file in which different terrain types influence movement costs:

        @ (like water )
        T (for tree)
        . (normal walkable terrain)
        Other characters (default high cost)
2. Running the Algorithms
    The following functions execute the respective pathfinding algorithms:
    PS: The file "filename" can of the type (.map) it will also work
        algo_dijkstra("filename.txt", (x_init, y_init), (x_final, y_final))
        algo_bfs("filename.txt", (x_init, y_init), (x_final, y_final))
        algo_a_star("filename.txt", (x_init, y_init), (x_final, y_final))
        algo_greedy_bfs("filename.txt", (x_init, y_init), (x_final, y_final))
    # EXEMPLE 
        algo_dijkstra("didactic0.txt", (12,5), (2,12))

Algorithms Explained

1. Dijkstra's Algorithm
        Finds the shortest path from a source to all nodes using a priority queue.
2. Breadth-First Search (BFS)
        Explores neighbors level by level, ensuring the shortest unweighted path.
3. A* Algorithm**
        Uses a heuristic (Manhattan distance) to prioritize paths that seem closer to the goal.
4. Greedy Best-First Search
        Similar to A*, but only considers the heuristic for prioritization.
        
Output Example

    Each algorithm prints:

    Computation time

    Distance from source to target

    Number of states evaluated

    Path from source to target
