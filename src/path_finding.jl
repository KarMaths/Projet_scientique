#=

AUTEUR: JEAN Karl Philippe 
    SCIENTIFIC COMPUTER SCIENCE PROJECT
    Software solution developed in Julia
    Implementation of pathfinding algorithms
        1- Dijkstra
        2- 
            2.1- A*
            2.2- WA*      version of A*: bounded suboptimal search  and various algorithms designed to find near-optimal paths efficiently
        3- BFS: Breadth-First-Search (recherche en largeur; Flood Fill)
        4- Glouton: Greedy Best-First-Search
=# 

# Define a Node structure with a tuple for position
#       Fields:-
#           A position (x, y) (Tuple of integers) and 
#           a dictionary (neighbors) associating positions (x, y) with weights (Float64).
#       Output: A node in the graph with its neighbors and the associated movement costs. 
#       Complexity: O(1)
mutable struct Node
    position::Tuple{Int64, Int64}
    neighbors::Dict{Tuple{Int64, Int64}, Float64}  # Neighbors and edge weights
end

# Define a Graph structure with a dictionnary of node 
#       Fields:-
#           nodes : A dictionary (nodes) associating positions (x, y) with (Node) objects. 
#       Output: A graph represented as a dictionary.
#       Complexity: O(1)
mutable struct Graph 
    nodes::Dict{Tuple{Int64, Int64}, Node}  
end 

# File of priority
#       A list of tuples (key, priority).
#       Output: Empty priority queue
#       Complexity: O(1)
mutable struct priority_queue 
    elements::Vector{Tuple{Tuple{Int64, Int64}, Float64}}
    # Constructor 
    function priority_queue()
        new([])
    end
end 

# function add to queue ( Adding an element to the priority queue )
#       Input: pq (priority_queue), a key (x, y), and a priority (Float64). 
#       Output: Adds an element and adjusts the structure.
#       Complexity: O(log n), , n = |pq| 
function add_to_queue!(pq::priority_queue, key::Tuple{Int64, Int64}, new_priority::Float64)
    push!(pq.elements, (key, new_priority))
    rebalancing_up!(pq, length(pq.elements)) # reequilibrer vers le haut 
end 

# Checks if the queue is empty  
#       Outputs: Boolean (True if empty, False otherwise)
#       Complexity: O(1)
function is_empty(pq::priority_queue)
    return length(pq.elements) == 0
end

# Removes the highest-priority element
#       Input: pq: priority_queue
#       Outputs: smallest value
#       Complexity: O(log n), n = |pq|
function remove_to_queue!(pq)
    if !is_empty(pq)
        change!(pq.elements, 1, length(pq.elements))
        min_element = pop!(pq.elements) # supprime 
        rebalancing_down!(pq, 1)
    end
    return min_element[1]
end

# Updates the priority of an element
#       Outputs: Updated priority queue with new priority
#       Complexity: O(log n), n = |pq|
function update_priority!(pq::priority_queue, key::Tuple{Int64, Int64}, new_priority::Float64)
    for i in eachindex(pq.elements)
        if pq.elements[i][1] == key
            if new_priority < pq.elements[i][2]
                pq.elements[i] = (key, new_priority)
                rebalancing_up!(pq, i)
            elseif new_priority > pq.elements[i][2]
                pq.elements[i] = (key, new_priority)
                rebalancing_down!(pq, i)
            end
            return
        end
    end
    add_to_queue!(pq, key, new_priority)
end

# function Rebalancing upwards
#       Inputs: pq (priority_queue), an index (Int).   
#       Outputs: Reorganizes the queue to maintain the order.  
#       Complexity: O(log n).  
function rebalancing_up!(pq, index)
    while index > 1
        parent = div(index, 2)
        if pq.elements[index][2] < pq.elements[parent][2]
            change!(pq.elements, index, parent)
            index = parent
        else
            break
        end
    end
end

# function Rebalancing downwards
#       Inputs: A (priority_queue), an index (Int).   
#       Outputs: Reorganizes the queue to maintain the order.    
#       Complexity: O(log n).  
function rebalancing_down!(pq, index)
    n = length(pq.elements)
    while 2 * index <= n
        left = 2 * index
        right = left + 1
        smallest = left

        if right <= n && pq.elements[right][2] < pq.elements[left][2]
            smallest = right
        end
        if pq.elements[index][2] <= pq.elements[smallest][2]
            break
        end
        change!(pq.elements, index, smallest)
        index = smallest
    end
end

# Swap two elements in a list
#       Inputs: A (Vector), two indices (i) and (j) Int64.   
#       Outputs: Modifies the list by swapping the values.   
#       Complexity: O(1).
function change!(vec, i, j)
    tmp = vec[i]
    vec[i] = vec[j]
    vec[j] = tmp 

end

# Function cost movement
#       Inputs: (value: Char) (Terrain type)
#       Outputs: Float64 (Cost of movement)
#       Complexity: O(1) 
function movement_cost(value)
    if value == '@' 
        return Inf
    elseif value == 'W' 
        return 8.0
    elseif value =='S'
        return 5.0
    elseif value =='T'
        return 10.0
    else
        return 1.0
    end 
end 

# Function to read graph from a map file
#       Inputs: A file name (String).   
#       Outputs: Returns a Graph of node.   
#       Complexity: O(n * m) (where n and m are the map dimensions).
function read_graph_from_file(filename)
    graph = Graph(Dict())
    grid = []
    # Read the file to extract the information from the map 
    open(filename, "r") do file
        # Ignore the first 4 lines 
        for i in 1:4
            readline(file)  # Read and ignore each line from header 
        end
        
        # Read the map file and store it in a 2D array
        for line in eachline(file)
            push!(grid, collect(line))
        end
    end

    height = length(grid)
    width = length(grid[1])

    # Now create the nodes and the graph with neighbors
    for x in 1:width 
        for y in 1:height
            # Create a node for each position
            position = (x,y)
            neighborhood = Dict{Tuple{Int64, Int64}, Float64}()

            # Check the 4 possible directions (up, down, left, right)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

            for (dx,dy) in directions 
                (nx,ny)=(x + dx, y + dy)
                if nx>=1 && nx<=width && ny>=1 && ny<=height
                    neighbor_position=(nx,ny)
                    cost = movement_cost(grid[nx][ny])
                    # Crucial verification: do not add a neighbor if it is a wall
                    if cost != Inf
                        neighborhood[neighbor_position] = cost
                    end
                end
            end

            # Add the node to the graph
            graph.nodes[position] = Node(position, neighborhood)
        end 
    end

    # Return the graph and its dimension
    return graph 
end 

# Dijsktra's algorithm
#       Inputs: A Graph, a source postion(x,y), a target postion(x,y).  
#       Outputs: A (previous) dictionary for paths, the distance (distance[target]), number of evaluated states.  
#       Complexity: O((n + m) log n) with a priority queue. 
function dijkstra(graph::Graph, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    distance = Dict{Tuple{Int64, Int64}, Float64}()
    previous = Dict{Tuple{Int64, Int64}, Union{Nothing, Tuple{Int64, Int64}}}()
    visited_node = Set{Tuple{Int64, Int64}}()
    evaluated_states = 0

    for node in keys(graph.nodes) 
        distance[node] = Inf 
        previous[node] = nothing 
        
    end
    distance[source]=0

    # Create a priority queue
    Q = priority_queue()
    add_to_queue!(Q, source, distance[source])

    while !is_empty(Q) 
        u = remove_to_queue!(Q)
        
        if u in visited_node
            continue
        end
        
        push!(visited_node, u)
        evaluated_states += 1

        # If the summit u is the target vertex, we stop the algorithm
        if u == target
            break
        end

        for (v, weight) in graph.nodes[u].neighbors
            alt= distance[u] + weight
            if distance[v] > alt
                distance[v] = alt
                previous[v] = u
                update_priority!(Q, v, alt)
            end 
            
        end
    end

    return previous, distance[target], evaluated_states
end 

# Breadth-First Search (BFS) Algorithm
#       Inputs: A Graph, a source postion(x,y), a target postion(x,y).    
#       Outputs: Minimum distance in number of steps, (previous) path, number of evaluated states.   
#       Complexity: O(n + m).
function Breadth_First_Search(graph::Graph, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    shortest_path = Dict{Tuple{Int64, Int64}, Float64}()
    previous = Dict{Tuple{Int64, Int64}, Union{Nothing, Tuple{Int64, Int64}}}()
    evaluated_states = 0
    mark = Dict{Tuple{Int64, Int64}, Bool}()
    

    for node in keys(graph.nodes)
        shortest_path[node] = Inf # unvisited node (summit) 
        previous[node] = nothing
        mark[node] = false
    end
    shortest_path[source] = 0

    tail = [source]
    mark[source] = true  
    
    while !isempty(tail)
        current = popfirst!(tail)
        evaluated_states += 1
        
        if current == target
            break
        end

        for neighbor in keys(graph.nodes[current].neighbors)
            if !mark[neighbor] 
                shortest_path[neighbor] = shortest_path[current] + 1
                previous[neighbor] = current
                mark[neighbor] = true  
                push!(tail, neighbor) 
            end
        end
    end

    return shortest_path, previous , evaluated_states 
end

# Heuristic function (Manhattan distance)
#       Inputs: Two tuples (x, y).   
#       Outputs: Heuristic distance (Int).   
#       Complexity: O(1).  
function heuristic_cost(source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    dx = abs(target[1] - source[1])
    dy = abs(target[2] - source[2])
    return  (dx + dy)
end

# A* Algorithm
#       Inputs: A Graph, a source postion(x,y), a target postion(x,y).  
#       Outputs: (g[target]) distance, (previous) dictionary, number of evaluated states.  
#       Complexity: O((n + m) log n)
function a_star(graph::Graph, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    g = Dict{Tuple{Int64, Int64}, Float64}()  # Cost from start to current node
    f = Dict{Tuple{Int64, Int64}, Float64}()  # Estimated cost (g + heuristic)
    previous = Dict{Tuple{Int64, Int64}, Union{Nothing, Tuple{Int64, Int64}}}()
    evaluated_states = 0
    visited = Set{Tuple{Int64, Int64}}()

    for node in keys(graph.nodes)
        g[node] = Inf
        f[node] = Inf
        previous[node] = nothing
    end
    g[source] = 0
    f[source] = heuristic_cost(source, target)

    Q = priority_queue()
    add_to_queue!(Q, source, f[source]) 

    while !is_empty(Q)
        current = remove_to_queue!(Q)

        if current in visited
            continue
        end
        push!(visited, current)
        evaluated_states += 1

        if current == target
            break
        end

        for (neighbor, cost) in graph.nodes[current].neighbors
            tentative_g = g[current] + cost

            if tentative_g < g[neighbor]
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + heuristic_cost(neighbor, target)
                previous[neighbor] = current
                update_priority!(Q, neighbor, f[neighbor]) 
            end
        end
    end
    return g, previous, evaluated_states
end

# WA*
#       Inputs: A Graph, a source node, a target node, a weighting factor w.  
#       Outputs: (g[target]) distance, (previous) dictionary, number of evaluated states.  
#       Complexity: O((n + m) log n)
function bounded_suboptimal_search(graph::Graph, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64}, w::Float64)
    g = Dict{Tuple{Int64, Int64}, Float64}()
    f = Dict{Tuple{Int64, Int64}, Float64}()
    previous = Dict{Tuple{Int64, Int64}, Union{Nothing, Tuple{Int64, Int64}}}()
    evaluated_states = 0
    visited = Set{Tuple{Int64, Int64}}()

    for node in keys(graph.nodes)
        g[node] = Inf
        f[node] = Inf
        previous[node] = nothing
    end

    g[source] = 0
    f[source] = g[source] + w * heuristic_cost(source, target)

    Q = priority_queue()
    add_to_queue!(Q, source, f[source])

    while !is_empty(Q)
        current = remove_to_queue!(Q)

        if current in visited
            continue
        end

        push!(visited, current)
        evaluated_states += 1

        if current == target
            break
        end

        for (neighbor, cost) in graph.nodes[current].neighbors
            tentative_g = g[current] + cost
            if tentative_g < g[neighbor]
                g[neighbor] = tentative_g
                f[neighbor] = g[neighbor] + w * heuristic_cost(neighbor, target)
                previous[neighbor] = current
                update_priority!(Q, neighbor, f[neighbor])
            end
        end
    end
    return g, previous, evaluated_states
end


# Greedy Best-First Search Algorithm
#       Inputs: A Graph, a source node, a target node.    
#       Outputs: Approximate distance (d[target]), (previous) dictionary, number of evaluated states.    
#       Complexity: O((n + m) log n).
function greedy_best_first_search(graph::Graph, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    previous = Dict{Tuple{Int64, Int64}, Union{Nothing, Tuple{Int64, Int64}}}()
    evaluated_states = 0
    visited = Set{Tuple{Int64, Int64}}()
    shortest_path = Dict{Tuple{Int64, Int64}, Float64}() 

    for node in keys(graph.nodes)
        previous[node] = nothing
        shortest_path[node] = Inf 
    end

    shortest_path[source] = 0

    # Priority queue with heuristic (Manhattan distance)
    Q = priority_queue()
    add_to_queue!(Q, source, Float64(heuristic_cost(source, target)))

    while !is_empty(Q)
        current = remove_to_queue!(Q)

        if current in visited
            continue
        end

        push!(visited, current)
        evaluated_states += 1

        if current == target
            break
        end

        for (neighbor, cost) in (graph.nodes[current].neighbors)
            if !(neighbor in visited)
                previous[neighbor] = current
                shortest_path[neighbor] = shortest_path[current] + cost
                update_priority!(Q, neighbor, Float64(heuristic_cost(neighbor, target)))
            end
        end
    end

    return shortest_path[target], previous, evaluated_states
end


# function displaying the shortest path
function print_shortest_path(source, target)
    path = []
    current = target 
    while current !== nothing
        push!(path, current)
        current = source[current]
    end

    for (i, p) in enumerate(reverse(path))
        print("$p")
        if i < length(path)
            print(" → ")
        end
    end
    println()
end 

# Main DIJKSTRA
function algo_dijkstra(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)

    start_time = time()
    previous, distance, count = dijkstra(graph, source, target)
    end_time = time()
    
    println("_______________DIJKSTRA______________")
    println("\nSolution :")
    println("CPUtime (s)                        :   ", round(end_time - start_time, digits=6))
    println("Distance ", source, " → ", target, "   :   ",(distance) )
    println("Number of states evaluated         :   ", count)
    print("Path ", source, " → ", target)
    println()
    print_shortest_path(previous, target)
    
end

#  Main BFS (Flood Fill)
function algo_bfs(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)

    start_time = time()
    distance, pred, visited_count = Breadth_First_Search(graph, source, target)
    end_time = time()

    println("_______________Breadth_First_Search______________")
    println("\nSolution :")
    println("CPUtime (s)                        :   ", round(end_time - start_time, digits=6))
    println("Distance ", source, " → ", target, "   :   ",(distance[target]))
    println("Number of states evaluated         :   ", visited_count)
    print("Path ", source, " → ", target)
    println()
    #print_shortest_path(pred, target)

end 

# Main A* 
function algo_a_star(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)
    start_time = time()
    g, previous, evaluated_states = a_star(graph, source, target)
    end_time = time()
    
    println("_______________A* Algorithm______________")
    println("\nSolution :")
    println("CPU time (s)                           :   ", round(end_time - start_time, digits=6))
    println("Distance ", source, " → ", target, "       :   ", g[target])
    println("Number of states evaluated             :   ", evaluated_states)
    print("Path ", source, " → ", target)
    println()
    #print_shortest_path(previous, target)
end

# Main BSS (WA*)
function algo_bss(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64}, w::Float64)
    graph = read_graph_from_file(filename)
    start_time = time()
    g, previous, evaluated_states = bounded_suboptimal_search(graph, source, target, w)
    end_time = time()

    println("_______________Bounded Suboptimal Search (BSS)______________")
    println("\nSolution :")
    println("CPU time (s)                           :   ", round(end_time - start_time, digits=6))
    println("Distance ", source, " → ", target, "       :   ", g[target])
    println("Number of states evaluated             :   ", evaluated_states)
    print("Path ", source, " → ", target)
    println()
    #print_shortest_path(previous, target)
end


# Main Greedy Best-First Search (Glouton)
function algo_greedy_bfs(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)
    
        start_time = time()
        d,previous, evaluated_states = greedy_best_first_search(graph, source, target)
        end_time = time()
    
    println("_______________Greedy Best-First Search______________")
    println("\nSolution :")
    println("CPU time (s)                           :   ", round(end_time - start_time, digits=6))
    println("Distance ", source, " → ", target, "       :   ", d)
    println("Number of states evaluated             :   ", evaluated_states)
    print("Path ", source, " → ", target)
    println()
    #print_shortest_path(previous, target)
end

# Appeler la fonction 
algo_dijkstra("theglaive.map", (189,193), (226,437))
algo_bfs("theglaive.map", (189,193), (226,437))
algo_a_star("theglaive.map", (189,193), (226,437))
algo_bss("theglaive.map", (189,193), (226,437), 1.5)
algo_greedy_bfs("theglaive.map", (189,193), (226,437))
