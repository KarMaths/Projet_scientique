using DataStructures
using Printf

# Define a Node structure with a tuple for position
mutable struct Node
    position::Tuple{Int64, Int64} 
    neighbors::Dict{Tuple{Int64, Int64}, Float64}  # Neighbors and edge weights
    # Node(position) = new(position, Dict{Tuple{Int64, Int64}, Float64}())  # Constructeur par défaut
end

# Define a Graph structure with a dictionnary of node 
mutable struct Graph 
    nodes::Dict{Tuple{Int64, Int64}, Node}
end 

# Function cost movement 
function movement_cost(value::Char)
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

#Function to read graph from a map file
function read_graph_from_file(filename::String)
    
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
    for y in 1:height
        for x in 1:width

            # Create a node for each position
            position = (x,y)
            neighborhood = Dict{Tuple{Int64, Int64}, Float64}()

            # Check the 4 possible directions (up, down, left, right)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

            for (dx,dy) in directions 
                (nx,ny)=(x + dx, y + dy)
                if nx>=1 && nx<=width && ny>=1 && ny<=height 
                    neighbor_position=(nx,ny)
                    cost = movement_cost(grid[ny][nx])
                    neighborhood[neighbor_position] = cost 
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
    Q = PriorityQueue{Tuple{Int64, Int64}, Float64}()
    enqueue!(Q, source, distance[source])

    while !isempty(Q)
        u = dequeue!(Q)

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
            
                if v in keys(Q)
                    Q[v] = alt  # Mettre à jour la priorité si déjà dans la file
                else
                    enqueue!(Q, v, alt)  # Sinon ajouter à la file
                end
            end 
        end
    end

    return previous, distance[target], evaluated_states
end 

#Breadth-First Search (BFS) Algorithm
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
    mark[source] = true  # Marquer la source comme visitée dès le début
    #=
    tail = PriorityQueue{Tuple{Int64, Int64}, Float64}()
    enqueue!(tail, source, shortest_path[source])
    =#
    
    while !isempty(tail)
        current = popfirst!(tail)
        #current = dequeue!(tail)
        evaluated_states += 1
        
        # Si on a trouvé la cible, on arrête
        if current == target
            break
        end

        for neighbor in keys(graph.nodes[current].neighbors)
            if !mark[neighbor] 
                shortest_path[neighbor] = shortest_path[current] + 1
                previous[neighbor] = current
                mark[neighbor] = true  # Marquer le voisin comme visité dès qu'on l'ajoute
                push!(tail, neighbor) 
                #enqueue!(tail,neighbor, shortest_path[neighbor])
            end
        end
    end

    return shortest_path, previous , evaluated_states
    
end

# Heuristic function (Manhattan distance)
function heuristic_cost(source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    return abs(source[1] - target[1]) + abs(source[2] - target[2])
end

# A* Algorithm
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

    # Use PriorityQueue from DataStructures with f-values as priorities
    Q = PriorityQueue{Tuple{Int64, Int64}, Float64}()
    enqueue!(Q, source, f[source])

    while !isempty(Q)
        current = dequeue!(Q)

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
                
                # Add or update neighbor in the priority queue
                if neighbor in keys(Q)
                    Q[neighbor] = f[neighbor]  # Update priority
                else
                    enqueue!(Q, neighbor, f[neighbor])
                end
            end
        end
    end

    return g, previous, evaluated_states
end

# Greedy Best-First Search Algorithm
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
    Q = PriorityQueue{Tuple{Int64, Int64}, Float64}()
    enqueue!(Q, source, heuristic_cost(source, target))

    while !isempty(Q)
        current = dequeue!(Q)

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
                # Add or update neighbor in the priority queue
                if neighbor in keys(Q)
                    Q[neighbor] = heuristic_cost(neighbor, target)  # Update priority
                else
                    enqueue!(Q, neighbor, heuristic_cost(neighbor, target))
                end
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


# principal function DIJKSTRA
function algo_dijkstra(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)

    start_time = time()
    previous, distance, count = dijkstra(graph, source, target)
    end_time = time()

    println("_______________DIJKSTRA______________")
    println("\nSolution :")
    @printf("CPUtime (s)                :   %.1e\n", end_time - start_time)
    println("Distance ", source, " → ", target, " :   ",distance ) # get(distance, target, "Inf")
    println("Number of states evaluated :   ", count)#count(x -> x != Inf, values(distance))
    print("Path ", source, " → ", target)
    println()
    print_shortest_path(previous, target)
end

#  principal function BFS
function algo_bfs(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)

    start_time = time()
    distance, pred, visited_count = Breadth_First_Search(graph, source, target)
    end_time = time()

    println("_______________Breadth_First_Search______________")
    println("\nSolution :")
    @printf("CPUtime (s)                 :   %.1e\n", end_time - start_time)
    println("Distance ", source, " → ", target, " :   ", Int64(distance[target]))
    println("Number of states evaluated  :   ", visited_count)
    print("Path ", source, " → ", target)
    println()
    print_shortest_path(pred, target)

end 

# Main A* function
function algo_a_star(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)
    start_time = time()
    g, previous, evaluated_states = a_star(graph, source, target)
    end_time = time()
    
    println("_______________A* Algorithm______________")
    println("\nSolution :")
    @printf("CPU time (s)                  :   %.1e\n", end_time - start_time)
    println("Distance ", source, " → ", target, "   :   ", get(g, target, "Inf"))
    println("Number of states evaluated    :   ", evaluated_states)
    print("Path ", source, " → ", target)
    println()
    print_shortest_path(previous, target)
end

# Main function to execute thrGreedy Best-First Search algorithm
function algo_greedy_bfs(filename::String, source::Tuple{Int64, Int64}, target::Tuple{Int64, Int64})
    graph = read_graph_from_file(filename)

    start_time = time()
    d,previous, evaluated_states = greedy_best_first_search(graph, source, target)
    end_time = time()

    println("_______________Greedy Best-First Search______________")
    println("\nSolution :")
    @printf("CPU time (s)                  :   %.1e\n", round(end_time - start_time, sigdigits=5)) #round(time_value, sigdigits=5)
    println("Distance ", source, " → ", target, "   :   ", d)
    println("Number of states evaluated    :   ", evaluated_states)
    print("Path ", source, " → ", target)
    println()
    print_shortest_path(previous, target)
end

# Appeler la fonction 

#algo_dijkstra("didactic0.txt", (12,5), (2,12))
#algo_bfs("didactic0.txt", (12,5), (2,12))
#algo_a_star("didactic0.txt", (12,5), (2,12))
algo_greedy_bfs("didactic0.txt", (12,5), (2,12))

#algo_a_star("32room_004.map", (52,14), (2,20))
#algo_bfs("32room_004.map", (52,14), (2,20))
