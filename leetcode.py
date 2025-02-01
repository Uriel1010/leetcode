class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        """
        881. Boats to Save People
        Medium
        You are given an array people where people[i] is the weight of the ith person, and an infinite number of boats where each boat can carry a maximum weight of limit. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most limit.
        Return the minimum number of boats to carry every given person.

        Example 1:

        Input: people = [1,2], limit = 3
        Output: 1
        Explanation: 1 boat (1, 2)
        Example 2:

        Input: people = [3,2,2,1], limit = 3
        Output: 3
        Explanation: 3 boats (1, 2), (2) and (3)
        Example 3:

        Input: people = [3,5,3,4], limit = 5
        Output: 4
        Explanation: 4 boats (3), (3), (4), (5)


        Constraints:

        1 <= people.length <= 5 * 104
        1 <= people[i] <= limit <= 3 * 104

        """
        people.sort()
        boats = 0
        left, right = 0, len(people) - 1

        while left <= right:
            if people[left] + people[right] <= limit:
                left += 1
            right -= 1
            boats += 1

        return boats

    def partitionString(self, s: str) -> int:
        """
        :param s:
        :return:
        Description: This function takes a string s as input and partitions the string
        into as many substrings as possible so that each letter in the substring appears
        only in that substring. It then returns the number of substrings created.

        Time complexity: The time complexity of this function is O(n), where n is the
        length of the input string. This is because the function iterates over each
        character in the input string only once.
        """
        l = []
        tmp = ''
        for char in s:
            if char not in tmp:
                tmp+=char
            else:
                l.append(tmp)
                tmp = char
        l.append(tmp)
        return len(l)

    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def prefixCount(self, words, pref):# List[str], pref: str) -> int:
        """
        Counts how many strings in the list 'words' start with the prefix 'pref'.
        
        :param words: A list of strings.
        :param pref: The prefix to check against each string in words.
        :return: The number of strings in 'words' that start with 'pref'.
        """
        return sum(word.startswith(pref) for word in words)

    def canConstruct(self, s, k):
        n = len(s)
        if k > n:
            return False      # cannot split into more parts than characters
        
        # Count frequency of each character
        freq = [0] * 26  # for 'a' to 'z'
        for char in s:
            freq[ord(char) - ord('a')] += 1

        # Count how many characters have odd frequency
        odd_count = sum(f % 2 for f in freq)

        # We need odd_count <= k <= n
        return odd_count <= k <= n

    def findThePrefixCommonArray(self, A, B):
        n = len(A)
        seenA = set()
        seenB = set()
        result = []

        for i in range(n):
            seenA.add(A[i])
            seenB.add(B[i])
            # Number of common elements so far is the size of the intersection
            result.append(len(seenA & seenB))
        
        return result

    def minimizeXor(self, num1, num2):
        # Function to count the number of set bits in a number
        def count_set_bits(x):
            count = 0
            while x:
                count += x & 1
                x >>= 1
            return count
        
        # Count the number of set bits in num2
        k = count_set_bits(num2)
        
        # Count the number of set bits in num1
        t = count_set_bits(num1)
        
        x = num1  # Initialize x with num1
        
        if t > k:
            # Need to turn off (t - k) set bits in x
            bits_to_turn_off = t - k
            for i in range(32):
                if x & (1 << i):
                    x ^= (1 << i)  # Flip the bit from 1 to 0
                    bits_to_turn_off -= 1
                    if bits_to_turn_off == 0:
                        break
        elif t < k:
            # Need to turn on (k - t) bits in x
            bits_to_turn_on = k - t
            for i in range(32):
                if not (x & (1 << i)):
                    x |= (1 << i)  # Flip the bit from 0 to 1
                    bits_to_turn_on -= 1
                    if bits_to_turn_on == 0:
                        break
        # If t == k, x remains num1, which already has the desired number of set bits
        
        return x

    def xorAllNums(self, nums1, nums2):
        # XOR of all elements in nums1
        xor1 = 0
        for x in nums1:
            xor1 ^= x
        
        # XOR of all elements in nums2
        xor2 = 0
        for x in nums2:
            xor2 ^= x
        
        # Final result
        result = 0
        
        # If nums2 has odd length, include xor1
        if len(nums2) % 2 == 1:
            result ^= xor1
        
        # If nums1 has odd length, include xor2
        if len(nums1) % 2 == 1:
            result ^= xor2
        
        return result

        def minCost(self, grid):
            m, n = len(grid), len(grid[0])
            INF = float('inf')
            
            # Distances array
            dist = [[INF]*n for _ in range(m)]
            dist[0][0] = 0
            
            # Directions keyed by the sign value:
            # 1 → (0, 1), 2 → (0, -1), 3 → (1, 0), 4 → (-1, 0)
            directions = {
                1: (0, 1),
                2: (0, -1),
                3: (1, 0),
                4: (-1, 0)
            }
            
            # Custom double-ended queue (0-1 BFS) without using imports
            class CustomDeque:
                def __init__(self, size):
                    # We allocate 2*size so we can pushLeft without re-allocating
                    self.q = [None]*(2*size)
                    # mid is the "middle" index, so we have space on both ends
                    self.mid = size
                    self.head = size  # front pointer
                    self.tail = size  # back pointer

                def pushLeft(self, item):
                    self.head -= 1
                    self.q[self.head] = item

                def pushRight(self, item):
                    self.q[self.tail] = item
                    self.tail += 1

                def popLeft(self):
                    item = self.q[self.head]
                    self.head += 1
                    return item

                def isEmpty(self):
                    return self.head == self.tail

            # Set up our custom deque
            dq = CustomDeque(m*n + 1)
            dq.pushRight((0, 0))

            # 0-1 BFS
            while not dq.isEmpty():
                r, c = dq.popLeft()
                if r == m - 1 and c == n - 1:
                    return dist[r][c]  # Found the bottom-right cell with minimal cost
                
                current_dist = dist[r][c]
                
                # Explore all 4 possible directions
                for sign, (dr, dc) in directions.items():
                    nr, nc = r + dr, c + dc
                    # Only proceed if inside the grid
                    if 0 <= nr < m and 0 <= nc < n:
                        # If the direction matches grid[r][c], cost = 0, else cost = 1
                        cost = 0 if sign == grid[r][c] else 1
                        new_dist = current_dist + cost
                        if new_dist < dist[nr][nc]:
                            dist[nr][nc] = new_dist
                            # 0 cost edges go to the front, 1 cost edges go to the back
                            if cost == 0:
                                dq.pushLeft((nr, nc))
                            else:
                                dq.pushRight((nr, nc))

            # If never reached (m-1, n-1) for some reason, return dist anyway
            return dist[m-1][n-1]

    def gridGame(self, grid):
        n = len(grid[0])
        
        # If there's only one column, Robot 2 can only collect 0
        # because Robot 1 will zero out the single cell (0,0) -> (1,0).
        if n == 1:
            return 0
        
        top = grid[0]
        bottom = grid[1]
        
        # Compute prefix sums
        # topPrefix[i] = sum of top[0..i-1], bottomPrefix[i] = sum of bottom[0..i-1]
        topPrefix = [0] * (n+1)
        bottomPrefix = [0] * (n+1)
        
        for i in range(n):
            topPrefix[i+1] = topPrefix[i] + top[i]
            bottomPrefix[i+1] = bottomPrefix[i] + bottom[i]
        
        # We'll try all possible 'c' where Robot 1 goes down
        # and pick the minimal max(...) for Robot 2's best response.
        ans = float('inf')
        
        for c in range(n):
            # Sum on top row AFTER column c: columns [c+1..n-1]
            # i.e. topPrefix[n] - topPrefix[c+1]
            score_top = topPrefix[n] - topPrefix[c+1] if c+1 <= n else 0
            
            # Sum on bottom row BEFORE column c: columns [0..c-1]
            # i.e. bottomPrefix[c]
            score_bottom = bottomPrefix[c]
            
            # Robot 2 picks the bigger segment
            second_robot_score = max(score_top, score_bottom)
            
            ans = min(ans, second_robot_score)
        
        return ans

    def eventualSafeNodes(self, graph):
        n = len(graph)
        color = [0] * n  # 0: unprocessed, 1: processing, 2: safe, 3: unsafe
        
        for i in range(n):
            if color[i] == 0:
                stack = [(i, False)]
                while stack:
                    node, processed = stack.pop()
                    if not processed:
                        if color[node] != 0:
                            continue
                        color[node] = 1
                        stack.append((node, True))
                        # Push neighbors in reverse order to maintain the order of processing
                        for neighbor in reversed(graph[node]):
                            if color[neighbor] == 0:
                                stack.append((neighbor, False))
                    else:
                        # Check if all neighbors are safe
                        safe = True
                        for neighbor in graph[node]:
                            if color[neighbor] != 2:
                                safe = False
                                break
                        if safe:
                            color[node] = 2
                        else:
                            color[node] = 3
        return [i for i in range(n) if color[i] == 2]        

    def lexicographicallySmallestArray(self, nums, limit):
        sorted_pairs = sorted((num, i) for i, num in enumerate(nums))
        n = len(nums)
        if n == 0:
            return []
        
        groups = []
        current_group = [sorted_pairs[0]]
        for i in range(1, n):
            if sorted_pairs[i][0] - sorted_pairs[i-1][0] <= limit:
                current_group.append(sorted_pairs[i])
            else:
                groups.append(current_group)
                current_group = [sorted_pairs[i]]
        groups.append(current_group)  # Add the last group
        
        res = [0] * n
        for group in groups:
            indices = [i for (val, i) in group]
            sorted_indices = sorted(indices)
            values = [val for (val, i) in group]
            for idx, val in zip(sorted_indices, values):
                res[idx] = val
        
        return res
    
    def maximumInvitations(self, favorite):
        n = len(favorite)
        
        # 1) Find all cycles and track the length of the longest cycle.
        #    We'll use a DFS approach with 'visited' states:
        #    0 = not visited, 1 = visiting (in stack), 2 = visited (done).
        visited = [0] * n
        # 'depth' will track the depth in DFS to calculate cycle length.
        depth = [0] * n
        max_cycle_len = 0

        def dfs(u, current_depth):
            nonlocal max_cycle_len
            visited[u] = 1
            depth[u] = current_depth
            v = favorite[u]
            
            if visited[v] == 0:
                dfs(v, current_depth + 1)
            elif visited[v] == 1:
                # A cycle is found: cycle length = current_depth - depth[v] + 1
                cycle_length = current_depth - depth[v] + 1
                max_cycle_len = max(max_cycle_len, cycle_length)
            # Mark as fully visited
            visited[u] = 2

        for i in range(n):
            if visited[i] == 0:
                dfs(i, 0)

        # 2) Calculate the chain lengths that lead into each node (for 2-cycles).
        #    We'll:
        #    - Build in-degree array.
        #    - Use a queue (Kahn's Algorithm style) for nodes with in-degree = 0.
        #    - chain_len[u] = length of the longest path ending at u (ignoring cycles).
        in_degree = [0] * n
        for i in range(n):
            in_degree[favorite[i]] += 1
        
        chain_len = [0] * n
        from collections import deque
        q = deque()

        # Enqueue nodes with in-degree 0
        for i in range(n):
            if in_degree[i] == 0:
                q.append(i)

        # Process all nodes with in-degree 0
        while q:
            u = q.popleft()
            v = favorite[u]
            # We can extend the chain of u by 1 to get to v
            chain_len[v] = max(chain_len[v], chain_len[u] + 1)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

        # 3) Sum up chain lengths for all 2-cycles:
        #    If i and j form a 2-cycle, we add chain_len[i] + chain_len[j] + 2
        #    to a running total. We then take the maximum between this sum
        #    (for all 2-cycles) and max_cycle_len.
        two_cycle_sum = 0
        visited_2cycle = [False] * n
        for i in range(n):
            j = favorite[i]
            # Check i < j to avoid double-counting the same pair
            if favorite[j] == i and i < j:
                # Mark as visited in 2-cycle so we don't double count
                visited_2cycle[i] = True
                visited_2cycle[j] = True
                two_cycle_sum += (chain_len[i] + chain_len[j] + 2)

        return max(max_cycle_len, two_cycle_sum)

    def checkIfPrerequisite(self, numCourses,
                            prerequisites,
                            queries):
        # Step 1: Initialize a reachability matrix
        reachable = [[False] * numCourses for _ in range(numCourses)]
        
        # Step 2: Mark direct prerequisites as reachable
        for pre, course in prerequisites:
            reachable[pre][course] = True
        
        # Step 3: Compute transitive closure using Floyd-Warshall style
        for k in range(numCourses):
            for i in range(numCourses):
                # If i->k isn't reachable, no need to proceed
                if not reachable[i][k]:
                    continue
                for j in range(numCourses):
                    # If k->j is reachable, then i->j is reachable
                    if reachable[k][j]:
                        reachable[i][j] = True
        
        # Step 4: Answer queries
        # If u -> v is reachable, then u is a prerequisite of v
        return [reachable[u][v] for u, v in queries]

    def findMaxFish(self, grid):
        m, n = len(grid), len(grid[0])
        visited = [[False]*n for _ in range(m)]
        
        def dfs(r, c):
            stack = [(r, c)]
            total = 0
            visited[r][c] = True
            while stack:
                x, y = stack.pop()
                total += grid[x][y]  # sum the fish in the current cell
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < m and 0 <= ny < n:
                        if not visited[nx][ny] and grid[nx][ny] > 0:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
            return total
        
        max_fish = 0
        for r in range(m):
            for c in range(n):
                if grid[r][c] > 0 and not visited[r][c]:
                    # Perform DFS on each unvisited water cell
                    max_fish = max(max_fish, dfs(r, c))
        
        return max_fish

    def findRedundantConnection(self, edges):
        # Union-Find helper functions
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # path compression
            return parent[x]
        
        def union(a, b):
            rootA = find(a)
            rootB = find(b)
            
            if rootA == rootB:
                return False  # a and b are already in the same set
            
            # union by rank
            if rank[rootA] > rank[rootB]:
                parent[rootB] = rootA
            elif rank[rootA] < rank[rootB]:
                parent[rootA] = rootB
            else:
                parent[rootB] = rootA
                rank[rootA] += 1
            
            return True

        n = len(edges)
        # Each node is initially its own parent
        parent = list(range(n+1))
        rank = [0] * (n+1)
        
        # Variable to store the last redundant edge
        last_cycle_edge = None
        
        # Process edges in the order they're given
        for u, v in edges:
            if not union(u, v): 
                # If union() returns False, u & v are already connected -> cycle
                last_cycle_edge = [u, v]
        
        return last_cycle_edge

    def magnificentSets(self, n, edges):
        # Build adjacency list (0-based)
        graph = [[] for _ in range(n)]
        for e in edges:
            u, v = e[0]-1, e[1]-1
            graph[u].append(v)
            graph[v].append(u)

        visited = [False] * n

        # 1) Gather each connected component with a simple DFS
        def get_component(start):
            stack = [start]
            comp = []
            visited[start] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in graph[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            return comp

        # 2) Check bipartite with 2-coloring
        #    Return False if the subgraph of comp_nodes isn't bipartite.
        def is_bipartite(comp_nodes):
            color = [-1]*n
            for nd in comp_nodes:
                if color[nd] == -1:  # not colored yet
                    color[nd] = 0
                    queue = [nd]  # BFS with a list
                    q_index = 0
                    while q_index < len(queue):
                        cur = queue[q_index]
                        q_index += 1
                        for nei in graph[cur]:
                            if nei in comp_set:  # only relevant inside this component
                                if color[nei] == -1:
                                    color[nei] = 1 - color[cur]
                                    queue.append(nei)
                                elif color[nei] == color[cur]:
                                    return False
            return True

        # 3) Attempt BFS layering from a given root.
        #    If valid layering, return max level; otherwise return -1.
        def bfs_layering(root):
            queue = [root]
            level = {root: 0}
            max_level = 0
            q_index = 0
            while q_index < len(queue):
                cur = queue[q_index]
                q_index += 1
                cur_lv = level[cur]
                if cur_lv > max_level:
                    max_level = cur_lv

                for nei in graph[cur]:
                    if nei in comp_set:  # edges within the component
                        if nei not in level:
                            level[nei] = cur_lv + 1
                            queue.append(nei)
                        # Check layering condition
                        if abs(level[nei] - level[cur]) != 1:
                            return -1  # invalid layering
            # If no edge violated the condition, layering is valid
            return max_level

        total_groups = 0

        # Process each connected component
        for i in range(n):
            if not visited[i]:
                comp_nodes = get_component(i)
                comp_set = set(comp_nodes)

                # Check bipartite
                if not is_bipartite(comp_nodes):
                    return -1

                best_for_this_comp = -1
                # Try BFS layering from *every node* in the component
                for node in comp_nodes:
                    max_lv = bfs_layering(node)
                    if max_lv >= 0 and max_lv > best_for_this_comp:
                        best_for_this_comp = max_lv

                # If no valid BFS layering from any node, return -1
                if best_for_this_comp == -1:
                    return -1

                # best_for_this_comp + 1 is the number of groups
                total_groups += (best_for_this_comp + 1)

        return total_groups

    def largestIsland(self, grid):
        n = len(grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        island_id = 2
        sizes = {}
        
        # Mark each island with a unique ID and calculate its size
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    stack = [(i, j)]
                    grid[i][j] = island_id
                    size = 0
                    while stack:
                        x, y = stack.pop()
                        size += 1
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                                grid[nx][ny] = island_id
                                stack.append((nx, ny))
                    sizes[island_id] = size
                    island_id += 1
        
        max_size = max(sizes.values()) if sizes else 0
        
        # Check each 0 to find the maximum possible island size after flipping
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    adjacent = set()
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] >= 2:
                            adjacent.add(grid[ni][nj])
                    current = 1 + sum(sizes[id] for id in adjacent)
                    if current > max_size:
                        max_size = current
        
        return max_size

    def isArraySpecial(self, nums):
        # Iterate through each adjacent pair
        for i in range(len(nums) - 1):
            # Check if both numbers have the same parity
            if nums[i] % 2 == nums[i + 1] % 2:
                return False
        return True

if __name__=="__main__":
    pass