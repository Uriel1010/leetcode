# LeetCode Solutions in Python 🚀

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains **optimized Python solutions** for LeetCode problems, featuring detailed explanations and complexity analysis. Perfect for coding interview preparation and algorithm enthusiasts!

## 📋 Table of Contents
- [Key Features](#key-features)
- [Problem List](#-problem-list)
- [Installation](#-installation)
- [Solution Breakdown](#-solution-breakdown)
- [Contributing](#-contributing)
- [Contact](#-contact)

<a name="key-features"></a>
## ✨ Key Features
- 🧠 Clean, well-documented code with OOP approach
- ⚡ Time/space complexity analysis for each solution
- 🎯 Optimal algorithms using modern Python features
- 🔗 LeetCode problem links for quick reference
- 🌐 Covers arrays, strings, graphs, and bit manipulation problems

<a name="problem-list"></a>
## 🧩 Problem List

| #   | Function Name                       | LeetCode Problem                                              | Difficulty | Key Technique        | Link                                                                              |
|-----|-------------------------------------|---------------------------------------------------------------|------------|----------------------|-----------------------------------------------------------------------------------|
| 1   | `numRescueBoats`                   | **Boats to Save People** (#881)                              | Medium     | Two Pointers         | [881](https://leetcode.com/problems/boats-to-save-people/)                       |
| 2   | `partitionString`                  | **Optimal Partition of String** (#2405)                      | Medium     | Greedy               | [2405](https://leetcode.com/problems/optimal-partition-of-string/)               |
| 3   | `search`                           | **Binary Search** (#704)                                     | Easy       | Binary Search        | [704](https://leetcode.com/problems/binary-search/)                              |
| 4   | `prefixCount`                      | **Counting Words With a Given Prefix** (#2185)               | Easy       | String Manipulation  | [2185](https://leetcode.com/problems/counting-words-with-a-given-prefix/)        |
| 5   | `canConstruct`                     | **Construct K Palindrome Strings** (#1400)                   | Medium     | Character Counting   | [1400](https://leetcode.com/problems/construct-k-palindrome-strings/)            |
| 6   | `findThePrefixCommonArray`         | **Find the Prefix Common Array** (#2657)                     | Medium     | Set Operations       | [2657](https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/)|
| 7   | `minimizeXor`                      | **Minimize XOR** (#2429)                                     | Medium     | Bit Manipulation     | [2429](https://leetcode.com/problems/minimize-xor/)                              |
| 8   | `xorAllNums`                       | **Bitwise XOR of All Pairings** (#2575)                      | Medium     | XOR Properties       | [2575](https://leetcode.com/problems/bitwise-xor-of-all-pairings/)               |
| 9   | `minCost`                          | **Minimum Cost to Make at Least One Valid Path in a Grid** (#1368) | Hard  | 0-1 BFS + Graph      | [1368](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/) |
| 10  | `gridGame`                         | **Grid Game** (#2017)                                        | Medium     | Prefix Sums          | [2017](https://leetcode.com/problems/grid-game/)                                 |
| 11  | `eventualSafeNodes`                | **Find Eventual Safe Nodes** (#802)                          | Medium     | Graph DFS            | [802](https://leetcode.com/problems/find-eventual-safe-nodes/)                   |
| 12  | `lexicographicallySmallestArray`*  | *(Custom / Example Function)*                                | N/A        | Sorting + Grouping   | *(No official link)*                                                              |
| 13  | `maximumInvitations`               | **Maximum Employees to Be Invited to a Meeting** (#2127)     | Hard       | Graph + Cycles       | [2127](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/) |
| 14  | `checkIfPrerequisite`              | **Course Schedule IV** (#1462)                               | Medium     | Floyd-Warshall / Graph | [1462](https://leetcode.com/problems/course-schedule-iv/)                      |

> *`lexicographicallySmallestArray` is a custom demo function not tied to an official LeetCode problem number.

<a name="installation"></a>
## 🛠️ Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/leetcode-solutions.git
   cd leetcode-solutions
   ```
2. **Set Up a Virtual Environment (Optional)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   # or
   venv\Scripts\activate      # Windows
   ```
3. **Install Dependencies** (if any):
   ```bash
   pip install -r requirements.txt
   ```

<a name="solution-breakdown"></a>
## 🔍 Solution Breakdown

Below are brief overviews of selected solutions from the code. Full details can be found in the source files.

### 1. Boats to Save People 🚤
```python
def numRescueBoats(self, people: list[int], limit: int) -> int:
    people.sort()
    boats = 0
    left, right = 0, len(people) - 1
    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1
    return boats
```
**Approach**:  
- Sort the array and use two pointers (`left` and `right`).
- Pair the heaviest person (`right`) with the lightest person (`left`) if possible.
- If they can’t fit together, send the heaviest alone.
- In either case, move `right` leftward by one and increment `boats`.

**Complexity**:  
- Time: O(n log n) due to sorting.  
- Space: O(1) extra space.

---

### 2. Optimal Partition of String ✂️
```python
def partitionString(self, s: str) -> int:
    partitions = []
    current = ''
    for char in s:
        if char in current:
            partitions.append(current)
            current = char
        else:
            current += char
    partitions.append(current)
    return len(partitions)
```
**Idea**:  
- Greedily build a substring until a repeating character is found.
- Once a repeat occurs, start a new partition.

**Complexity**:  
- Time: O(n), where n is the length of `s`.  
- Space: O(n) for storing partitions in the worst case.

---

### 3. Binary Search 🔎
```python
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
```
**Approach**: Standard binary search.  
**Complexity**:
- Time: O(log n).  
- Space: O(1).

---

### 8. Bitwise XOR of All Pairings
```python
def xorAllNums(self, nums1: list[int], nums2: list[int]) -> int:
    xor1 = 0Below is the updated **readme.md** file that now reflects all 19 functions from your `leetcode.py` file. You can copy and paste the following content into your repository’s `readme.md`:

---

```markdown
# LeetCode Solutions in Python 🚀

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains **optimized Python solutions** for LeetCode problems, featuring detailed explanations and complexity analysis. Perfect for coding interview preparation and algorithm enthusiasts!

## 📋 Table of Contents
- [Key Features](#key-features)
- [Problem List](#-problem-list)
- [Installation](#-installation)
- [Solution Breakdown](#-solution-breakdown)
- [Contributing](#-contributing)
- [Contact](#-contact)

<a name="key-features"></a>
## ✨ Key Features
- 🧠 Clean, well-documented code using an OOP approach
- ⚡ Time/space complexity analysis for each solution
- 🎯 Optimal algorithms using modern Python features
- 🔗 LeetCode problem links for quick reference
- 🌐 Covers arrays, strings, graphs, grids, and bit manipulation problems

<a name="problem-list"></a>
## 🧩 Problem List

| #   | Function Name                      | LeetCode Problem                                                   | Difficulty | Key Technique             | Link                                                                                 |
|-----|------------------------------------|--------------------------------------------------------------------|------------|---------------------------|--------------------------------------------------------------------------------------|
| 1   | `numRescueBoats`                   | Boats to Save People (#881)                                        | Medium     | Two Pointers              | [881](https://leetcode.com/problems/boats-to-save-people/)                          |
| 2   | `partitionString`                  | Optimal Partition of String (#2405)                                | Medium     | Greedy                    | [2405](https://leetcode.com/problems/optimal-partition-of-string/)                   |
| 3   | `search`                           | Binary Search (#704)                                               | Easy       | Binary Search             | [704](https://leetcode.com/problems/binary-search/)                                  |
| 4   | `prefixCount`                      | Counting Words With a Given Prefix (#2185)                         | Easy       | String Manipulation       | [2185](https://leetcode.com/problems/counting-words-with-a-given-prefix/)            |
| 5   | `canConstruct`                     | Construct K Palindrome Strings (#1400)                             | Medium     | Character Counting        | [1400](https://leetcode.com/problems/construct-k-palindrome-strings/)                |
| 6   | `findThePrefixCommonArray`         | Find the Prefix Common Array (#2657)                               | Medium     | Set Operations            | [2657](https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/)     |
| 7   | `minimizeXor`                      | Minimize XOR (#2429)                                               | Medium     | Bit Manipulation          | [2429](https://leetcode.com/problems/minimize-xor/)                                  |
| 8   | `xorAllNums`                       | Bitwise XOR of All Pairings (#2575)                                | Medium     | XOR Properties            | [2575](https://leetcode.com/problems/bitwise-xor-of-all-pairings/)                   |
| 9   | `minCost`                          | Minimum Cost to Make at Least One Valid Path in a Grid (#1368)       | Hard       | 0-1 BFS + Graph           | [1368](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/) |
| 10  | `gridGame`                         | Grid Game (#2017)                                                  | Medium     | Prefix Sums               | [2017](https://leetcode.com/problems/grid-game/)                                     |
| 11  | `eventualSafeNodes`                | Find Eventual Safe Nodes (#802)                                    | Medium     | Graph DFS                 | [802](https://leetcode.com/problems/find-eventual-safe-nodes/)                       |
| 12  | `lexicographicallySmallestArray`*  | Custom / Example Function                                          | N/A        | Sorting + Grouping        | *(No official link)*                                                                 |
| 13  | `maximumInvitations`               | Maximum Employees to Be Invited to a Meeting (#2127)               | Hard       | Graph + Cycle Detection   | [2127](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/)   |
| 14  | `checkIfPrerequisite`              | Course Schedule IV (#1462)                                           | Medium     | Floyd-Warshall / Graph    | [1462](https://leetcode.com/problems/course-schedule-iv/)                            |
| 15  | `findMaxFish`                      | Find Maximum Fish (Custom)                                           | Medium     | DFS on Grid               | *(No official link)*                                                                 |
| 16  | `findRedundantConnection`          | Redundant Connection (#684)                                          | Medium     | Union-Find                | [684](https://leetcode.com/problems/redundant-connection/)                           |
| 17  | `magnificentSets`                  | Divide Nodes Into the Maximum Number of Groups (#2493)             | Hard       | Graph + BFS/DFS, Bipartite| [2493](https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/)  |
| 18  | `largestIsland`                    | Making A Large Island (#827)                                         | Hard       | DFS/Union-Find, Grid      | [827](https://leetcode.com/problems/making-a-large-island/)                          |
| 19  | `isArraySpecial`                   | Special Array I (#3151)                                              | Easy       | Greedy / Iteration        | [3151](https://leetcode.com/problems/special-array-i/)                               |

> *`lexicographicallySmallestArray` is a custom demo function not tied to an official LeetCode problem number.

<a name="installation"></a>
## 🛠️ Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/leetcode-solutions.git
   cd leetcode-solutions
   ```
2. **Set Up a Virtual Environment (Optional)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   # or
   venv\Scripts\activate      # Windows
   ```
3. **Install Dependencies** (if any):
   ```bash
   pip install -r requirements.txt
   ```

<a name="solution-breakdown"></a>
## 🔍 Solution Breakdown

Below are brief overviews of selected solutions from the code. Full details can be found in the source files.

### 1. Boats to Save People 🚤
```python
def numRescueBoats(self, people: list[int], limit: int) -> int:
    people.sort()
    boats = 0
    left, right = 0, len(people) - 1
    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1
    return boats
```
**Approach**:  
- Sort the array and use two pointers (`left` and `right`).
- Pair the heaviest person (`right`) with the lightest person (`left`) if possible.
- Otherwise, send the heaviest alone.
- Move `right` leftward and increment the boat count.

**Complexity**:  
- **Time**: O(n log n) (due to sorting)  
- **Space**: O(1) extra space.

---

### 2. Optimal Partition of String ✂️
```python
def partitionString(self, s: str) -> int:
    partitions = []
    current = ''
    for char in s:
        if char in current:
            partitions.append(current)
            current = char
        else:
            current += char
    partitions.append(current)
    return len(partitions)
```
**Idea**:  
- Greedily build substrings until a repeated character is found.
- Start a new partition upon repetition.

**Complexity**:  
- **Time**: O(n)  
- **Space**: O(n) in the worst case.

---

### 3. Binary Search 🔎
```python
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
```
**Approach**:  
- Standard binary search algorithm.

**Complexity**:
- **Time**: O(log n)  
- **Space**: O(1).

---

### 4. Counting Words With a Given Prefix 🏷
```python
def prefixCount(self, words, pref):
    return sum(word.startswith(pref) for word in words)
```
**Insight**:  
- Leverages Python’s `startswith` for clarity and conciseness.

**Complexity**:
- **Time**: O(n * p) where p is the prefix length  
- **Space**: O(1).

---

### 5. Construct K Palindrome Strings 🎭
```python
def canConstruct(self, s, k):
    n = len(s)
    if k > n:
        return False
    freq = [0] * 26
    for char in s:
        freq[ord(char) - ord('a')] += 1
    odd_count = sum(f % 2 for f in freq)
    return odd_count <= k <= n
```
**Approach**:  
- Count the frequency of each character.
- Ensure the number of odd counts does not exceed k.

**Complexity**:
- **Time**: O(n)  
- **Space**: O(1).

---

### 6. Find the Prefix Common Array 🔗
```python
def findThePrefixCommonArray(self, A, B):
    n = len(A)
    seenA, seenB = set(), set()
    result = []
    for i in range(n):
        seenA.add(A[i])
        seenB.add(B[i])
        result.append(len(seenA & seenB))
    return result
```
**Idea**:  
- Use sets to track seen elements and compute their intersection size at each step.

**Complexity**:
- **Time**: O(n)  
- **Space**: O(n).

---

### 7. Minimize XOR 🔧
```python
def minimizeXor(self, num1, num2):
    def count_set_bits(x):
        count = 0
        while x:
            count += x & 1
            x >>= 1
        return count
    
    k = count_set_bits(num2)
    t = count_set_bits(num1)
    x = num1
    
    if t > k:
        bits_to_turn_off = t - k
        for i in range(32):
            if x & (1 << i):
                x ^= (1 << i)
                bits_to_turn_off -= 1
                if bits_to_turn_off == 0:
                    break
    elif t < k:
        bits_to_turn_on = k - t
        for i in range(32):
            if not (x & (1 << i)):
                x |= (1 << i)
                bits_to_turn_on -= 1
                if bits_to_turn_on == 0:
                    break
    return x
```
**Insight**:  
- Adjust bits in `num1` to match the number of set bits in `num2` while minimizing the XOR difference.

**Complexity**:
- **Time**: O(1) (fixed 32 iterations)  
- **Space**: O(1).

---

### 8. Bitwise XOR of All Pairings ⊕
```python
def xorAllNums(self, nums1: list[int], nums2: list[int]) -> int:
    xor1 = 0
    for x in nums1:
        xor1 ^= x
    xor2 = 0
    for x in nums2:
        xor2 ^= x
    
    result = 0
    if len(nums2) % 2 == 1:
        result ^= xor1
    if len(nums1) % 2 == 1:
        result ^= xor2
    return result
```
**Approach**:  
- Uses XOR properties to compute the result without iterating through every pairing.

**Complexity**:
- **Time**: O(n + m)  
- **Space**: O(1).

---

### 9. Minimum Cost Path in Grid 🛤 (0-1 BFS)
```python
def minCost(self, grid):
    m, n = len(grid), len(grid[0])
    INF = float('inf')
    dist = [[INF]*n for _ in range(m)]
    dist[0][0] = 0
    directions = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    
    class CustomDeque:
        def __init__(self, size):
            self.q = [None]*(2*size)
            self.mid = size
            self.head = size
            self.tail = size

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

    dq = CustomDeque(m*n + 1)
    dq.pushRight((0, 0))

    while not dq.isEmpty():
        r, c = dq.popLeft()
        if r == m - 1 and c == n - 1:
            return dist[r][c]
        current_dist = dist[r][c]
        for sign, (dr, dc) in directions.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                cost = 0 if sign == grid[r][c] else 1
                new_dist = current_dist + cost
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    if cost == 0:
                        dq.pushLeft((nr, nc))
                    else:
                        dq.pushRight((nr, nc))
    return dist[m-1][n-1]
```
**Insight**:  
- Implements a custom 0-1 BFS to efficiently traverse a grid with edge weights 0 or 1.

**Complexity**:
- **Time**: O(m*n)  
- **Space**: O(m*n).

---

### 10. Grid Game 🎲
```python
def gridGame(self, grid):
    n = len(grid[0])
    if n == 1:
        return 0
    top, bottom = grid[0], grid[1]
    topPrefix = [0] * (n+1)
    bottomPrefix = [0] * (n+1)
    for i in range(n):
        topPrefix[i+1] = topPrefix[i] + top[i]
        bottomPrefix[i+1] = bottomPrefix[i] + bottom[i]
    
    ans = float('inf')
    for c in range(n):
        score_top = topPrefix[n] - topPrefix[c+1] if c+1 <= n else 0
        score_bottom = bottomPrefix[c]
        second_robot_score = max(score_top, score_bottom)
        ans = min(ans, second_robot_score)
    return ans
```
**Approach**:  
- Computes prefix sums for both rows and tests each splitting column.

**Complexity**:
- **Time**: O(n)  
- **Space**: O(n).

---

### 11. Find Eventual Safe Nodes 🛡
```python
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
                    for neighbor in reversed(graph[node]):
                        if color[neighbor] == 0:
                            stack.append((neighbor, False))
                else:
                    safe = True
                    for neighbor in graph[node]:
                        if color[neighbor] != 2:
                            safe = False
                            break
                    color[node] = 2 if safe else 3
    return [i for i in range(n) if color[i] == 2]
```
**Idea**:  
- Uses DFS with color-marking to mark nodes as safe or unsafe based on cycle detection.

**Complexity**:
- **Time**: O(n + e)  
- **Space**: O(n).

---

### 12. Lexicographically Smallest Array 🔤 (Custom)
```python
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
    groups.append(current_group)
    
    res = [0] * n
    for group in groups:
        indices = [i for (val, i) in group]
        sorted_indices = sorted(indices)
        values = [val for (val, i) in group]
        for idx, val in zip(sorted_indices, values):
            res[idx] = val
    return res
```
**Insight**:  
- Groups numbers based on differences and assigns them in a lexicographically smallest order.

**Complexity**:
- **Time**: O(n log n)  
- **Space**: O(n).

---

### 13. Maximum Invitations 📧
```python
def maximumInvitations(self, favorite):
    n = len(favorite)
    visited = [0] * n
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
            cycle_length = current_depth - depth[v] + 1
            max_cycle_len = max(max_cycle_len, cycle_length)
        visited[u] = 2

    for i in range(n):
        if visited[i] == 0:
            dfs(i, 0)

    in_degree = [0] * n
    for i in range(n):
        in_degree[favorite[i]] += 1
    
    chain_len = [0] * n
    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        v = favorite[u]
        chain_len[v] = max(chain_len[v], chain_len[u] + 1)
        in_degree[v] -= 1
        if in_degree[v] == 0:
            q.append(v)
    
    two_cycle_sum = 0
    for i in range(n):
        j = favorite[i]
        if favorite[j] == i and i < j:
            two_cycle_sum += (chain_len[i] + chain_len[j] + 2)
    
    return max(max_cycle_len, two_cycle_sum)
```
**Approach**:  
- Uses DFS to detect cycles and a BFS-style (Kahn's algorithm) method to compute chain lengths feeding into 2-cycles.

**Complexity**:
- **Time**: O(n)  
- **Space**: O(n).

---

### 14. Course Schedule IV 🎓
```python
def checkIfPrerequisite(self, numCourses, prerequisites, queries):
    reachable = [[False] * numCourses for _ in range(numCourses)]
    for pre, course in prerequisites:
        reachable[pre][course] = True
    for k in range(numCourses):
        for i in range(numCourses):
            if reachable[i][k]:
                for j in range(numCourses):
                    if reachable[k][j]:
                        reachable[i][j] = True
    return [reachable[u][v] for u, v in queries]
```
**Approach**:  
- Constructs a reachability matrix using a Floyd–Warshall–like algorithm.

**Complexity**:
- **Time**: O(n³ + q)  
- **Space**: O(n²).

---

### 15. Find Maximum Fish 🎣 (Custom)
```python
def findMaxFish(self, grid):
    m, n = len(grid), len(grid[0])
    visited = [[False]*n for _ in range(m)]
    
    def dfs(r, c):
        stack = [(r, c)]
        total = 0
        visited[r][c] = True
        while stack:
            x, y = stack.pop()
            total += grid[x][y]
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and grid[nx][ny] > 0:
                    visited[nx][ny] = True
                    stack.append((nx, ny))
        return total
    
    max_fish = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] > 0 and not visited[r][c]:
                max_fish = max(max_fish, dfs(r, c))
    return max_fish
```
**Insight**:  
- Uses DFS on a grid to find the connected component (island) with the maximum sum (fish count).

**Complexity**:
- **Time**: O(m*n)  
- **Space**: O(m*n).

---

### 16. Redundant Connection 🔗
```python
def findRedundantConnection(self, edges):
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        rootA = find(a)
        rootB = find(b)
        if rootA == rootB:
            return False
        if rank[rootA] > rank[rootB]:
            parent[rootB] = rootA
        elif rank[rootA] < rank[rootB]:
            parent[rootA] = rootB
        else:
            parent[rootB] = rootA
            rank[rootA] += 1
        return True

    n = len(edges)
    parent = list(range(n+1))
    rank = [0] * (n+1)
    last_cycle_edge = None
    for u, v in edges:
        if not union(u, v):
            last_cycle_edge = [u, v]
    return last_cycle_edge
```
**Approach**:  
- Uses union-find with path compression and union by rank to detect the edge that creates a cycle.

**Complexity**:
- **Time**: O(n * α(n))  
- **Space**: O(n).

---

### 17. Divide Nodes Into the Maximum Number of Groups 🌐
```python
def magnificentSets(self, n, edges):
    graph = [[] for _ in range(n)]
    for e in edges:
        u, v = e[0]-1, e[1]-1
        graph[u].append(v)
        graph[v].append(u)
    
    visited = [False] * n
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
    
    def is_bipartite(comp_nodes):
        color = [-1]*n
        for nd in comp_nodes:
            if color[nd] == -1:
                color[nd] = 0
                queue = [nd]
                q_index = 0
                while q_index < len(queue):
                    cur = queue[q_index]
                    q_index += 1
                    for nei in graph[cur]:
                        if nei in comp_set:
                            if color[nei] == -1:
                                color[nei] = 1 - color[cur]
                                queue.append(nei)
                            elif color[nei] == color[cur]:
                                return False
        return True

    def bfs_layering(root):
        queue = [root]
        level = {root: 0}
        max_level = 0
        q_index = 0
        while q_index < len(queue):
            cur = queue[q_index]
            q_index += 1
            cur_lv = level[cur]
            max_level = max(max_level, cur_lv)
            for nei in graph[cur]:
                if nei in comp_set:
                    if nei not in level:
                        level[nei] = cur_lv + 1
                        queue.append(nei)
                    if abs(level[nei] - level[cur]) != 1:
                        return -1
        return max_level

    total_groups = 0
    for i in range(n):
        if not visited[i]:
            comp_nodes = get_component(i)
            comp_set = set(comp_nodes)
            if not is_bipartite(comp_nodes):
                return -1
            best_for_this_comp = -1
            for node in comp_nodes:
                max_lv = bfs_layering(node)
                if max_lv >= 0 and max_lv > best_for_this_comp:
                    best_for_this_comp = max_lv
            if best_for_this_comp == -1:
                return -1
            total_groups += (best_for_this_comp + 1)
    return total_groups
```
**Insight**:  
- Divides nodes into groups using valid BFS layerings while ensuring each component is bipartite.

**Complexity**:
- **Time**: O(n*(n+e)) per component in the worst case  
- **Space**: O(n).

---

### 18. Making A Large Island 🌴
```python
def largestIsland(self, grid):
    n = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    island_id = 2
    sizes = {}
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
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                adjacent = set()
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] >= 2:
                        adjacent.add(grid[ni][nj])
                current = 1 + sum(sizes[id] for id in adjacent)
                max_size = max(max_size, current)
    return max_size
```
**Approach**:  
- Uses DFS to mark and measure islands, then checks each water cell to determine the maximum possible island size if flipped.

**Complexity**:
- **Time**: O(n²)  
- **Space**: O(n²).

---

### 19. Special Array I 🚩
```python
def isArraySpecial(self, nums):
    for i in range(len(nums) - 1):
        if nums[i] % 2 == nums[i + 1] % 2:
            return False
    return True
```
**Idea**:  
- Iterates over adjacent pairs to ensure that they have different parity.

**Complexity**:
- **Time**: O(n)  
- **Space**: O(1).

---

<a name="contributing"></a>
## 🤝 Contributing
1. **Fork** the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`).
4. **Push** to the branch (`git push origin feature/amazing-feature`).
5. **Open** a Pull Request.

<a name="contact"></a>
## 📬 Contact
**Author**: Uriel Manzur  
**Email**: [uriel1010@gmail.com](mailto:uriel1010@gmail.com)  
**LinkedIn**: [Uriel Manzur](https://www.linkedin.com/in/uriel-manzur/)

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/leetcode-solutions?style=social)](https://github.com/Uriel1010/leetcode)

---

> **License**: This project is licensed under the [MIT License](LICENSE).  
> Feel free to use and modify these solutions for your own learning or interview preparation!
```

---

### Notes

- The updated **Problem List** now includes all 19 functions from your `leetcode.py` file.
- Each function’s **Solution Breakdown** section includes a brief explanation, the approach used, and its time/space complexity.
- For custom or non-official LeetCode functions (like `lexicographicallySmallestArray` and `findMaxFish`), the link is marked as “No official link.”

This updated readme should now accurately reflect your repository’s content and provide clear guidance for anyone browsing your solutions.
    for x in nums1:
        xor1 ^= x
    xor2 = 0
    for x in nums2:
        xor2 ^= x
    
    result = 0
    if len(nums2) % 2 == 1:
        result ^= xor1
    if len(nums1) % 2 == 1:
        result ^= xor2
    return result
```
**Insight**:  
- If an array has an odd number of elements, each element in the other array will XOR with it once more than if it had an even count.

**Complexity**:  
- Time: O(n + m).  
- Space: O(1).

---

### 11. Find Eventual Safe Nodes 🛡
```python
def eventualSafeNodes(self, graph: list[list[int]]) -> list[int]:
    n = len(graph)
    color = [0] * n  # 0=unvisited, 1=visiting, 2=safe, 3=unsafe
    
    def dfs(node):
        if color[node] != 0:
            return color[node] == 2
        color[node] = 1
        for neighbor in graph[node]:
            if color[neighbor] == 1 or (color[neighbor] == 0 and not dfs(neighbor)):
                color[node] = 3  # unsafe
                return False
        color[node] = 2  # safe
        return True
    
    return [i for i in range(n) if dfs(i)]
```
**Approach**:  
- Use DFS and a coloring system to detect cycles.  
- Nodes that end up in cycles are unsafe; others are safe.

**Complexity**:  
- Time: O(n + e) where e is the number of edges.  
- Space: O(n) for recursion stack/visited states.

---

### 14. Course Schedule IV
```python
def checkIfPrerequisite(self, numCourses, prerequisites, queries):
    # Step 1: Initialize a reachability matrix
    reachable = [[False]*numCourses for _ in range(numCourses)]
    
    # Step 2: Mark direct prerequisites
    for pre, course in prerequisites:
        reachable[pre][course] = True
    
    # Step 3: Floyd-Warshall for transitive closure
    for k in range(numCourses):
        for i in range(numCourses):
            if reachable[i][k]:
                for j in range(numCourses):
                    if reachable[k][j]:
                        reachable[i][j] = True
    
    # Step 4: Answer queries
    return [reachable[u][v] for u, v in queries]
```
**Approach**:  
- Build a `reachable[i][j]` table.
- Apply the Floyd-Warshall-like approach for transitive closure in O(n³).
- Answer each query in O(1).

**Complexity**:  
- Time: O(n³ + q).
- Space: O(n²).

---

<a name="contributing"></a>
## 🤝 Contributing
1. **Fork** the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

<a name="contact"></a>
## 📬 Contact
**Author**: Uriel Manzur  
**Email**: [uriel1010@gmail.com](mailto:uriel1010@gmail.com)  
**LinkedIn**: [Uriel Manzur](https://www.linkedin.com/in/uriel-manzur/)

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/leetcode-solutions?style=social)](https://github.com/Uriel1010/leetcode)

---

> **License**: This project is licensed under the [MIT License](LICENSE).  
> Feel free to use and modify these solutions for your own learning or interview prep!
