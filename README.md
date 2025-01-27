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
