# LeetCode Solutions in Python

This repository contains concise, well-documented solutions to various LeetCode problems. Each solution is written inside a single class `Solution` for simplicity and easy reference. Below is a summary of each function, its purpose, and some insights on time complexity and approach.

---

## Table of Contents

1. [Overview](#overview)  
2. [Installation and Usage](#installation-and-usage)  
3. [Functions Breakdown](#functions-breakdown)  
   - [1. `numRescueBoats`](#1-numrescueboats)  
   - [2. `partitionString`](#2-partitionstring)  
   - [3. `search`](#3-search)  
   - [4. `prefixCount`](#4-prefixcount)  
   - [5. `canConstruct`](#5-canconstruct)  
   - [6. `findThePrefixCommonArray`](#6-findtheprefixcommonarray)  
   - [7. `minimizeXor`](#7-minimizexor)  
   - [8. `xorAllNums`](#8-xorallnums)  
   - [9. `minCost`](#9-mincost)  
4. [Contact](#contact)

---

## Overview

- Each function within the `Solution` class addresses a different coding problem.  
- The code is clean, readable, and follows Pythonic conventions.  
- Designed to illustrate problem-solving skills, clear logic, and knowledge of fundamental algorithms and data structures.  

**Why might recruiters be interested?**  
- Demonstrates familiarity with classic coding interview questions.  
- Shows ability to optimize solutions with respect to time complexity and space complexity.  
- Provides clarity and maintainability through docstrings and code comments.

---

## Installation and Usage

1. **Clone or Download** this repository.
2. Ensure you have **Python 3.x** installed.
3. Navigate to the folder containing `solution.py` (or whatever you name this file).
4. You can either:
   - **Import the class** `Solution` into your own file and invoke the methods directly.
   - Run `python solution.py` (currently the `if __name__=="__main__": pass` block does nothing, but you can add test code there).

Example usage:
```python
from solution import Solution

sol = Solution()
boats_needed = sol.numRescueBoats([3,2,2,1], 3)
print(boats_needed)  # Expected output: 3
```

---

## Functions Breakdown

### 1. `numRescueBoats(people: list[int], limit: int) -> int`
**Description**  
- **LeetCode 881**: Boats to Save People.  
- Returns the minimum number of boats required to carry all people, given each boat can hold up to 2 people without exceeding a weight limit.  
- **Approach**:  
  1. Sort the array of people’s weights.  
  2. Use two-pointer technique from both ends: if the heaviest person can pair with the lightest person, move both pointers inward. Otherwise, move only the pointer from the heavier side.  
- **Time Complexity**: O(n log n), due to sorting.  
- **Space Complexity**: O(1) additional space.

### 2. `partitionString(s: str) -> int`
**Description**  
- Splits string `s` into the maximum number of substrings such that each substring contains no repeated characters.  
- **Approach**:  
  1. Greedily build a substring until a repeated character is found.  
  2. Once a repeat is detected, start a new substring.  
- **Time Complexity**: O(n).  
- **Space Complexity**: O(n) in the worst case if each character is unique.

### 3. `search(nums: list[int], target: int) -> int`
**Description**  
- Performs a binary search to find the `target` in a sorted list `nums`. Returns the index if found, otherwise `-1`.  
- **Approach**:  
  1. Standard binary search logic (divide and conquer).  
- **Time Complexity**: O(log n).  
- **Space Complexity**: O(1).

### 4. `prefixCount(words: list[str], pref: str) -> int`
**Description**  
- Counts how many strings in the list `words` start with the prefix `pref`.  
- **Approach**:  
  1. Use Python’s `startswith` method for each word; sum up the results.  
- **Time Complexity**: O(n * p), where `p` is the length of the prefix in the worst case.  
- **Space Complexity**: O(1).

### 5. `canConstruct(s: str, k: int) -> bool`
**Description**  
- Determines if string `s` can be split into `k` palindromic substrings.  
- **Approach**:  
  1. If `k` > length of `s`, it is immediately impossible.  
  2. Count the frequencies of each character.  
  3. Count how many characters appear an odd number of times; let that be `odd_count`.  
  4. We can form a palindrome only if `odd_count <= k <= len(s)`.  
- **Time Complexity**: O(n), for counting characters.  
- **Space Complexity**: O(1), as frequency array is fixed size (26 letters).

### 6. `findThePrefixCommonArray(A: list[int], B: list[int]) -> list[int]`
**Description**  
- For each index `i`, determines the size of the intersection of the prefixes `A[:i+1]` and `B[:i+1]`.  
- **Approach**:  
  1. Maintain two sets: `seenA` and `seenB`.  
  2. Insert `A[i]` into `seenA` and `B[i]` into `seenB`.  
  3. Compute the intersection size using `len(seenA & seenB)`.  
- **Time Complexity**: O(n), assuming set intersection is optimized.  
- **Space Complexity**: O(n) for the sets in the worst case.

### 7. `minimizeXor(num1: int, num2: int) -> int`
**Description**  
- Reorders bits in `num1` to match the number of set bits in `num2` (if `num2` has more set bits, `num1` sets bits in lower positions; if fewer, `num1` turns off bits).  
- **Approach**:  
  1. Count set bits in `num1` and `num2`.  
  2. If `num1` has more set bits, flip off the excess from the least significant bits.  
  3. If `num1` has fewer set bits, flip on bits from the least significant positions.  
- **Time Complexity**: O(1) or O(32) specifically, since we only check at most 32 bits for integers.  
- **Space Complexity**: O(1).

### 8. `xorAllNums(nums1: list[int], nums2: list[int]) -> int`
**Description**  
- Given two arrays, forms all pairwise XORs (`nums1[i] ^ nums2[j]`) and returns the XOR of the resulting set of values.  
- **Key Insight**:
  - If `len(nums2)` is even, every element of `nums1` is XORed an even number of times and thus cancels out. If `len(nums2)` is odd, they contribute to the final XOR. The same logic applies for elements in `nums2` depending on the parity of `len(nums1)`.  
- **Time Complexity**: O(n + m).  
- **Space Complexity**: O(1).

### 9. `minCost(grid: list[list[int]]) -> int`
**Description**  
- **LeetCode 1368**: Minimum Cost to Make at Least One Valid Path in a Grid.  
- Each cell has a direction (up, down, left, right). Moving in the indicated direction has zero cost; changing direction costs 1. Find the minimum cost to travel from `(0,0)` to `(m-1, n-1)`.  
- **Approach**:  
  1. Use a **0-1 BFS** to handle edges of cost 0 or 1.  
  2. Maintain a custom double-ended queue to push 0-cost moves to the front and 1-cost moves to the back.  
- **Time Complexity**: O(m·n).  
- **Space Complexity**: O(m·n) for the distance matrix and the queue in the worst case.

---

## Contact

If you have any questions or suggestions, feel free to reach out or open an issue.  
- **Author**: [Your Name Here]  
- **Email**: [Your Contact Email]

Thank you for checking out these solutions!