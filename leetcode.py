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

if __name__=="__main__":
    pass