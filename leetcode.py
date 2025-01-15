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
if __name__=="__main__":
    pass