class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
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

        The time complexity of the given function numRescueBoats can be analyzed as follows:

Sorting the input array people takes O(nlogn) time, where n is the length of the array.
The function then iterates over the sorted people array once, and for each person, it tries to find a partner (if possible) whose combined weight is less than or equal to the given limit. This iteration takes O(n) time in the worst case, where n is the length of the array.
For each boat, the function removes the people who have already been assigned to the boat from the input array. Removing an element from a list takes O(n) time in the worst case.
Finally, the function returns the length of the boats list, which takes O(1) time.
Therefore, the overall time complexity of the numRescueBoats function can be expressed as O(nlogn + n^2), which simplifies to O(n^2) in the worst case.
        """
        tmp = sorted(people)[::-1]
        boats = []
        lim = limit
        boat = []
        while tmp != []:
            boats.append(boat)
            lim = limit
            boat = []
            for i in range(len(tmp)):
                if lim >= tmp[i]:
                    lim -= tmp[i]
                    boat.append(tmp[i])
                    if lim == 0:
                        break
            for j in range(len(boat)):
                tmp.remove(boat[j])
        return len(boats)






if __name__=="__main__":
    h = Solution()
    print(h.Boats_to_Save_People(people = [1,2], limit = 3))
    print(h.Boats_to_Save_People(people = [3,2,2,1], limit = 3))
    print(h.Boats_to_Save_People(people = [3,5,3,4], limit = 5))