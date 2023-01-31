"""
Rifat Monzur
ID: 2200367
Variable naming: snake case
O(?): worst case if not mentioned otherwise
"""

"""
List product of all numbers except the number at specific index
Arguments:
    num: list of numbers
Returns:
    list of number containing product of all numbers except the i th index
Time Complexity:
    O(n) where n is the length of num
Space Complexity:
    O(n) where n is the length of num
"""
def Exercise1(num):
    left, right = [1], [1]
    sz = len(num)

    """
    Calculating list 'left' 
    where left[i] is product of all the numbers of 'num' from 0 to i-1 
    In other word, left[i] is the product of all number in 'num' which are left of i th index
    """
    for i in range(sz-1):
        left.append(left[i] * num[i])

    """
    Calculating list 'right' 
    where right[i] is product of all the numbers of 'num' from i+1 to sz-1 
    In other word, right[i] is the product of all number in 'num' which are right of i th index
    """
    for i in range(sz-1, 0, -1):
        right.insert(0, right[0] * num[i])

    return [left[i] * right[i] for i in range(sz)]


"""
Convert two dimensional list into one dimensional list in spiral order
Arguments:
    matrix: two dimensional list of numbers
Returns:
    one dimensional list in spiral order of 'matrix' list
Time Complexity:
    O(m*n), where m is number of rows & n is number of columns
Space Complexity:
    O(m*n), where m is number of rows & n is number of columns
"""
def Exercise2(matrix):
    m = len(matrix)  # m = number of rows
    n = len(matrix[0])  # n = number of columns
    sz = m * n  # sz = number of elements in 'matrix'

    """
    vis: boolean list with dimension m * n
    vis[i][j] = False means matrix[i][j] still have not been visited in spiral order
    vis[i][j] = True means matrix[i][j] have been visited in spiral order
    """
    vis = [[False for _ in range(n)] for _ in range(m)]

    """
    r: spiral iteration at r th row
    c: spiral iteration at c th column
    result: list of numbers added in spiral order after each iteration  
    """
    r, c, result = 0, -1, []
    is_left_to_right = True  # if spiral will go left to right or right to left
    is_up_to_down = True  # if spiral will go up to down or down to up

    """
    Nested while loops are simulating the behaviour of spiral order
    Outer loop:
    Run until all the elements are iterate over. Toggle between horizontal and vertical direction.
    First inner loop: 
    Iterate in horizontal directions. It goes from left to right until reaches end of column of unvisited numbers. 
    In the next iteration it from goes from right to left until reaches start of column  of unvisited numbers.
    It toggle horizontal direction everytime, as it reaches end or start of unvisited column after each iteration. 
    Second inner loop: 
    Iterate in vertical directions. It goes from up to down until reaches end of row of unvisited numbers. 
    In the next iteration it from goes from down to up until reaches start of row  of unvisited numbers.
    It toggle vertical direction everytime, as it reaches end or start of unvisited row after each iteration.
    """
    while len(result) < sz:  # while loop runs until all the elements of 'matrix' are added
        c_adder = 1 if is_left_to_right else -1  # c_adder is adder with 'c' to go to right horizontal direction
        while c + c_adder >= 0 and c + c_adder < n and not vis[r][c + c_adder]:
            c += c_adder
            result.append(matrix[r][c])
            vis[r][c] = True

        is_left_to_right = not is_left_to_right  # toggle horizontal direction

        r_adder = 1 if is_up_to_down else -1 # r_adder is adder with 'r' to go to right vertical direction
        while r + r_adder >= 0 and r + r_adder < m and not vis[r+r_adder][c]:
            r += r_adder
            result.append(matrix[r][c])
            vis[r][c] = True

        is_up_to_down = not is_up_to_down  # toggle vertical direction

    return result


"""
Number of distinct indexes where sum is 0 for four equally length arrays
Arguments:
    num1: list of integers 
    num2: list of integers
    num3: list of integers
    num4: list of integers
Returns:
    number of tuples where num1[i] + num2[j] + num3[k] + num4[l] = 0
Time Complexity:
    average case O(n^3), where n is the length of num1
    worst case O(n^4), where n is the length of num1
Space Complexity:
    O(1), No extra space needed
"""
def Exercise3(num1, num2, num3, num4):
    sz, mp = len(num1), {}

    # mapping the counting of num4 list numbers into a dictionary
    for i in range(sz):
        mp[num4[i]] = mp[num4[i]] + 1 if num4[i] in mp else 1

    ans = 0
    for idx1 in range(sz):
        for idx2 in range(sz):
            for idx3 in range(sz):
                val = -(num1[idx1] + num2[idx2] + num3[idx3])  # negative of sum of num1[idx1], num[idx2] & num[idx3]
                ans += mp.setdefault(val, 0)  # check the count in num4 for 'val' which will make sum = 0

    return ans


"""
Maximum area of two lines contain from a given list
Arguments:
    height: list of integers 
Returns:
    maximum area two numbers from height can make
Time Complexity:
    O(n), where n is the length of height
Space Complexity:
    O(1), No extra space needed
"""
def Exercise4(height):
    left, max_area = 0,0
    right = len(height) - 1

    """
    Followed two pointer approach.
    It can be observed, if we move from left and right point, which among them is shorter, 
    we need to move to it to find bigger area. Because, shorter one height will be counted.
    As any water above the shorter line will overflow. 
    """
    while left < right:
        # take maximum of current area & previous maximum area
        max_area = max(min(height[left], height[right]) * (right-left), max_area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area


"""
Calculate length of the longest consecutive numbers sequence from a given unordered list of numbers
Arguments:
    nums: list of integers 
Returns:
    length of the longest consecutive elements sequence
Time Complexity:
    O(n*log(n)), where n is the length of nums
Space Complexity:
    O(n), to sort nums
"""
def Exercise5(nums):
    nums = sorted(nums)  # Time Complexity: O(n*log(n))
    left, ans = 0, 0
    sz = len(nums)

    """
    Following two pointers approach. Time Complexity: O(n)
    Make right = left + 1  and move right until you find consecutive numbers.
    Calculate the length of consecutive numbers.
    Make left = right, as no longer consecutive number exist between current left and right
    """
    while left < sz:
        right = left+1
        while right < sz and nums[right]-nums[right-1] == 1:
            right += 1

        ans = max(right-left, ans)
        left = right

    return ans

"""
Find the duplicate element of range [1,n] from a list of numbers 
Arguments:
    nums: list of integers 
Returns:
    duplicate element from the list
Time Complexity:
    average case O(n), where n is the length of nums
    worst case O(n^2), where n is the length of nums
Space Complexity:
    O(n), where n is the length of nums
"""
def Exercise6(nums):
    st = set()
    for num in nums:
        # check if number already exist in Set, if exist this number is the duplicate
        if num in st:  # Time Complexity: average case O(1) worst case O(n)
            return num
        st.add(num)  # Time Complexity: average case O(1) worst case O(n)


"""
Find longest substring length with k distinct characters
Arguments:
    s: string of characters
    k: distinct number of characters for substring
Returns:
    longest substring length with k distinct characters
Time Complexity:
    O(n), where n is the length of s
Space Complexity:
    O(1), constant space required
"""
def Exercise7(s, k):
    char_set, char_last_occurrence, sz = set(), dict(), len(s)
    left, ans = 0, 0

    for i in range(sz):
        char_set.add(s[i])
        char_last_occurrence[s[i]] = i  # mapping the last occurrence index of character

        if len(char_set) > k:  # check current substring got more than k characters
            left_char_index = len(s) + 1
            left_char = ''

            for ch in char_set:  # finding the left most character in the current substring to remove it
                if char_last_occurrence[ch] < left_char_index:
                    left_char_index = char_last_occurrence[ch]
                    left_char = ch

            left = left_char_index+1  # moving beginning of substring to make it have k characters
            char_set.remove(left_char)  # removing the left most character from set

        ans = max(ans, i - left + 1)

    return ans

from collections import deque
"""
Find the duplicate element of range [1,n] from a list of numbers 
Arguments:
    nums: list of integers 
    k: length of sliding window
Returns:
    list of maximum elements from sliding windows 
Time Complexity:
    O(n), where n is the length of nums
Space Complexity:
    O(k), where k is the sliding window length
"""
def Exercise8(nums, k):
    """
    Using data structure deque which is especial kind of list.
    Deque can insert and pop element from both front and back at O(1) Time Complexity.
    Pop from front needed to remove any value outside sliding window range.
    Pop from back needed to keep deque always in order.
    """
    dq, ans = deque(), []

    for i in range(len(nums)):
        if dq and dq[0] == i - k:  # Remove index outside of current sliding window
            dq.popleft()
        """
        Remove any index with value less than i th value from the back of deque
        By doing so, we always keep the deque always in order 
        """
        while dq and nums[dq[len(dq)-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k-1:
            ans.append(nums[dq[0]])

    return ans

"""
Find minimum length substring of s containing all characters of t
Arguments:
    s: string of characters
    t: string of characters
Returns:
    substring of s contains all characters of t
Time Complexity:
    O(n^2), where n is the length of s
Space Complexity:
    O(1), constant space
"""
def Exercise9(s,t):
    t_char_count = dict()
    ans = ""

    for ch in t:  # mapping character count of t
        t_char_count[ch] = t_char_count.setdefault(ch, 0) + 1

    for i in range(len(s)):
        s_char_count = dict()
        distinct_char_count = len(t_char_count.keys())

        for j in range(i, len(s)):  # substring start from i th index
            s_char_count[s[j]] = s_char_count.setdefault(s[j], 0) + 1  # mapping character count substring
            if s[j] in t_char_count and t_char_count[s[j]] == s_char_count[s[j]]:  # check character exist in t and count is equal
                distinct_char_count -= 1

            if distinct_char_count == 0:  # all distinct character found
                if len(ans) == 0 or len(ans) > j - i + 1:
                    ans = s[i:j+1]
                break

    return ans



class Cache:
    mp = dict()
    def __init__(self, capacity):
        self.capacity = capacity

    """
    Establish key value mapping and remove key value if exceed capacity
    Arguments:
        key: any type of data type
        value: any type of data type
    Time Complexity:
        Average Case O(1)
        Worst Case O(n), where n is the number of elements 
    """
    def put(self, key, value):
        self.mp[key] = value
        if len(self.mp.keys()) > self.capacity:
            extra_key = list(self.mp.keys())[0]
            del self.mp[extra_key]

    """
    Get mapped value for the key
    Arguments:
        key: any type of data type
    Returns:
        value against the key or -1
    Time Complexity:
        Average Case O(1)
        Worst Case O(n), where n is the number of elements 
    """
    def get(self, key):
        if key in self.mp:
            return self.mp[key]

        return -1

def run_test():
    print("TESTING EXERCISE 1")
    print(Exercise1([1, 2, 3, 4]))

    print("TESTING EXERCISE 2")
    print(Exercise2([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(Exercise2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))

    print("TESTING EXERCISE 3")
    print(Exercise3([1, 2], [-2, -1], [-1, 2], [0, 2]))
    print(Exercise3([0], [0], [0], [0]))
    print(Exercise3([], [], [], []))

    print("TESTING EXERCISE 4")
    print(Exercise4([1, 8, 6, 2, 5, 4, 8, 3, 7]))

    print("TESTING EXERCISE 5")
    print(Exercise5([100, 4, 200, 1, 3, 2]))
    print(Exercise5([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))

    print("TESTING EXERCISE 6")
    print(Exercise6([1, 3, 4, 2, 2]))
    print(Exercise6([3, 1, 3, 4, 2]))

    print("TESTING EXERCISE 7")
    print(Exercise7("eceba", 2))
    print(Exercise7("aa", 1))
    print(Exercise7("baece", 2))

    print("TESTING EXERCISE 8")
    print(Exercise8([1, 3, -1, -3, 5, 3, 6, 7], 3))
    print(Exercise8([1], 1))
    print(Exercise8([13, 9, 12,7], 3))

    print("TESTING EXERCISE 9")
    print(Exercise9("ADOBECODEBANC", "ABC"))
    print(Exercise9("a", "a"))
    print(Exercise9("a", "aa"))

    print("TESTING EXERCISE 10")
    C = Cache(2)
    C.put(1, 1)
    C.put(2, 2)
    print(C.get(1))
    C.put(3, 3)
    print(C.get(1))
    C.put(4, 4)
    print(C.get(2))
    print(C.get(3))
    print(C.get(4))

run_test()