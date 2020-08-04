# 栈和队列

## 简介

栈的特点是后入先出

![image.png](https://img.fuiboom.com/img/stack.png)

根据这个特点可以临时保存一些数据，之后用到依次再弹出来，常用于 DFS 深度搜索

队列一般常用于 BFS 广度搜索，类似一层一层的搜索

## Stack 栈

155.[min-stack](https://leetcode-cn.com/problems/min-stack/)(easy)

> 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

思路：用两个栈实现或插入元组实现，保证当前最小值在栈顶即可

```Python
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        if len(self.stack) > 0:
            self.stack.append((x, min(x, self.stack[-1][1])))
        else:
            self.stack.append((x, x))  # 当前值，当前值对应的min

    def pop(self) -> int:
        return self.stack.pop()[0]

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
        
    #--------两个栈--------------
        self.stack = []
        self.min_stack = []  # 记录最小值的idx，如果对应idx的elem pop了就同步pop

    def push(self, x: int) -> None:
        self.stack.append(x)
        if self.min_stack == [] or x < self.stack[self.min_stack[-1]]:
            self.min_stack.append(len(self.stack)-1)  # 记录最小值第一次入栈的位置

    def pop(self) -> None:
        # 如果出栈的是最小值的最后一个
        if len(self.stack)-1 == self.min_stack[-1]:
            self.min_stack.pop()
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1] if self.stack else None

    def getMin(self) -> int:
        return self.stack[self.min_stack[-1]] if self.min_stack else None
```

150.[evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)(medium)

> **波兰表达式计算** > **输入:** ["2", "1", "+", "3", "*"] > **输出:** 9
> **解释:** ((2 + 1) \* 3) = 9

思路：通过栈保存原来的元素，遇到表达式弹出运算，再推入结果，重复这个过程

```Python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        
        # 数字在前，符号在后
        # 数字和对应符合当即计算，结果参与后续计算
        # 数字栈保存数字
        nums = []
        for i in tokens:
            if i in '+-*/':  # 如果是运算符，立马就拿最近的两个num计算并入栈结果 
                nxt = nums.pop()
                prev = nums.pop()
                res = str(int(eval(prev+i+nxt)))
                nums.append(res)
            else:
                nums.append(i)
        return int(nums[-1])
```

394.[decode-string](https://leetcode-cn.com/problems/decode-string/)(medium)

> 给定一个经过编码的字符串，返回它解码后的字符串。
> s = "3[a]2[bc]", 返回 "aaabcbc".
> s = "3[a2[c]]", 返回 "accaccacc".
> s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".

思路：通过两个栈进行操作，一个用于存数，另一个用来存字符串

```Python
class Solution:
    def decodeString(self, s: str) -> str:
        
        stack_str = ['']
        stack_num = []
        
        num = 0
        for c in s:
            if c >= '0' and c <= '9':
                num = num * 10 + int(c)
            elif c == '[':
                stack_num.append(num)
                stack_str.append('')
                num = 0
            elif c == ']':
                cur_str = stack_str.pop()
                stack_str[-1] += cur_str * stack_num.pop()
            else:
                stack_str[-1] += c
        
        return stack_str[0]
        
        # ----------2 stacks----------
        stack = []
        res = ''
        multi = 0
        for char in s:
            if '0'<=char<='9':
                multi = multi*10 +int(char)
            # 开始处理括号内部，外部入栈,重新计数
            elif char =='[':
                stack.append((multi,res))  # 保留之前的所有字符串
                res = ''
                multi = 0
            # 方框里的str拼接到res上
            elif char == ']':
                curr_m, prev_r = stack.pop()
                res = prev_r + res * curr_m
            # 字符直接加到res上
            else:
                res += char 
        return res
```

利用栈进行 DFS 迭代搜索模板

```Python
def DFS(vertex):
    visited = set([])
    stack_dfs = [vertex]

    while len(stack_dfs) > 0:
        v = stack_dfs.pop()
        if v is not in visited:
            visit(v)
            visited.add(v)
            for n in v.neighbors:
                stack_dfs.append(n)

    return
        # --------------递归法--------------
        # 总体思路与辅助栈法一致，不同点在于将 [ 和 ] 分别作为递归的开启与终止条件：
        # 当 s[i] == ']' 时，返回当前括号内记录的 res 字符串与 ] 的索引 i （更新上层递归指针位置）；
        # 当 s[i] == '[' 时，开启新一层递归，记录此 [...] 内字符串 tmp 和递归后的最新索引 i，并执行 res + multi * tmp 拼接字符串。
        # 遍历完毕后返回 res。
        # 时间复杂度 O(N)O(N)O(N)，递归会更新索引，因此实际上还是一次遍历 s；
        # 空间复杂度 O(N)O(N)O(N)，极端情况下递归深度将会达到线性级别。

        def dfs(s, i):
            res = ''  # 当前此层处理字符串结果
            multi = 0  # 重复次数
            while i < len(s):
                if '0' <= s[i] <= '9':  # 数字加和
                    multi = multi*10 +int(s[i])
                elif s[i] == '[':  # 开始递归
                    i, tmp = dfs(s, i+1)  # 统计[]内字符串，并返回已处理的位置防止重复处理
                    res += multi * tmp  
                    multi = 0  # reset
                elif s[i] == ']':
                    return i, res  # finish [],递归结束
                else:
                    res += s[i]  # 字符直接统计
                i +=1  # next char
            return res
        return dfs(s,0)
```

94.[binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)(medium)

> 给定一个二叉树，返回它的*中序*遍历。

- [reference](https://en.wikipedia.org/wiki/Tree_traversal#In-order)

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        
        stack, inorder = [], []
        node = root
 
        while len(stack) > 0 or node is not None:
            if node is not None: 
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                inorder.append(node.val)
                node = node.right
        
        return inorder
```

133.[clone-graph](https://leetcode-cn.com/problems/clone-graph/)(medium)

> 给你无向连通图中一个节点的引用，请你返回该图的深拷贝（克隆）。

- BFS

```Python
class Solution:
    def cloneGraph(self, start: 'Node') -> 'Node':
        
        if start is None:
            return None
        
        visited = {start: Node(start.val, [])}
        bfs = collections.deque([start])
        
        while len(bfs) > 0:
            curr = bfs.popleft()
            curr_copy = visited[curr]
            for n in curr.neighbors:
                if n not in visited:
                    visited[n] = Node(n.val, [])
                    bfs.append(n)
                curr_copy.neighbors.append(visited[n])
        
        return visited[start]
```

- DFS iterative

```Python
class Solution:
    def cloneGraph(self, start: 'Node') -> 'Node':
        
        if start is None:
            return None
        
        if not start.neighbors:
            return Node(start.val)
        
        visited = {start: Node(start.val, [])}
        dfs = [start]
        
        while len(dfs) > 0:
            peek = dfs[-1]
            peek_copy = visited[peek]
            if len(peek_copy.neighbors) == 0:
                for n in peek.neighbors:
                    if n not in visited:
                        visited[n] = Node(n.val, [])
                        dfs.append(n)
                    peek_copy.neighbors.append(visited[n])
            else:
                dfs.pop()
        
        return visited[start]
```



200.[number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)(medium)

> 给定一个由  '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

High-level problem: number of connected component of graph

思路：通过深度搜索遍历可能性（注意标记已访问元素）

```Python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])

        def dfs_iter(i, j):
            dfs = []
            dfs.append((i, j))
            while len(dfs) > 0:
                i, j = dfs.pop()
                if grid[i][j] == '1':
                    grid[i][j] = '0'
                    if i - 1 >= 0:
                        dfs.append((i - 1, j))
                    if j - 1 >= 0:
                        dfs.append((i, j - 1))
                    if i + 1 < m:
                        dfs.append((i + 1, j))
                    if j + 1 < n:
                        dfs.append((i, j + 1))
            return
        
        num_island = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    num_island += 1
                    dfs_iter(i, j)
        
        return num_island
```

84.[largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)(hard)

> 给定 _n_ 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
> 求在该柱状图中，能够勾勒出来的矩形的最大面积。

思路 1：蛮力法，比较每个以 i 开始 j 结束的最大矩形，A(i, j) = (j - i + 1) * min_height(i, j)，时间复杂度 O(n^2) 无法AC

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        max_area = 0
        
        n = len(heights)
        for i in range(n):
            min_height = heights[i]
            for j in range(i, n):
                min_height = min(min_height, heights[j])
                max_area = max(max_area, min_height * (j - i + 1))
        
        return max_area
```

思路 2: 设 A(i, j) 为区间 [i, j) 内最大矩形的面积，k 为 [i, j) 内最矮 bar 的坐标，则 A(i, j) = max((j - i) * heights[k], A(i, k), A(k+1, j)), 使用分治法进行求解。时间复杂度 O(nlogn)，其中使用简单遍历求最小值无法 AC (最坏情况退化到 O(n^2))，使用线段树优化后勉强AC

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        n = len(heights)
        
        seg_tree = [None] * n
        seg_tree.extend(list(zip(heights, range(n))))
        for i in range(n - 1, 0, -1):
            seg_tree[i] = min(seg_tree[2 * i], seg_tree[2 * i + 1], key=lambda x: x[0])
        
        def _min(i, j):
            min_ = (heights[i], i)
            i += n
            j += n
            while i < j:
                if i % 2 == 1:
                    min_ = min(min_, seg_tree[i], key=lambda x: x[0])
                    i += 1
                if j % 2 == 1:
                    j -= 1
                    min_ = min(min_, seg_tree[j], key=lambda x: x[0])
                i //= 2
                j //= 2
            
            return min_
        
        def LRA(i, j):
            if i == j:
                return 0
            min_k, k = _min(i, j)
            return max(min_k * (j - i), LRA(k + 1, j), LRA(i, k))
        
        return LRA(0, n)
```

思路 3：包含当前 bar 最大矩形的边界为左边第一个高度小于当前高度的 bar 和右边第一个高度小于当前高度的 bar。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        n = len(heights)
        
        stack = [-1]
        max_area = 0
        
        for i in range(n):
            while len(stack) > 1 and heights[stack[-1]] > heights[i]:
                h = stack.pop()
                max_area = max(max_area, heights[h] * (i - stack[-1] - 1))
            stack.append(i)
        
        while len(stack) > 1:
            h = stack.pop()
            max_area = max(max_area, heights[h] * (n - stack[-1] - 1))
        
        return max_area
```

## Queue 队列

常用于 BFS 宽度优先搜索

232.[implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)(easy)

> 使用栈实现队列

```Python
class MyQueue:

    def __init__(self):
        self.cache = []
        self.out = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.cache.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.out) == 0:
            while len(self.cache) > 0:
                self.out.append(self.cache.pop())

        return self.out.pop() 

    def peek(self) -> int:
        """
        Get the front element.
        """
        if len(self.out) > 0:
            return self.out[-1]
        else:
            return self.cache[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.cache) == 0 and len(self.out) == 0
```

102.[binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)(medium)

> 二叉树的层序遍历

```Python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        
        levels = []
        if root is None:
            return levels
        
        bfs = collections.deque([root])
        
        while len(bfs) > 0:
            levels.append([])
            
            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                levels[-1].append(node.val)
                
                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)
        
        return levels
```

542.[01-matrix](https://leetcode-cn.com/problems/01-matrix/)(medium)

> 给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
> 两个相邻元素间的距离为 1

思路 1: 从 0 开始 BFS, 遇到距离最小值需要更新的则更新后重新入队更新后续结点

```Python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return matrix
        
        m, n = len(matrix), len(matrix[0])
        dist = [[float('inf')] * n for _ in range(m)]
        
        bfs = collections.deque([])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dist[i][j] = 0
                    bfs.append((i, j))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while len(bfs) > 0:
            i, j = bfs.popleft()
            for dn_i, dn_j in neighbors:
                n_i, n_j = i + dn_i, j + dn_j
                if n_i >= 0 and n_i < m and n_j >= 0 and n_j < n:
                    if dist[n_i][n_j] > dist[i][j] + 1:
                        dist[n_i][n_j] = dist[i][j] + 1
                        bfs.append((n_i, n_j))
        
        return dist        
```

思路 2: 2-pass DP，dist(i, j) = max{dist(i - 1, j), dist(i + 1, j), dist(i, j - 1), dist(i, j + 1)} + 1

```Python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return matrix
        
        m, n = len(matrix), len(matrix[0])
        
        dist = [[float('inf')] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i - 1 >= 0:
                        dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j])
                    if j - 1 >= 0:
                        dist[i][j] = min(dist[i][j - 1] + 1, dist[i][j])
                else:
                    dist[i][j] = 0
        
        for i in range(-1, -m - 1, -1):
            for j in range(-1, -n - 1, -1):
                if matrix[i][j] == 1:
                    if i + 1 < 0:
                        dist[i][j] = min(dist[i + 1][j] + 1, dist[i][j])
                    if j + 1 < 0:
                        dist[i][j] = min(dist[i][j + 1] + 1, dist[i][j])
        
        return dist
```



## 总结

- 熟悉栈的使用场景
  - 后出先出，保存临时值
  - 利用栈 DFS 深度搜索
- 熟悉队列的使用场景
  - 利用队列 BFS 广度搜索

## 练习

- [ ] 155.[min-stack](https://leetcode-cn.com/problems/min-stack/)(easy)
- [ ] 150.[evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)(medium)
- [ ] 394.[decode-string](https://leetcode-cn.com/problems/decode-string/)(medium)
- [ ] 94.[binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)(medium)
- [ ] 133.[clone-graph](https://leetcode-cn.com/problems/clone-graph/)(medium)
- [ ] 200.[number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)(medium)
- [ ] 84.[largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)(hard)
- [ ] 232.[implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)(easy)
- [ ] 542.[01-matrix](https://leetcode-cn.com/problems/01-matrix/)(medium)
