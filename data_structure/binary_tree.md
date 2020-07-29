# 二叉树

## 知识点

### 二叉树遍历

**前序遍历**：**先访问根节点**，再前序遍历左子树，再前序遍历右子树
**中序遍历**：先中序遍历左子树，**再访问根节点**，再中序遍历右子树
**后序遍历**：先后序遍历左子树，再后序遍历右子树，**再访问根节点**

注意点

- 以根访问顺序决定是什么遍历
- 左子树都是优先右子树

#### 递归模板

递归实现二叉树遍历非常简单，不同顺序区别仅在于访问父结点顺序

```Python
def preorder_rec(root):
    if root is None:
        return
    visit(root)
    preorder_rec(root.left)
    preorder_rec(root.right)
    return

def inorder_rec(root):
    if root is None:
        return
    inorder_rec(root.left)
    visit(root)
    inorder_rec(root.right)
    return

def postorder_rec(root):
    if root is None:
        return
    postorder_rec(root.left)
    postorder_rec(root.right)
    visit(root)
    return
```

#### 144.[前序非递归](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

本质上是图的DFS的一个特例，因此可以用栈来实现

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        
        if root is None:
            return []
        prestack = []
        prelist = []

        prestack = [root]

        while prestack :
            node = prestack.pop()
            # 两种写法
            # prelist.append(node.val)
            # if node.right is not None:
            #     prestack.append(node.right)
            # if node.left is not None:
            #     prestack.append(node.left)
            
            if node:
                prelist.append(node.val)
                prestack.append(node.right)
                prestack.append(node.left)
        return prelist
        # # 递归
        # def get_prelist(root):
        #     if root is None:
        #         return
        #     res.append(root.val)
        #     get_prelist(root.left)
        #     get_prelist(root.right)
        #     return
        # res = []
        # get_prelist(root)
        # return res
        
        # Divide and conquer
        if root is None:
            return []
        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)
        return [root.val]+left+right
```

#### 94.[中序非递归](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        s, inorder = [], []
        node = root
        while len(s) > 0 or node is not None:
            if node is not None:
                s.append(node)
                node = node.left
            else:
                node = s.pop()
                inorder.append(node.val)
                node = node.right
        return inorder
        # # 递归版本
        # def get_inorder_list(root):
        #     if root is None:
        #         return
        #     if root.left is not None:
        #         get_inorder_list(root.left)
        #     inorder_list.append(root.val)
        #     if root.right is not None:
        #         get_inorder_list(root.right)
        #     return
        # inorder_list = []
        # get_inorder_list(root)
        # return inorder_list

        # 迭代版本
        # 一直找到最左下的node，途经的一切left node全部入栈
        # 出栈就visit，然后查是不是有right node，有就继续入栈所有left node
        # 没有就接着出栈，visit
        # def push_node(root):
        #     if root is None:
        #         return
        #     left_stack.append(root)
        #     push_node(root.left)
        # left_stack = []
        # inorder_list = []
        # push_node(root)
        # while left_stack:
        #     node = left_stack.pop()
        #     inorder_list.append(node.val)
        #     if node.right is not None:
        #         push_node(node.right)
        # return inorder_list
        
        # divide &conquer
        if root is None:
            return []

        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)
        return left+ [root.val]+right
```

#### 145.[后序非递归](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

```Python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:

        s, postorder = [], []
        node, last_visit = root, None
        
        while len(s) > 0 or node is not None:
            if node is not None:
                s.append(node)
                node = node.left
            else:
                peek = s[-1]
                if peek.right is not None and last_visit != peek.right:
                    node = peek.right
                else:
                    last_visit = s.pop()
                    postorder.append(last_visit.val)
        
        
        return postorder
        
        # # 递归版本
        # def get_post_list(root):
        #     if root is None:
        #         return
        #     if root.left is not None:
        #         get_post_list(root.left)
        #     if root.right is not None:
        #         get_post_list(root.right)
        #     post_list.append(root.val)
        #     return
        # post_list = []
        # get_post_list(root)
        # return post_list

        # 迭代版本
        # 最后visit，先探索左下，途经都入栈
        # 如果有right node，就继续探索right node
        # 如果没有right node或者刚刚visit过right node，visit 当前栈底node
        def get_left_list(root):
            if root is None:
                return
            left_list.append(root)
            get_left_list(root.left)
            
        left_list = []
        post_list = []
        last_node = None
        get_left_list(root)  # 探索到最左下
        while left_list:
            tmp = left_list[-1]
            # 如果有right node，而且上一个visit的node不是tmp的right node防止重复入栈right tree
            while tmp.right is not None and last_node != tmp.right:
                get_left_list(tmp.right)  # 每个right仅探索一次，避免出栈后重复探索
                tmp = left_list[-1]
            # 如果没有right node或者已经visit过right node就访问当前栈底node（left node或者father）
            # 即vist顺序为left-father 或者left-right-father
            last_node = left_list.pop()
            post_list.append(last_node.val)
        return post_list
        
        #------divide&conquer---------
        if root is None:
            return []

        left = self.postorderTraversal(root.left)
        right = self.postorderTraversal(root.right)
        return left+ right+[root.val]
        
```

注意点

- 核心就是：根节点必须在右节点弹出之后，再弹出

DFS 深度搜索-从下向上（分治法）

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        
        if root is None:
            return []
        
        left_result = self.preorderTraversal(root.left)
        right_result = self.preorderTraversal(root.right)
        
        return [root.val] + left_result + right_result
```

注意点：

> DFS 深度搜索（从上到下） 和分治法区别：前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并

#### 102.[BFS 层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

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

### 分治法应用

先分别处理局部，再合并结果

适用场景

- 快速排序
- 归并排序
- 二叉树相关问题

分治法模板(对比层次遍历的代码看)

- 递归返回条件( if root is None: return [] )
- 分段处理   ( func(root.left);func(root.right) )
- 合并结果   ( return [root.val]+left_vals+right_vals )

常见题目示例

#### [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最大深度。

思路 1：分治法

```Python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # --递归返回条件--
        if root is None:
            return 0  当前node为None则无深度
        # --子树分别处理--,左右子树的最大深度
        depth = max(self.maxDepth(root.left),self.maxDepth(root.right))
        # 合并结果
        return 1 + depth  # 子树最大深度加本层深度
```

思路 2：层序遍历

```Python
class Solution:
    def maxDepth(self, root: TreeNode) -> List[List[int]]:
        
        depth = 0
        if root is None:
            return depth
        
        bfs = collections.deque([root])  # 双向队列
        
        while len(bfs) > 0:  # 如果本层有node
            depth += 1  # 本层有node就深度加1
            level_size = len(bfs)  # 本层有几个node
            for _ in range(level_size):
                node = bfs.popleft()  # 左侧出队
                if node.left is not None:
                    bfs.append(node.left)  # 右侧入队
                if node.right is not None:
                    bfs.append(node.right)
        
        return depth
```

#### 110.[balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

思路 1：分治法，左边平衡 && 右边平衡 && 左右两边高度 <= 1，

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # # O(n),自底向上
        # def check(root):
        #     if root is None:
        #         return 0, True

        #     left_depth, left_label = check(root.left)
        #     right_depth, right_label = check(root.right)
        #     max_depth = max(left_depth, right_depth)
        #     return 1 + max_depth, left_label and right_label and abs(left_depth-right_depth)<2
        # depth, label = check(root)
        # return label
        # ----------------------------------------
        # 树高
        # 对二叉树做先序遍历，从底至顶返回子树最大高度，若判定某子树不是平衡树则 “剪枝” ，直接向上返回。
        # 统计树高的过程中，计算高度差，一旦超过1，即一直返回-1表示不平衡，否则一直计算高度
        return self.height(root) != -1

    def height(self, root):
        if root is None:
            return 0  # 叶子平衡，高度为0
        left = self.height(root.left)
        if left == -1:
            return -1
        right = self.height(root.right)
        if right == -1:
            return -1
        if abs(left-right)<2:
            return max(left,right) + 1
        else:
            return -1
```

思路 2：使用后序遍历实现分治法的迭代版本

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        s = [[TreeNode(), -1, -1]]
        node, last = root, None
        while len(s) > 1 or node is not None:
            if node is not None:
                s.append([node, -1, -1])
                node = node.left
                if node is None:
                    s[-1][1] = 0
            else:
                peek = s[-1][0]
                if peek.right is not None and last != peek.right:
                    node = peek.right
                else:
                    if peek.right is None:
                        s[-1][2] = 0
                    last, dl, dr = s.pop()
                    if abs(dl - dr) > 1:
                        return False
                    d = max(dl, dr) + 1
                    if s[-1][1] == -1:
                        s[-1][1] = d
                    else:
                        s[-1][2] = d
        
        return True
```

#### 124.[binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/) (hard)

> 给定一个**非空**二叉树，返回其最大路径和。

思路：分治法。最大路径的可能情况：左子树的最大路径，右子树的最大路径，或通过根结点的最大路径。其中通过根结点的最大路径值等于以左子树根结点为端点的最大路径值加以右子树根结点为端点的最大路径值再加上根结点值，这里还要考虑有负值的情况即负值路径需要丢弃不取。

```Python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        
        self.maxPath = float('-inf')
        
        def largest_path_ends_at(node):
            if node is None:
                return float('-inf')
            
            e_l = largest_path_ends_at(node.left)
            e_r = largest_path_ends_at(node.right)
            
            self.maxPath = max(self.maxPath, node.val + max(0, e_l) + max(0, e_r), e_l, e_r)
            
            return node.val + max(e_l, e_r, 0)
        
        largest_path_ends_at(root)
        return self.maxPath
        #---------------------------------------------------------
        # 空节点的最大贡献值等于 0.
        # 非空节点的最大贡献值等于节点值与其子节点中的最大贡献值之和（对于叶节点而言，最大贡献值等于节点值）.root+max(left,right)
        # 该节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值. root+maxleft+maxright     
        self.maxsum = float('-inf')
        def maxnodesum(root):
            if root is None:
                return 0
            # 分别计算子树的最大和
            maxleft = max(maxnodesum(root.left),0)
            maxright = max(maxnodesum(root.right),0)

            # 计算该root子树下最大路径和
            maxrootsum = maxleft + maxright + root.val  

            self.maxsum = max(self.maxsum, maxrootsum)

            return root.val + max(maxleft, maxright)
        maxnodesum(root)
        return self.maxsum
```

#### 236.[lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)(medium)

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

思路：分治法，有左子树的公共祖先或者有右子树的公共祖先，就返回子树的祖先，否则返回根节点

```Python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        if root is None:
            return None
        
        if root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left is not None and right is not None:
            return root
        elif left is not None:
            return left
        elif right is not None:
            return right
        else:
            return None
        # --------------------------------------
        # 1.root is None: return False
        # 2.搜索左子树是否包含二者之一;搜索右子树是否包含二者之一
        # 3.根据公式找最近公共祖先(bool_left_son && bool_right_son) || [(root==q||root==p)&&(bool_left_son||bool_right_son)] 
        # 4.合并结果,左右子树或者自己是否包含指定node
        self.anc = root
        def searchanc(root,p,q):  # 
            # 递归条件
            if root is None:  
                return False  #子树可以有，自己可以是，否则是false
            # 搜索左右子树是否包含目标node
            left = searchanc(root.left,p,q)
            right = searchanc(root.right,p,q)
            # 根据公式找最近公共祖先,其他祖先只能是left或者right==true，且自己不可能是指定node
            if ((left and right) or ((root==q or root==p) and (left or right))):
                self.anc = root
            # 合并结果,左右子树或者自己是否包含指定node
            return left or right or (root==p or root==q)    
        searchanc(root,p,q)
        return self.anc     
```

### BFS 层次应用

#### 103.[binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)(medium)

> 给定一个二叉树，返回其节点值的锯齿形层次遍历。Z 字形遍历

思路：在BFS迭代模板上改用双端队列控制输出顺序

```Python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        
        levels = []
        if root is None:
            return levels
        
        s = collections.deque([root])

        start_from_left = True
        while len(s) > 0:
            levels.append([])
            level_size = len(s)
            
            if start_from_left:
                for _ in range(level_size):
                    node = s.popleft()
                    levels[-1].append(node.val)
                    if node.left is not None:
                        s.append(node.left)
                    if node.right is not None:
                        s.append(node.right)
            else:
                for _ in range(level_size):
                    node = s.pop()
                    levels[-1].append(node.val)
                    if node.right is not None:
                        s.appendleft(node.right)
                    if node.left is not None:
                        s.appendleft(node.left)
            
            start_from_left = not start_from_left
            
        
        return levels
        
        #------------------------------------
        if root is None:
            return []
        level_vals = []  # 记录遍历结果
        stack = [root]  # 记录当前要遍历的层nodes
        left_first = True  # 是否left node先入栈
        while len(stack)>0:
            tmp = []  # 临时记录下一层nodes的入栈结果
            level_vals.append([])
            for _ in range(len(stack)):
                node = stack.pop()
                level_vals[-1].append(node.val)
                if left_first:
                    if node.left is not None:
                        tmp.append(node.left)
                    if node.right is not None:
                        tmp.append(node.right)
                else:
                    if node.right is not None:
                        tmp.append(node.right)
                    if node.left is not None:
                        tmp.append(node.left)
            left_first = not left_first
            stack = tmp
        return level_vals
```

### 二叉搜索树应用

####  98.[validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)(medium)

> 给定一个二叉树，判断其是否是一个有效的二叉搜索树。

思路 1：中序遍历后检查输出是否有序，缺点是如果不平衡无法提前返回结果， 代码略

思路 2：分治法，一个二叉树为合法的二叉搜索树当且仅当左右子树为合法二叉搜索树且根结点值大于右子树最小值小于左子树最大值。缺点是若不用迭代形式实现则无法提前返回，而迭代实现右比较复杂。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None: return True
        
        def valid_min_max(node):
            
            isValid = True
            if node.left is not None:
                l_isValid, l_min, l_max = valid_min_max(node.left)
                isValid = isValid and node.val > l_max
            else:
                l_isValid, l_min = True, node.val

            if node.right is not None:
                r_isValid, r_min, r_max = valid_min_max(node.right)
                isValid = isValid and node.val < r_min
            else:
                r_isValid, r_max = True, node.val

                
            return l_isValid and r_isValid and isValid, l_min, r_max
        
        return valid_min_max(root)[0]
        # ---------------------------
        def isBST(root):
            if root is None:
                return True, float('-inf'),float('inf')   # left_max, right_min
            left,l_max,_ = isBST(root.left)
            right,_,r_min = isBST(root.right)
            curr = True
            if root.left:
                curr = (l_max < root.val)
            if root.right:
                curr = (r_min > root.val)
            root_max = max(l_max,root.val)
            root_min = min(r_min,root.val)
            return curr and left and right,root_max,root_min
        return isBST(root)[0]
```

思路 3：利用二叉搜索树的性质，根结点为左子树的右边界，右子树的左边界，使用先序遍历自顶向下更新左右子树的边界并检查是否合法，迭代版本实现简单且可以提前返回结果。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None:
            return True
        
        s = [(root, float('-inf'), float('inf'))]
        while len(s) > 0:
            node, low, up = s.pop()
            if node.left is not None:
                if node.left.val <= low or node.left.val >= node.val:
                    return False
                s.append((node.left, low, node.val))
            if node.right is not None:
                if node.right.val <= node.val or node.right.val >= up:
                    return False
                s.append((node.right, node.val, up))
        return True
```

#### [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。

思路：如果只是为了完成任务则找到最后一个叶子节点满足插入条件即可。但此题深挖可以涉及到如何插入并维持平衡二叉搜索树的问题，并不适合初学者。

```Python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        
        if root is None:
            return TreeNode(val)
        
        node = root
        while True:
            if val > node.val:
                if node.right is None:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            else:
                if node.left is None:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
```

## 总结

- 掌握二叉树递归与非递归遍历
- 理解 DFS 前序遍历与分治法
- 理解 BFS 层次遍历

## 练习

- [ ] [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
- [ ] [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
- [ ] [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [ ] [binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
- [ ] [binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)
- [ ] [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
