# 链表

## 基本技能

链表相关的核心点

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 常见题型

### 83.[remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)(easy)

> 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

```Python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 两个指针，如果快指针与慢指针val同，快指针再进一步，慢指针.next指向快指针
        if head is None:
            return head
        quick, slow = head.next, head

        while (quick is not None):
            # 如果重复，quick指向下一个node，slow连接到quick
            if quick.val == slow.val:
                quick = quick.next
                slow.next = quick

            else:  # 如果不重复，slow走到quick，quick指向下一个node
                slow = quick
                quick = quick.next
        return head

        # -------------一个指针--------------
        if head is None:
            return head
        curr = head
        while (curr.next is not None):
            if curr.next.val == curr.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head
```

### 82.[remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)(medium)

> 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中   没有重复出现的数字。

思路：链表头结点可能被删除，所以用 dummy node 辅助删除

```Python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        
        if head is None:
            return head
        
        dummy = ListNode(next=head)
        
        current, peek = dummy, head
        find_dup = False
        while peek.next is not None:
            if peek.next.val == peek.val:
                find_dup = True
                peek.next = peek.next.next
            else:
                if find_dup:
                    find_dup = False
                    current.next = current.next.next
                else:
                    current = current.next
                peek = peek.next
        
        if find_dup:
            current.next = current.next.next
        
        return dummy.next
	# --------------------
        if head is None:
            return head
        dummpy = ListNode(0)
        dummpy.next = head

        last = dummpy  # dummpy链表中不重复的最后一个node
        quick = head  # 指向待查链表的第一个node
        while quick:
	    # 如果有重复则找到最后一个重复的node
            while quick.next and quick.val == quick.next.val:
                quick = quick.next  # 一直找到不重复的那个node的前一个
	    # 如果待查链表的第一个node与其后没有重复
            if last.next == quick:  # quick此时指向不重复的尾node
                last = last.next  # last后移一位，指向新的不重复尾node
            else:  # 如果中间有重复的，跳过重复node
                last.next = quick.next  # quick此时指向被跳过的最后一个重复的node
            quick = quick.next  # quick指向待查链表的第一个node
        return dummpy.next
```

注意点
• A->B->C 删除 B，A.next = C
• 删除用一个 Dummy Node 节点辅助（允许头节点可变）
• 访问 X.next 、X.value 一定要保证 X != nil

### 206.[reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)(easy)

> 反转一个单链表。

- 思路：将当前结点放置到头结点

```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        
        new_head = None  # 新链表的头结点
        old_tail = head  # 待翻转的链表的‘第一个’node
        while old_tail:  # 如果待反转链表还有node
            tmp = old_tail.next  # 暂存之后的链表
            old_tail.next = new_head  # 反转当前node方向
            new_head = old_tail  # 更新新链表头node
            old_tail = tmp  # 指向下一个待处理node
        return new_head
```
- Recursive method is tricky
```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        
        if head is None or head.next is None:
            return head
        
        rev_next = self.reverseList(head.next)  # 返回最后一个node作为新head
        head.next.next = head  # 将下一个node的next指向自己，两个node间形成互相指向
        head.next = None  # 反转当前node的指向，指向None，如果是最后一个node就正好指向None，其他的会再指向前一个ndoe
        
        return rev_next 
```

### 92.[reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)(medium)

> 反转从位置  *m*  到  *n*  的链表。请使用一趟扫描完成反转。

思路：先找到 m 处, 再反转 n - m 次即可

```Python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:

        if head is None:
            return head
        n = n - m  # 反转数量
        dummy = ListNode(0)
        dummy.next = head
        curr = dummy
        while m > 1:
            # find node at m-1
            curr = curr.next  # 1->2->3->4->5->6->None
            m -= 1
        start = curr.next  # node m
        # 依次把(m,n]node挪到curr.next上
        # start一直指向node m， start.next一直指向下一个需要挪到curr.next的node
        while n > 0:  
            tmp = start.next  # tmp保留当前需要挪到curr.next的node
            start.next = tmp.next  # start.next指向下一次循环需要挪到curr.next的node
            # 当前node挪到curr.next之后
            tmp.next = curr.next  
            curr.next = tmp
            n -= 1
        return dummy.next
```

### 21.[merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)(easy)

> 将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

思路：通过 dummy node 链表，连接各个元素

```Python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)  # 设置头结点
        curr = dummy

        while l1 and l2:  # 如果两个都有node
            if l1.val <= l2.val:
                curr.next = l1 # 并入
                l1 = l1.next # 指向下一个待处理node
            else:
                curr.next = l2  # 并入
                l2 = l2.next  # 指向下一个待处理node
            curr = curr.next  # 指向合并链表的tail node
        curr.next = l1 if l2 is None else l2  # 如果其中一个链表并完，就直接把另一个并入
        return dummy.next
```

### 86.[partition-list](https://leetcode-cn.com/problems/partition-list/)(medium)

> 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于  *x*  的节点都在大于或等于  *x*  的节点之前。

思路：将大于 x 的节点，放到另外一个链表，最后连接这两个链表

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 分为两个链表再拼接
        low = ListNode(0)
        high = ListNode(0)
        low_curr = low
        high_curr = high

        while head:
            if head.val < x:  # 小于
                low_curr.next = head
                low_curr = low_curr.next
            else:  #大于等于
                high_curr.next = head
                high_curr = high_curr.next
            head = head.next
        low_curr.next = high.next
        high_curr.next = None  # 尾部是None
        return low.next
```

哑巴节点使用场景

> 当头节点不确定的时候，使用哑巴节点

### 148.[sort-list](https://leetcode-cn.com/problems/sort-list/)(medium)

> 在  *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。

思路：归并排序，slow-fast找中点

```Python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 归并排序，空间复杂度O(n)
        def findmid(head):
            # 快慢指针找中点, O(nlogn * 1/2)
            slow, fast = head, head.next
            while fast is not None and fast.next is not None:
                slow = slow.next  # 走一步
                fast = fast.next.next  # 走两步
            return slow

        def merge(h1, h2):  # 
            dummy = ListNode(0)
            curr = dummy
            while h1 and h2:
                if h1.val <= h2.val:
                    curr.next = h1
                    h1 = h1.next
                else:
                    curr.next = h2
                    h2 = h2.next
                curr = curr.next  # O(n)
            curr.next = h1 if h2 is None else h2
            return dummy.next
                
        # 递归条件
        if head is None or head.next is None:
            return head
        # 处理分支，分成两个链表
        mid = findmid(head)
        h2 = mid.next
        mid.next = None
        # 合并结果
        return merge(self.sortList(head), self.sortList(h2))  # O(n) + 2T(n/2)
```

注意点

- 快慢指针 判断 fast 及 fast.Next 是否为 nil 值，单node不用找mid
- 递归 mergeSort 需要断开中间节点，补一个None
- 递归返回条件为 head 为 nil 或者 head.Next 为 nil，单node不用找mid

### 143.[reorder-list](https://leetcode-cn.com/problems/reorder-list/)(medium)

> 给定一个单链表  *L*：*L*→*L*→…→*L\_\_n*→*L*
> 将其重新排列后变为： *L*→*L\_\_n*→*L*→*L\_\_n*→*L*→*L\_\_n*→…

思路：找到中点断开，翻转后面部分，然后合并前后两个链表

```Python
class Solution:
    
    def reverseList(self, head: ListNode) -> ListNode:
        
        prev, curr = None, head
        
        while curr is not None:
            curr.next, prev, curr = prev, curr, curr.next
            
        return prev
    
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head is None or head.next is None or head.next.next is None:
            return
        
        slow, fast = head, head.next
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
        
        h, m = head, slow.next
        slow.next = None
        
        m = self.reverseList(m)
        
        while h is not None and m is not None:
            p = m.next
            m.next = h.next
            h.next = m
            h = h.next.next
            m = p
            
        return
    # ----------------------------------------
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # 先都入栈，统计数量，找到中心点
        # 出栈得到尾部node，与head拼接
        # 中心点.next = None
        if head is None:
            return head
        stack = []
        p = head
        while p:
            stack.append(p)
            p = p.next
        cnt = len(stack)
        # 1-2-3-4-5 => 1-5-2-4-3
        # 1-2-3-4-5-6 => 1-6-2-5-3-4
        # 可知，后半部即弹出的应该是小于等于一半的数量
        mid = cnt//2
        p = head
        while mid:
            tmp = stack.pop()  # 尾node
            tmp.next = p.next  # 插入到前半截
            p.next = tmp
            p = tmp.next  # 指向前半截的下一个待插入节点
            mid -= 1
        # 奇数个node，最后p指向前半截最后的node
        # 偶数个node，最后p指向tmp即后半截的第一个node，因为tmp.next == tmp 
        p.next = None  
```

### 141.[linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)(easy)

> 给定一个链表，判断链表中是否有环。

思路1：Hash Table 记录所有结点判断重复，空间复杂度 O(n) 非最优，时间复杂度 O(n) 但必然需要 n 次循环
思路2：快慢指针，快慢指针相同则有环，证明：如果有环每走一步快慢指针距离会减 1，空间复杂度 O(1) 最优，时间复杂度 O(n) 但循环次数小于等于 n
![fast_slow_linked_list](https://img.fuiboom.com/img/fast_slow_linked_list.png)

```Python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        # 快慢指针,空间O(1)，时间O(n)
        q1 = head
        q2 = head
        while q2 and q2.next:
            q2 = q2.next.next
            q1 = q1.next
            if q2 == q1:
                return True
        return False
```

### 142.[linked-list-cycle-ii](https://leetcode-cn.com/problems/linked-list-cycle-ii/)(medium)

> 给定一个链表，返回链表开始入环的第一个节点。  如果链表无环，则返回  `null`。

思路：快慢指针，快慢相遇之后，慢指针回到头，快慢指针步调一致一起移动，相遇点即为入环点
![cycled_linked_list](https://img.fuiboom.com/img/cycled_linked_list.png)

```Python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        # 首先假设头结点到环入口的距离为L，环的长度为C，而且L<=C，那么：
        # 1、当慢指针到达环入口时，快指针比它多走L，即快指针需再多走C-L才能相遇；
        # 2、由于v(快)-v(慢)=1，故相遇时慢指针又走了C-L，此时它与环起点距离为C-(C-L)=L；
        # 至于L>C的情况，其实可看作先将L不断减去C直到L<=C ，就相当于快指针先在环内多跑了几圈，并不影响结果。
        
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                fast = head  # 此时slow距离环口L个node  
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None  # 无环
```

坑点

- 指针比较时直接比较对象，不要用值比较，链表中有可能存在重复值情况
- 第一次相交后，快指针需要从下一个节点开始和头指针一起匀速移动


注意，此题中使用 slow = fast = head 是为了保证最后找环起始点时移动步数相同，但是作为找中点使用时**一般用 fast=head.Next 较多**，因为这样可以知道中点的上一个节点，可以用来删除等操作。

- fast 如果初始化为 head.Next 则中点在 slow.Next
- fast 初始化为 head,则中点在 slow

### 234.[palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)(easy)

> 请判断一个链表是否为回文链表。

- 思路：O(1) 空间复杂度的解法需要破坏原链表（找中点 -> 反转后半个list -> 判断回文），在实际应用中往往还需要复原（后半个list再反转一次后拼接），操作比较复杂，这里给出更工程化的做法

```Python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        
        s = []
        slow = fast = head
        while fast is not None and fast.next is not None:
            s.append(slow.val)
            slow = slow.next
            fast = fast.next.next
        
        if fast is not None:
            slow = slow.next
        
        while len(s) > 0:
            if slow.val != s.pop():
                return False
            slow = slow.next
            
        return True
	# -----------------------------------
        # all elem are pushed in stack
        # compare each node.val
        stack = []
        p = head
        while p:  # push
            stack.append(p)
            p = p.next
        cnt = len(stack)
        mid = cnt // 2
        p = head
        while mid:
            tmp = stack.pop()  # pop
            if tmp.val != p.val:  # compare
                return False
            p = p.next  # next node
            mid -= 1 
        return True  # include head is None
```

### 138.[copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)(medium)

> 给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
> 要求返回这个链表的 深拷贝。

- 思路1：hash table 存储 random 指针的连接关系

```Python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        if head is None:
            return None
        
        parent = collections.defaultdict(list)
        
        out = Node(0)
        o, n = head, out
        while o is not None:
            n.next = Node(o.val)
            n = n.next
            if o.random is not None:
                parent[o.random].append(n)
            o = o.next
            
        o, n = head, out.next
        while o is not None:
            if o in parent:
                for p in parent[o]:
                    p.random = n
            o = o.next
            n = n.next
        
        return out.next
```

- 思路2：复制结点跟在原结点后面，间接维护连接关系，优化空间复杂度，建立好新 list 的 random 链接后分离

```Python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head is None:
            return None
        p = head
        while p:  # copy after old
            p.next = Node(p.val, p.next)
            p = p.next.next
        p = head
        while p:
            if p.random is not None:
                p.next.random = p.random.next
            p = p.next.next
        
        # segment
        new_head = head.next  # must
        old, new = head, head.next
        while new.next:
            old.next = new.next
            new.next = new.next.next

            old = old.next
            new = new.next
        old.next = None
        return new_head
```

## 总结

链表必须要掌握的一些点，通过下面练习题，基本大部分的链表类的题目都是手到擒来~

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 练习

- [ ] 83.[remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)(easy)
- [ ] 82.[remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)(medium)
- [ ] 206.[reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)(easy)
- [ ] 92.[reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)(medium)
- [ ] 21.[merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)(easy)
- [ ] 86.[partition-list](https://leetcode-cn.com/problems/partition-list/)(medium)
- [ ] 148.[sort-list](https://leetcode-cn.com/problems/sort-list/)(medium)
- [ ] 143.[reorder-list](https://leetcode-cn.com/problems/reorder-list/)(medium)
- [ ] 141.[linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)(easy)
- [ ] 142.[linked-list-cycle-ii](https://leetcode-cn.com/problems/https://leetcode-cn.com/problems/linked-list-cycle-ii/)()medium
- [ ] 234.[palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)(easy)
- [ ] 138.[copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)(medium)
