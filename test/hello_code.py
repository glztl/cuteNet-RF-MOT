from contextlib import nullcontext


class ListNode:
    def __init__(self, val: int):
        self.val = val
        # self.next = None
        self.next = None

""" 列表（动态数组） """
class MyList:

    def __init__(self):
        """构造方法"""
        self._capacity = 10 # 列表容量
        self._arr = [0] * self._capacity    # 数组
        self._size = 0 # 当前列表长度(当前元素数量)
        self._extend_ratio = 2 # 每次扩容的倍数
    def size(self):
        """获取列表长度"""
        return self._size

    def capacity(self):
        """获取列表容量"""
        return self._capacity

    def get(self, index):
        """访问元素
        如果索引越界，则抛出异常"""
        if index < 0 or index >= self._size:
            raise IndexError("索引越界")
        return self._arr[index]

    def set(self, num, index):
        """更新元素"""
        if index < 0 or index >= self._size:
            raise IndexError("索引越界")
        self._arr[index] = num

    def add(self, num):
        """在尾部添加元素，元素超出容量时，出发扩容机制"""
        if self.size() == self.capacity():
            self.extend_capacity()
        self._arr[self._size] = num
        self._size += 1

    def insert(self, num, index):
        """在中间插入元素"""
        if index < 0 or index >= self._size:
            raise IndexError("索引越界")
        # 元素容量超出容量时，出发扩容机制
        if self._size == self.capacity():
            self.extend_capacity()
        # 将索引 index 以及之后的元素都向后移动一位
        for j in range(self._size - 1, index - 1, -1):
            self._arr[j + 1] = self.arr[j]
        self.arr[index] = num
        # 更新元素数量
        self._size += 1

    def remove(self, index):
        if index < 0 or index >= self.size():
            raise IndexError("索引越界")
        num = self._arr[index]
        # 将index之后的元素都向前移动一位
        for j in range (index, self._size - 1):
            self._arr[j] = self._arr[j + 1]
        # 更新元素数量
        self._size -= 1
        # 返回被删除元素
        return sum

    def extend_capacity(self):
        """列表扩容"""
        # 新建一个长度为原来数组 _extend_ratio 倍的新数组，并将原来数组复制到新数组
        self._arr = self._arr + [0] * self.capacity() * (self._extend_ratio - 1)
        # 更新列表容量
        self._capacity = len(self._arr)

    def to_array(self):
        """返回有效列表长度"""
        return self._arr[: self._size]


def insert(n0: ListNode, p: ListNode):
    """在链表节点n0之后插入p"""
    n1 = n0.next
    p.next = n1
    n0.next = p

def remove(n0: ListNode):
    if not n0.next:
        return
    p = n0.next
    n1 = p.next
    n0.next = n1

def access(head, index):
    """访问链表中索引为index的元素"""
    for _ in range(index):
        if not head:
            return None
        head = head.next
    return head

def find(head, target):
    """在链表中查找值为target的首个节点"""
    index = 0
    while head:
        if head.val == target:
            return index
        head = head.next
        index += 1
    return -1