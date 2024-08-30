class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        current_new_list = dummy
        current = head.next  
        sum_val = 0

        while current:
            if current.val == 0:
                
                if sum_val > 0:
                    current_new_list.next = ListNode(sum_val)
                    current_new_list = current_new_list.next
                    sum_val = 0  
            else:
                
                sum_val += current.val

            current = current.next

        return dummy.next  
