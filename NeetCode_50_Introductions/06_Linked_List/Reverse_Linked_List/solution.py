# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    """
    Reverse a linked list.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Your solution here
    pass

# Helper function to create linked list from list
def create_linked_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Helper function to convert linked list to list
def linked_list_to_list(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

# Test cases
if __name__ == "__main__":
    # Test 1
    head = create_linked_list([1,2,3,4,5])
    result = reverseList(head)
    assert linked_list_to_list(result) == [5,4,3,2,1]
    print("Test 1 passed!")
    
    # Test 2
    head = create_linked_list([1,2])
    result = reverseList(head)
    assert linked_list_to_list(result) == [2,1]
    print("Test 2 passed!")
    
    print("All tests passed!")

