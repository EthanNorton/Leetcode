
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    # If both nodes are None, trees are identical at this point
    if not p and not q:
        return True
    
    # If one node is None and other isn't, trees are different
    if not p or not q:
        return False
    
    # Check if current nodes have same value and recursively check left and right subtrees
    return (p.val == q.val and 
            self.isSameTree(p.left, q.left) and 
            self.isSameTree(p.right, q.right))