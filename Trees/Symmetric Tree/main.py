def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True
        
    def isMirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        # If both nodes are None, this part is symmetric
        if not left and not right:
            return True
            
        # If one node is None and other isn't, not symmetric
        if not left or not right:
            return False
            
        # Check if values match and subtrees are mirrors of each other
        return (left.val == right.val and 
                isMirror(left.left, right.right) and  # outer pairs
                isMirror(left.right, right.left))     # inner pairs
    
    return isMirror(root.left, root.right)