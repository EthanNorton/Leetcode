class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def check_height(node):
            # Base case: empty node
            if not node:
                return 0
            
            # Check left subtree
            left_height = check_height(node.left)
            if left_height == -1:
                return -1
            
            # Check right subtree
            right_height = check_height(node.right)
            if right_height == -1:
                return -1
            
            # Check if current node is balanced
            if abs(left_height - right_height) > 1:
                return -1
            
            # Return height of current node
            return max(left_height, right_height) + 1
        
        # Tree is balanced if check_height doesn't return -1
        return check_height(root) != -1