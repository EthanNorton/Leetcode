class Solution:
    def calPoints(self, operations: List[str]) -> int:
        valid_scores = [] 
        
        for op in operations:
            if op == '+':
                valid_scores.append(valid_scores[-1] + valid_scores[-2])
            elif op == 'D':
                valid_scores.append(2 * valid_scores[-1])
            elif op == 'C':
                valid_scores.pop()
            else:
                valid_scores.append(int(op))
        
        return sum(valid_scores)
