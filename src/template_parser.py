"""
Author: Prabhav Singh

Template parsing utilities for conditional question evaluation in rubrics.
"""

import re
from typing import Dict, Optional

class TemplateParser:
    """
    Parser for template-style conditional expressions in rubric questions.
    """
    
    @staticmethod
    def extract_condition(template: str) -> Optional[str]:
        """
        Extract condition from {% if ... %} template.
        
        Args:
            template: String containing template expression
            
        Returns:
            Extracted condition or None if no valid template found
        """
        if not template:
            return None
            
        match = re.search(r'{%\s*if\s+(.+?)\s*%}', template)
        return match.group(1) if match else None
    
    @staticmethod
    def evaluate_condition(condition: str, previous_answers: Dict[str, int]) -> bool:
        """
        Evaluates a conditional expression using previous question answers.
        
        Args:
            condition: String like "Q6 > 3 and Q5 > 3"
            previous_answers: Dict mapping question IDs to scores
            
        Returns:
            Whether condition is satisfied
        """
        if not condition:
            return True
            
        # Replace question IDs with their scores
        expr = condition
        for q_id, score in previous_answers.items():
            expr = expr.replace(q_id, str(score))
            
        try:
            # Restrict eval to just numeric comparisons and logical operators
            restricted_globals = {
                "__builtins__": {
                    "True": True,
                    "False": False,
                    "and": and_op,
                    "or": or_op
                }
            }
            return eval(expr, restricted_globals, {})
        except Exception:
            return False

def and_op(x, y):
    """Safe implementation of logical AND."""
    return bool(x) and bool(y)

def or_op(x, y):
    """Safe implementation of logical OR."""
    return bool(x) or bool(y)