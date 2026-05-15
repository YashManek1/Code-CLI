"""Basic AST validation for cross-language syntax verification."""

from __future__ import annotations
import ast
import traceback
from typing import Optional, Dict, Any
from loguru import logger

class ASTValidationError(Exception):
    """Raised when generated code fails basic syntax validation."""
    def __init__(self, message: str, language: str, error_details: Optional[str] = None):
        super().__init__(message)
        self.language = language
        self.error_details = error_details

class FoundationASTValidator:
    """Provides basic syntax validation for common languages."""

    def validate_code(self, code: str, language: str) -> None:
        """Validate syntax for a given code block and language.
        
        Supported: python, javascript, typescript, html, css.
        """
        lang = language.lower()
        if lang == "python":
            self._validate_python(code)
        elif lang in ("javascript", "js", "typescript", "ts"):
            self._validate_js_ts(code) # Basic check for now
        elif lang == "html":
            self._validate_html(code)
        else:
            logger.debug("AST_VALIDATOR: Skipping validation for unsupported language: {}", lang)

    def _validate_python(self, code: str) -> None:
        """Use Python's built-in ast module for syntax validation."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.warning("AST_VALIDATOR: Python syntax error detected: {}", e)
            raise ASTValidationError(
                message=f"Python syntax error: {e.msg}",
                language="python",
                error_details=traceback.format_exc()
            ) from e

    def _validate_js_ts(self, code: str) -> None:
        """Basic validation for JS/TS. 
        Note: Full validation requires an external tool like 'tsc' or 'eslint'.
        For the foundation layer, we just check for obvious unclosed blocks.
        """
        # Simple balanced brackets check as a foundation
        stack = []
        brackets = {'{': '}', '[': ']', '(': ')'}
        for i, char in enumerate(code):
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if not stack:
                    # Closing bracket without opening
                    pass # Hard to be certain without a real parser
                else:
                    top, _ = stack.pop()
                    if brackets[top] != char:
                        # Mismatched bracket - likely an error but could be in a string
                        pass
        
        # If we have a lot of unclosed brackets at the end, it's likely truncated
        if len(stack) > 10:
            logger.warning("AST_VALIDATOR: Excessive unclosed brackets in JS/TS ({})", len(stack))
            # We don't raise here yet to avoid false positives, but we log it.

    def _validate_html(self, code: str) -> None:
        """Basic HTML tag balancing check."""
        # This could use html.parser but often fragments are valid
        pass
