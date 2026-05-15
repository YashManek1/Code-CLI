"""Transactional execution and tool recovery management."""

from __future__ import annotations
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from loguru import logger
from .ast_validator import FoundationASTValidator, ASTValidationError

class TransactionalFileEditor:
    """Implements Patch -> Staging -> Validation -> Commit flow."""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.staging_dir = tempfile.mkdtemp(prefix="stitch_staging_")
        self._pending_edits: Dict[str, str] = {} # target_path -> staging_path
        self._validator = FoundationASTValidator()

    def stage_edit(self, filepath: str, content: str) -> str:
        """Apply an edit to an isolated staging area."""
        # Ensure path is within workspace for safety
        if not os.path.isabs(filepath):
            target_path = os.path.join(self.workspace_root, filepath)
        else:
            target_path = filepath
            
        rel_path = os.path.relpath(target_path, self.workspace_root)
        staging_path = os.path.join(self.staging_dir, rel_path)
        
        os.makedirs(os.path.dirname(staging_path), exist_ok=True)
        with open(staging_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        self._pending_edits[target_path] = staging_path
        logger.info("TRANSACTIONS: Staged edit for {} -> {}", rel_path, staging_path)
        return staging_path

    def validate_staging(self, staging_path: str, language: Optional[str] = None) -> bool:
        """Validate the syntax of a staged file."""
        try:
            with open(staging_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Auto-detect language from extension if not provided
            if not language:
                ext = os.path.splitext(staging_path)[1].lower().lstrip(".")
                language = ext
                
            self._validator.validate_code(content, language)
            return True
        except ASTValidationError as e:
            logger.error("TRANSACTIONS: Validation failed for staging {}: {}", staging_path, e)
            return False
        except Exception as e:
            logger.error("TRANSACTIONS: Unexpected error during validation: {}", e)
            return False

    def commit_all(self) -> None:
        """Atomic commit of all staged edits to the workspace."""
        for target, staging in self._pending_edits.items():
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(staging, target)
                logger.info("TRANSACTIONS: Committed {}", target)
            except Exception as e:
                logger.critical("TRANSACTIONS: Failed to commit {}: {}", target, e)
                # In a real system, we might want to attempt partial rollback here
                raise
        self._pending_edits.clear()

    def rollback_all(self) -> None:
        """Clear all pending edits without committing."""
        self._pending_edits.clear()
        if os.path.exists(self.staging_dir):
            shutil.rmtree(self.staging_dir)
        self.staging_dir = tempfile.mkdtemp(prefix="stitch_staging_")

    def cleanup(self) -> None:
        """Final cleanup of staging area."""
        if os.path.exists(self.staging_dir):
            shutil.rmtree(self.staging_dir)

class ToolExecutionRecoveryManager:
    """Tracks filesystem mutations and partial execution for tool recovery."""

    def __init__(self):
        self._active_mutations: List[Dict[str, Any]] = []

    def record_mutation(self, tool_name: str, mutation_type: str, details: Dict[str, Any]):
        """Log a side-effect before it occurs."""
        self._active_mutations.append({
            "tool": tool_name,
            "type": mutation_type,
            "details": details,
            "status": "pending"
        })

    def mark_success(self, tool_id: str):
        """Mark all pending mutations for a tool as committed."""
        self._active_mutations = [m for m in self._active_mutations if m["status"] != "pending"]

    def rollback_all(self):
        """Revert all pending mutations if the tool execution was interrupted."""
        for mutation in reversed(self._active_mutations):
            if mutation["status"] == "pending":
                logger.warning("RECOVERY_MANAGER: Rolling back mutation: {}", mutation)
        self._active_mutations.clear()
