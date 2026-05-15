"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count
from providers.registry import ProviderRegistry

from . import dependencies
from .dependencies import get_settings, require_api_key
from .execution_state_store import ExecutionStateStore
from .gateway_model_ids import gateway_model_id, no_thinking_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.execution_state import (
    CheckpointState,
    ExecutionPhase,
    ExecutionStateUpdate,
)
from .models.responses import ModelResponse, ModelsListResponse
from .services import ClaudeProxyService

router = APIRouter()

DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def get_proxy_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ClaudeProxyService:
    """Build the request service for route handlers."""
    execution_state_store = (
        getattr(request.app.state, "execution_state_store", None)
        if settings.enable_execution_state_orchestration
        else None
    )
    return ClaudeProxyService(
        settings,
        provider_getter=lambda provider_type: dependencies.resolve_provider(
            provider_type, app=request.app, settings=settings
        ),
        token_counter=get_token_count,
        execution_state_store=execution_state_store,
    )


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


def _discovered_model_response(model_id: str, *, display_name: str) -> ModelResponse:
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
    )


def _append_unique_model(
    models: list[ModelResponse], seen: set[str], model: ModelResponse
) -> None:
    if model.id in seen:
        return
    seen.add(model.id)
    models.append(model)


def _append_provider_model_variants(
    models: list[ModelResponse],
    seen: set[str],
    provider_model_ref: str,
    *,
    supports_thinking: bool | None = None,
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=f"{provider_model_ref} (no thinking)",
        ),
    )


def _build_models_list_response(
    settings: Settings, provider_registry: ProviderRegistry | None
) -> ModelsListResponse:
    models: list[ModelResponse] = []
    seen: set[str] = set()

    for ref in settings.configured_chat_model_refs():
        supports_thinking = None
        if provider_registry is not None:
            supports_thinking = provider_registry.cached_model_supports_thinking(
                ref.provider_id, ref.model_id
            )
        _append_provider_model_variants(
            models,
            seen,
            ref.model_ref,
            supports_thinking=supports_thinking,
        )

    if provider_registry is not None:
        for model_info in provider_registry.cached_prefixed_model_infos():
            _append_provider_model_variants(
                models,
                seen,
                model_info.model_id,
                supports_thinking=model_info.supports_thinking,
            )

    for model in SUPPORTED_CLAUDE_MODELS:
        _append_unique_model(models, seen, model)

    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


# =============================================================================
# Routes
# =============================================================================
_FORWARDED_HEADER_NAMES = frozenset(
    {
        "anthropic-beta",
        "anthropic-version",
        "x-session-id",
        "x-parent-session-id",
        "anthropic-conversation-id",
    }
)


@router.post("/v1/messages")
async def create_message(
    request: Request,
    request_data: MessagesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create a message response, streaming unless the client requested JSON."""
    forwarded = {
        k: v for k, v in request.headers.items() if k.lower() in _FORWARDED_HEADER_NAMES
    }
    if forwarded:
        request_data.forwarded_headers = forwarded
    if request_data.stream is False:
        return await service.create_message_nonstreaming(request_data)
    return service.create_message(request_data)


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    return service.count_tokens(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    """Respond to compatibility probes for the root endpoint (no auth required)."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint (no auth required)."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/api/bootstrap")
async def bootstrap():
    """Bootstrap endpoint for CLI account info (no auth required)."""
    return {"client_data": None, "additional_model_options": []}


@router.get("/api/claude_cli/bootstrap")
async def bootstrap_legacy():
    """Legacy bootstrap path used by older CLI versions."""
    return {"client_data": None, "additional_model_options": []}


@router.get("/api/claude_code_penguin_mode")
async def penguin_mode():
    """Fast mode availability for Claude Code (penguin mode)."""
    return {"enabled": True}


@router.api_route("/api/claude_code_penguin_mode", methods=["HEAD", "OPTIONS"])
async def probe_penguin_mode():
    """Compatibility probe for penguin mode endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to Claude-compatible clients."""
    registry = getattr(request.app.state, "provider_registry", None)
    provider_registry = registry if isinstance(registry, ProviderRegistry) else None
    return _build_models_list_response(settings, provider_registry)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}


# =============================================================================
# Execution State Orchestration API
# =============================================================================


def _get_execution_state_store(request: Request) -> ExecutionStateStore:
    """Retrieve the execution state store from app state."""
    store = getattr(request.app.state, "execution_state_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Execution state orchestration is disabled. "
                "Set ENABLE_EXECUTION_STATE_ORCHESTRATION=true in .env and restart."
            ),
        )
    return store


@router.get("/v1/execution_state/{session_id}")
async def get_execution_state(
    session_id: str,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Retrieve the current execution state for a session."""
    store = _get_execution_state_store(request)
    state = store.load(session_id)
    if state is None:
        return {"session_id": session_id, "exists": False}
    return {"exists": True, **state.model_dump(mode="json")}


@router.put("/v1/execution_state/{session_id}")
async def update_execution_state(
    session_id: str,
    body: dict,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Update the execution state for a session (partial update)."""
    store = _get_execution_state_store(request)
    patch = ExecutionStateUpdate.model_validate(body)
    updated = store.update(session_id, patch)
    return {"status": "updated", **updated.model_dump(mode="json")}


@router.post("/v1/execution_state/{session_id}/complete_step")
async def complete_step(
    session_id: str,
    body: dict,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Mark a plan step as completed."""
    store = _get_execution_state_store(request)
    step_id = body.get("step_id")
    if not step_id:
        raise HTTPException(status_code=400, detail="step_id is required")

    state = store.load(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    target_step = None
    for step in state.remaining_steps:
        if step.step_id == step_id:
            target_step = step
            break

    if target_step is None:
        raise HTTPException(
            status_code=404, detail=f"Step {step_id} not found in remaining steps"
        )

    updated = store.append_completed_step(session_id, target_step)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to update state")

    return {"status": "completed", **updated.model_dump(mode="json")}


@router.post("/v1/execution_state/{session_id}/set_checkpoint")
async def set_checkpoint(
    session_id: str,
    body: dict,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Create or update a checkpoint for a session."""
    store = _get_execution_state_store(request)
    name = body.get("name", "checkpoint")
    description = body.get("description", "")
    phase_str = body.get("phase")

    state = store.ensure_state(session_id)
    phase_normalized = phase_str.lower().replace("-", "_") if phase_str else None
    phase = (
        ExecutionPhase(phase_normalized)
        if phase_normalized
        else state.implementation_phase
    )

    checkpoint = CheckpointState(
        name=name,
        description=description,
        phase=phase,
    )
    updated = store.update(
        session_id, ExecutionStateUpdate(current_checkpoint=checkpoint)
    )
    logger.info("CHECKPOINT_SET: session={} name={} phase={}", session_id, name, phase)
    return {"status": "checkpoint_set", **updated.model_dump(mode="json")}


@router.post("/v1/execution_state/{session_id}/restore_checkpoint")
async def restore_checkpoint(
    session_id: str,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Restore the active checkpoint for a session."""
    store = _get_execution_state_store(request)
    from .orchestration.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(store)
    updated = manager.restore_checkpoint(session_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="No checkpoint found to restore")
    return {"status": "checkpoint_restored", **updated.model_dump(mode="json")}


@router.post("/v1/execution_state/{session_id}/apply_plan")
async def apply_plan(
    session_id: str,
    body: dict,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Parse and persist an approved execution plan."""

    store = _get_execution_state_store(request)

    from .orchestration.execution_tracker import ExecutionTracker

    tracker = ExecutionTracker(store)

    plan_text = body.get("plan_text")

    if not plan_text:
        raise HTTPException(
            status_code=400,
            detail="plan_text is required",
        )

    updated = tracker.apply_approved_plan(
        session_id=session_id,
        plan_text=plan_text,
        parent_session_id=body.get("parent_session_id")
        or request.headers.get("x-parent-session-id"),
    )

    logger.info(
        "PLAN_APPLIED: session={} steps={}",
        session_id,
        len(updated.remaining_steps),
    )

    return {
        "status": "plan_applied",
        **updated.model_dump(mode="json"),
    }


@router.delete("/v1/execution_state/{session_id}")
async def delete_execution_state(
    session_id: str,
    request: Request,
    _auth=Depends(require_api_key),
):
    """Delete the execution state for a session."""
    store = _get_execution_state_store(request)
    deleted = store.delete(session_id)
    return {"status": "deleted" if deleted else "not_found"}


@router.get("/v1/execution_states")
async def list_execution_states(
    request: Request,
    _auth=Depends(require_api_key),
):
    """List all sessions with persisted execution state."""
    store = _get_execution_state_store(request)
    sessions = store.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}
