"""Interaction lifecycle & Server-Sent Events (SSE) streaming router for AI Brain (M5.5)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from ai_brain.agents.models import AgentExecutionRequest
from ai_brain.agents.simulated.coding import CodingAgent
from ai_brain.agents.simulated.research import ResearchAgent
from ai_brain.agents.simulated.review import ReviewAgent
from ai_brain.auth import AuthenticatedIdentity, Permission
from ai_brain.loop.exceptions import InputValidationError, ResponseValidationError
from ai_brain.loop.models import (
    AgentInteractionInput,
    AgentResponseInput,
    InteractionRecord,
    InteractionStatus,
)
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    get_container,
    require_permission,
)
from ai_brain.server.idempotency import (
    IdempotencyManager,
    IdempotencyReservation,
    IdempotencyStatus,
    StaleFencingTokenError,
)
from ai_brain.server.schemas import (
    ContextItemDTO,
    ContextPackageDTO,
    ErrorResponse,
    InteractionCompleteRequest,
    InteractionCompleteResponse,
    InteractionPrepareRequest,
    InteractionPrepareResponse,
    InteractionRecordDTO,
)

logger = logging.getLogger("ai_brain.server.interactions")

router = APIRouter(
    prefix="/api/v1/projects/{project_id}/interactions",
    tags=["Interactions"],
)


@router.post(
    "/prepare",
    response_model=InteractionPrepareResponse,
    status_code=status.HTTP_200_OK,
    summary="Prepare an agent interaction and deliver retrieved context",
    responses={
        400: {"model": ErrorResponse, "description": "Validation or bad request error"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authentication"},
        403: {"model": ErrorResponse, "description": "Permission denied for project"},
        409: {"model": ErrorResponse, "description": "Idempotency conflict"},
    },
)
async def prepare_interaction(
    project_id: str,
    payload: InteractionPrepareRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> InteractionPrepareResponse:
    """Prepares an interaction context package with relevant memories within the token budget."""
    interaction_id = f"int_{uuid.uuid4().hex[:12]}"

    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:8]}"

    # Check for existing idempotency key match
    if payload.idempotency_key:
        try:
            existing = container.interaction_store.get_by_idempotency_key(payload.idempotency_key)
        except TypeError:
            existing = container.interaction_store.get_by_idempotency_key(payload.idempotency_key, project_id=project_id)
        if existing and existing.project_id == project_id:
            # Check for conflict: different input payload
            input_content = existing.input.content if existing.input else ""
            if input_content.strip() != payload.input_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Idempotency key '{payload.idempotency_key}' was previously used with different input text.",
                )
            output_content = (existing.response.content if existing.response else None) or ""
            tokens_val = existing.context_result.context_package.tokens_used if existing.context_result and existing.context_result.context_package else 0
            return InteractionPrepareResponse(
                interaction_id=existing.interaction_id,
                project_id=existing.project_id,
                session_id=existing.session_id,
                agent_id=existing.agent_id,
                status=existing.status.value if hasattr(existing.status, "value") else str(existing.status),
                tokens_used=tokens_val,
                prompt_context_markdown=output_content,
                is_cached=True,
            )

    agent_input = AgentInteractionInput(
        interaction_id=interaction_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=payload.agent_id or identity.identity_id,
        content=payload.input_text,
        idempotency_key=payload.idempotency_key,
        token_budget=payload.max_context_tokens,
        query=payload.search_query,
        metadata=payload.metadata or {},
    )

    try:
        prep_result = container.brain_loop.prepare_context(agent_input)
    except InputValidationError as e:
        if "reused with mismatched input payload" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {e}",
        )
    except Exception as e:
        logger.error("Failed to prepare interaction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context preparation failed: {e}",
        )

    # Format context package DTO
    ctx_pkg = None
    if prep_result.context_package:
        items_dto = [
            ContextItemDTO(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type),
                status=item.status.value if hasattr(item.status, "value") else str(item.status),
                relevance_score=item.relevance_score,
                importance_score=item.priority_score,
                source_agents=[],
            )
            for item in prep_result.context_package.items
        ]
        ctx_pkg = ContextPackageDTO(
            project_id=project_id,
            query=prep_result.context_package.query,
            token_budget=prep_result.context_package.token_budget,
            tokens_used=prep_result.context_package.tokens_used,
            items=items_dto,
            disputed_omitted=len(prep_result.context_package.excluded_items),
        )

    tokens_used = prep_result.context_package.tokens_used if prep_result.context_package else 0
    markdown_text = (prep_result.context_package.rendered_markdown or prep_result.context_package.rendered_text) if prep_result.context_package else ""

    return InteractionPrepareResponse(
        interaction_id=prep_result.interaction_id,
        project_id=project_id,
        session_id=prep_result.session_id,
        agent_id=payload.agent_id or identity.identity_id,
        status="PREPARED",
        tokens_used=tokens_used,
        prompt_context_markdown=markdown_text,
        context_package=ctx_pkg,
        is_cached=False,
    )


@router.post(
    "/{interaction_id}/complete",
    response_model=InteractionCompleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Complete an interaction with agent output and extract knowledge",
    responses={
        400: {"model": ErrorResponse, "description": "Validation or lifecycle state error"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authentication"},
        403: {"model": ErrorResponse, "description": "Permission denied for project"},
        404: {"model": ErrorResponse, "description": "Interaction not found"},
    },
)
async def complete_interaction(
    project_id: str,
    interaction_id: str,
    payload: InteractionCompleteRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_EXECUTE)),
    container: BrainServiceContainer = Depends(get_container),
) -> InteractionCompleteResponse:
    """Completes interaction execution, triggers memory extraction, updates graphs, and writes events."""
    existing = container.brain_loop.get_interaction(interaction_id)
    if not existing or existing.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interaction '{interaction_id}' not found in project '{project_id}'.",
        )

    response_input = AgentResponseInput(
        interaction_id=interaction_id,
        project_id=project_id,
        session_id=payload.session_id or existing.session_id,
        agent_id=payload.agent_id or existing.agent_id or identity.identity_id,
        content=payload.agent_response,
        metadata=payload.metrics or {},
    )

    try:
        completion_result = container.brain_loop.complete_interaction(response_input)
    except Exception as e:
        logger.error("Failed to complete interaction '%s': %s", interaction_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interaction completion failed: {e}",
        )

    # Persist extracted memories to container memory store if configured
    if hasattr(container, "memory_store") and container.memory_store is not None:
        for m in completion_result.extracted_memories:
            try:
                container.memory_store.store(m)
            except Exception as e:
                logger.warning("Failed to store extracted memory '%s' in memory_store: %s", m.memory_id, e)

    mem_created = [
        {"memory_id": m.memory_id, "content": m.content, "version": m.version}
        for m in completion_result.extracted_memories
    ]
    mem_updated = [
        {"memory_id": u.current_memory.memory_id, "content": u.current_memory.content, "version": u.current_memory.version}
        for u in completion_result.update_results
        if u.current_memory and u.superseded
    ]
    conflicts = [
        {"conflict_id": getattr(c, "conflict_id", str(i)), "type": getattr(c, "conflict_type", "CONTRADICTION")}
        for i, c in enumerate(completion_result.conflicts_detected)
    ]

    return InteractionCompleteResponse(
        interaction_id=interaction_id,
        status="COMPLETED",
        memories_created=mem_created,
        memories_updated=mem_updated,
        conflicts_detected=conflicts,
        graph_nodes_created=completion_result.graph_nodes_integrated,
        vectors_indexed=completion_result.vectors_indexed,
        execution_time_ms=0.0,
    )


@router.get(
    "/{interaction_id}",
    response_model=InteractionRecordDTO,
    summary="Retrieve persistent interaction record and results",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
    },
)
async def get_interaction(
    project_id: str,
    interaction_id: str,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> InteractionRecordDTO:
    """Retrieve an interaction record by identifier."""
    interaction = container.brain_loop.get_interaction(interaction_id)
    if not interaction or interaction.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interaction '{interaction_id}' not found in project '{project_id}'.",
        )

    inp_text = interaction.input.content if interaction.input else ""
    out_text = interaction.response.content if interaction.response else None
    return InteractionRecordDTO(
        interaction_id=interaction.interaction_id,
        project_id=interaction.project_id,
        session_id=interaction.session_id,
        agent_id=interaction.agent_id,
        idempotency_key=interaction.idempotency_key,
        status=interaction.status.value if hasattr(interaction.status, "value") else str(interaction.status),
        input_text=inp_text,
        output_text=out_text,
        decision_records=[],
        metrics={},
        created_at=interaction.created_at.isoformat() if interaction.created_at else datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "",
    response_model=List[InteractionRecordDTO],
    summary="List interactions within project",
)
async def list_interactions(
    project_id: str,
    session_id: Optional[str] = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> List[InteractionRecordDTO]:
    """List historical interactions in project matching filters."""
    stat = InteractionStatus(status_filter) if status_filter in InteractionStatus.__members__ else None
    interactions = container.brain_loop.list_interactions(
        project_id=project_id,
        session_id=session_id,
        status=stat,
    )
    if agent_id:
        interactions = [it for it in interactions if it.agent_id == agent_id]
    if limit and limit > 0:
        interactions = interactions[:limit]

    return [
        InteractionRecordDTO(
            interaction_id=it.interaction_id,
            project_id=it.project_id,
            session_id=it.session_id,
            agent_id=it.agent_id,
            idempotency_key=it.idempotency_key,
            status=it.status.value if hasattr(it.status, "value") else str(it.status),
            input_text=it.input.content if it.input else "",
            output_text=it.response.content if it.response else None,
            decision_records=[],
            metrics={},
            created_at=it.created_at.isoformat() if it.created_at else datetime.now(timezone.utc).isoformat(),
        )
        for it in interactions
    ]


# =============================================================================
# Server-Sent Events (SSE) Streaming Endpoint
# =============================================================================
@router.post(
    "/stream",
    summary="Execute interaction with real-time Server-Sent Events (SSE) progress and response streaming",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Real-time SSE event stream (events: prepare, context_delivered, agent_chunk, completion, done, error)",
        },
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        409: {"model": ErrorResponse, "description": "Conflict: Concurrent request or payload mismatch"},
    },
)
async def stream_interaction(
    project_id: str,
    payload: InteractionPrepareRequest,
    request: Request,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_EXECUTE)),
    container: BrainServiceContainer = Depends(get_container),
) -> StreamingResponse:
    """Streams interaction execution via SSE with atomic reservation, disconnect recovery, and concurrency safety."""
    # 1. Atomically reserve idempotency key before execution begins
    reservation: Optional[IdempotencyReservation] = None
    interaction_id = f"int_sse_{uuid.uuid4().hex[:12]}"
    existing_record: Optional[InteractionRecord] = None

    if payload.idempotency_key:
        idempotency_mgr: IdempotencyManager = container.idempotency_manager
        reservation = idempotency_mgr.reserve(
            project_id=project_id,
            idempotency_key=payload.idempotency_key,
            input_content=payload.input_text,
            interaction_store=container.interaction_store,
        )
        existing_record = reservation.existing_record
        interaction_id = existing_record.interaction_id if existing_record else reservation.interaction_id

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Reconnection optimization: if already completed, replay cached events without duplicate writes
            if existing_record and existing_record.status in (InteractionStatus.COMPLETED, InteractionStatus.PARTIAL_SUCCESS):
                yield f"event: prepare\ndata: {json.dumps({'interaction_id': interaction_id, 'status': 'PREPARING', 'replayed': True, 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
                if await request.is_disconnected():
                    return

                tok_used = (
                    existing_record.context_result.context_package.tokens_used
                    if existing_record.context_result and existing_record.context_result.context_package
                    else 0
                )
                items_cnt = (
                    len(existing_record.context_result.context_package.items)
                    if existing_record.context_result and existing_record.context_result.context_package
                    else 0
                )
                yield f"event: context_delivered\ndata: {json.dumps({'interaction_id': interaction_id, 'tokens_used': tok_used, 'memories_count': items_cnt, 'prompt_context': 'Replayed cached context', 'replayed': True})}\n\n"
                if await request.is_disconnected():
                    return

                out_text = existing_record.response.content if existing_record.response else ""
                yield f"event: agent_chunk\ndata: {json.dumps({'interaction_id': interaction_id, 'index': 0, 'chunk': out_text, 'replayed': True})}\n\n"
                if await request.is_disconnected():
                    return

                mem_cnt = len(existing_record.completion_result.extracted_memories) if existing_record.completion_result else 0
                nodes_cnt = existing_record.completion_result.graph_nodes_integrated if existing_record.completion_result else 0
                vec_cnt = existing_record.completion_result.vectors_indexed if existing_record.completion_result else 0
                yield f"event: completion\ndata: {json.dumps({'interaction_id': interaction_id, 'status': 'COMPLETED', 'memories_extracted': mem_cnt, 'graph_nodes_created': nodes_cnt, 'vectors_indexed': vec_cnt, 'replayed': True})}\n\n"
                yield f"event: done\ndata: {json.dumps({'interaction_id': interaction_id, 'status': 'FINISHED', 'replayed': True})}\n\n"
                return

            def _mark_interrupted(reason: str) -> None:
                logger.info("Marking interaction '%s' as INTERRUPTED: %s", interaction_id, reason)
                if reservation is not None:
                    try:
                        reservation.mark_interrupted(reason)
                    except Exception as e:
                        logger.warning("Failed to mark reservation interrupted: %s", e)
                rec = container.brain_loop.interaction_store.get(interaction_id)
                if rec and rec.status not in (InteractionStatus.COMPLETED, InteractionStatus.PARTIAL_SUCCESS):
                    failed_rec = InteractionRecord(
                        interaction_id=rec.interaction_id,
                        idempotency_key=rec.idempotency_key,
                        project_id=rec.project_id,
                        agent_id=rec.agent_id,
                        session_id=rec.session_id,
                        status=InteractionStatus.FAILED,
                        current_stage="INTERRUPTED",
                        stages_completed=rec.stages_completed,
                        input=rec.input,
                        input_event_id=rec.input_event_id,
                        context_event_id=rec.context_event_id,
                        context_result=rec.context_result,
                        error_message=f"INTERRUPTED: {reason}",
                        created_at=rec.created_at,
                        updated_at=datetime.now(timezone.utc),
                    )
                    container.brain_loop.interaction_store.save(failed_rec)

            # 1. Emit prepare event
            yield f"event: prepare\ndata: {json.dumps({'interaction_id': interaction_id, 'status': 'PREPARING', 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
            await asyncio.sleep(0.01)

            # Check client disconnect before preparing context
            if await request.is_disconnected():
                _mark_interrupted("Client disconnected before context preparation")
                return

            # 2. Prepare Context
            if existing_record is not None:
                stream_session_id = existing_record.session_id
                agent_id = existing_record.agent_id
            else:
                stream_session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:8]}"
                agent_id = payload.agent_id or identity.identity_id
            agent_input = AgentInteractionInput(
                interaction_id=interaction_id,
                project_id=project_id,
                session_id=stream_session_id,
                agent_id=agent_id,
                content=payload.input_text,
                idempotency_key=payload.idempotency_key,
                token_budget=payload.max_context_tokens,
                query=payload.search_query,
                metadata=payload.metadata or {},
            )

            prep_result = container.brain_loop.prepare_context(agent_input)

            # 3. Emit context_delivered event
            tokens_val = prep_result.context_package.tokens_used if prep_result.context_package else 0
            prompt_md = (
                (prep_result.context_package.rendered_markdown or prep_result.context_package.rendered_text)
                if prep_result.context_package
                else ""
            )
            context_data = {
                "interaction_id": interaction_id,
                "tokens_used": tokens_val,
                "memories_count": len(prep_result.context_package.items) if prep_result.context_package else 0,
                "prompt_context": prompt_md[:300] + "...",
            }
            yield f"event: context_delivered\ndata: {json.dumps(context_data)}\n\n"
            await asyncio.sleep(0.01)

            if await request.is_disconnected():
                _mark_interrupted("Client disconnected after context delivery")
                return

            # 4. Resolve agent and stream progressive chunks
            if "research" in agent_id.lower():
                agent = ResearchAgent(agent_id=agent_id)
            elif "review" in agent_id.lower():
                agent = ReviewAgent(agent_id=agent_id)
            else:
                agent = CodingAgent(agent_id=agent_id)

            exec_req = AgentExecutionRequest(
                project_id=project_id,
                session_id=stream_session_id,
                content=payload.input_text,
                query=payload.search_query,
                token_budget=payload.max_context_tokens,
                retrieval_method="HYBRID",
                idempotency_key=payload.idempotency_key,
                metadata=payload.metadata or {},
            )

            full_response = ""
            for idx, chunk in enumerate(agent.stream_response(prep_result, exec_req)):
                if await request.is_disconnected():
                    _mark_interrupted(f"Client disconnected during chunk streaming (chunk {idx})")
                    return

                # Heartbeat lease renewal
                if reservation is not None:
                    try:
                        reservation.heartbeat()
                    except StaleFencingTokenError as e:
                        logger.warning("Lease lost or fencing token obsolete during chunk %d: %s", idx, e)
                        _mark_interrupted(f"Lease lost: {e}")
                        safe_message = f"StaleFencingTokenError: {e}"
                        err_data = {"interaction_id": interaction_id, "error": "FENCING_ERROR", "message": safe_message}
                        yield f"event: error\ndata: {json.dumps(err_data)}\n\n"
                        return

                full_response += chunk
                chunk_data = {
                    "interaction_id": interaction_id,
                    "index": idx,
                    "chunk": chunk,
                }
                yield f"event: agent_chunk\ndata: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)

            # Check client disconnect before completion commit
            if await request.is_disconnected():
                _mark_interrupted("Client disconnected before completion commit")
                return

            # Atomically commit completion reservation with fencing token validation
            if reservation is not None:
                try:
                    reservation.commit_completion()
                except StaleFencingTokenError as e:
                    logger.warning("Fencing token check failed before completion: %s", e)
                    _mark_interrupted(f"Fencing token obsolete: {e}")
                    safe_message = f"StaleFencingTokenError: {e}"
                    err_data = {"interaction_id": interaction_id, "error": "FENCING_ERROR", "message": safe_message}
                    yield f"event: error\ndata: {json.dumps(err_data)}\n\n"
                    return

            # 5. Complete interaction and extract memories
            resp_input = AgentResponseInput(
                interaction_id=interaction_id,
                project_id=project_id,
                session_id=stream_session_id,
                agent_id=agent_id,
                content=full_response,
            )
            completion = container.brain_loop.complete_interaction(resp_input)

            # Persist extracted memories to container memory store if configured
            if hasattr(container, "memory_store") and container.memory_store is not None:
                for m in completion.extracted_memories:
                    try:
                        container.memory_store.store(m)
                    except Exception as e:
                        logger.warning("Failed to store extracted memory '%s' in memory_store: %s", m.memory_id, e)

            comp_data = {
                "interaction_id": interaction_id,
                "status": "COMPLETED",
                "memories_extracted": len(completion.extracted_memories),
                "graph_nodes_created": completion.graph_nodes_integrated,
                "vectors_indexed": completion.vectors_indexed,
            }
            yield f"event: completion\ndata: {json.dumps(comp_data)}\n\n"

            # 6. Stream termination
            yield f"event: done\ndata: {json.dumps({'interaction_id': interaction_id, 'status': 'FINISHED'})}\n\n"

        except Exception as e:
            logger.error("SSE streaming failure on '%s': %s", interaction_id, e)
            safe_message = f"{type(e).__name__}: {e}"
            err_data = {"interaction_id": interaction_id, "error": "STREAM_ERROR", "message": safe_message}
            yield f"event: error\ndata: {json.dumps(err_data)}\n\n"

        finally:
            if reservation is not None:
                reservation.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
