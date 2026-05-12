/**
 * Execution State API client for Claude Code CLI.
 * Connects to the local proxy server to query and manage persistent orchestration state.
 */

import { getParentSessionId, getSessionId } from '../bootstrap/state.js'
import { getAuthHeaders } from '../utils/http.js'

export type ExecutionPhase =
	| 'idle'
	| 'backend_planning'
	| 'backend_execution'
	| 'backend_validation'
	| 'frontend_planning'
	| 'frontend_execution'
	| 'frontend_validation'
	| 'integration_check'

export interface PlanStep {
	step_id: string
	description: string
	status: 'pending' | 'in_progress' | 'completed' | 'skipped'
	completed_at: string | null
}

export interface CheckpointState {
	name: string
	description: string
	phase: ExecutionPhase
	created_at: string
}

export interface ExecutionState {
	session_id: string
	parent_session_id?: string | null
	active_model: string
	current_checkpoint: CheckpointState | null
	implementation_phase: ExecutionPhase
	approved_plan: string | null
	completed_steps: PlanStep[]
	remaining_steps: PlanStep[]
	locked_rules: string[]
	active_files: string[]
	validation_findings: string[]
	last_updated: string
	version: number
}

export interface ExecutionStateResponse extends Partial<ExecutionState> {
	exists: boolean
}

function getExecutionStateBaseUrl(): string | null {
	if (!isExecutionStateOrchestrationEnabled()) return null
	const baseUrl = process.env.ANTHROPIC_BASE_URL
	if (!baseUrl) return null
	return baseUrl.replace(/\/+$/, '').replace(/\/v1$/, '')
}

export function isExecutionStateOrchestrationEnabled(): boolean {
	const value = process.env.CLAUDE_CODE_ENABLE_EXECUTION_STATE_ORCHESTRATION
	if (!value) return false
	return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase())
}

function getExecutionStateHeaders(): Record<string, string> | null {
	const auth = getAuthHeaders()
	if (auth.error) return null

	const sessionId = getSessionId()
	const parentSessionId = getParentSessionId()

	return {
		...auth.headers,
		'x-session-id': sessionId,
		...(parentSessionId ? { 'x-parent-session-id': parentSessionId } : {}),
	}
}

/**
 * Get the full execution state for the current session.
 */
export async function fetchExecutionState(): Promise<ExecutionStateResponse | null> {
	try {
		const proxyUrl = getExecutionStateBaseUrl()
		if (!proxyUrl) return null
		const sessionId = getSessionId()
		const headers = getExecutionStateHeaders()
		if (!headers) return null

		const response = await fetch(`${proxyUrl}/v1/execution_state/${sessionId}`, {
			method: 'GET',
			headers: {
				...headers,
				Accept: 'application/json',
			},
		})

		if (!response.ok) {
			if (response.status === 404) return null
			throw new Error(`Failed to fetch execution state: ${response.statusText}`)
		}

		return (await response.json()) as ExecutionStateResponse
	} catch {
		// Graceful fallback if proxy is unavailable or endpoint missing
		return null
	}
}

/**
 * Partially update the execution state.
 */
export async function updateExecutionState(
	update: Partial<ExecutionState>,
): Promise<ExecutionState | null> {
	try {
		const proxyUrl = getExecutionStateBaseUrl()
		if (!proxyUrl) return null
		const sessionId = getSessionId()
		const headers = getExecutionStateHeaders()
		if (!headers) return null

		const response = await fetch(`${proxyUrl}/v1/execution_state/${sessionId}`, {
			method: 'PUT',
			headers: {
				...headers,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify(update),
		})

		if (!response.ok) {
			throw new Error(`Failed to update execution state: ${response.statusText}`)
		}

		return (await response.json()) as ExecutionState
	} catch {
		return null
	}
}

/**
 * Parse and persist an approved execution plan.
 */
export async function applyPlan(planText: string): Promise<ExecutionState | null> {
	try {
		const proxyUrl = getExecutionStateBaseUrl()
		if (!proxyUrl) return null
		const sessionId = getSessionId()
		const headers = getExecutionStateHeaders()
		if (!headers) return null
		const parentSessionId = getParentSessionId()

		const response = await fetch(`${proxyUrl}/v1/execution_state/${sessionId}/apply_plan`, {
			method: 'POST',
			headers: {
				...headers,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify({
				plan_text: planText,
				...(parentSessionId ? { parent_session_id: parentSessionId } : {}),
			}),
		})

		if (!response.ok) {
			throw new Error(`Failed to apply plan: ${response.statusText}`)
		}

		return (await response.json()) as ExecutionState
	} catch {
		return null
	}
}

/**
 * Mark a specific step as completed in the plan.
 */
export async function completePlanStep(stepId: string): Promise<ExecutionState | null> {
	try {
		const proxyUrl = getExecutionStateBaseUrl()
		if (!proxyUrl) return null
		const sessionId = getSessionId()
		const headers = getExecutionStateHeaders()
		if (!headers) return null

		const response = await fetch(`${proxyUrl}/v1/execution_state/${sessionId}/complete_step`, {
			method: 'POST',
			headers: {
				...headers,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify({ step_id: stepId }),
		})

		if (!response.ok) {
			throw new Error(`Failed to complete step: ${response.statusText}`)
		}

		return (await response.json()) as ExecutionState
	} catch {
		return null
	}
}

/**
 * Set a checkpoint in the current session.
 */
export async function setCheckpoint(
	name: string,
	description = '',
	phase?: ExecutionPhase,
): Promise<ExecutionState | null> {
	try {
		const proxyUrl = getExecutionStateBaseUrl()
		if (!proxyUrl) return null
		const sessionId = getSessionId()
		const headers = getExecutionStateHeaders()
		if (!headers) return null

		const response = await fetch(`${proxyUrl}/v1/execution_state/${sessionId}/set_checkpoint`, {
			method: 'POST',
			headers: {
				...headers,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify({ name, description, phase }),
		})

		if (!response.ok) {
			throw new Error(`Failed to set checkpoint: ${response.statusText}`)
		}

		return (await response.json()) as ExecutionState
	} catch {
		return null
	}
}

/**
 * Restore a checkpoint in the current session.
 */
export async function restoreCheckpoint(): Promise<ExecutionState | null> {
	try {
		const proxyUrl = getExecutionStateBaseUrl()
		if (!proxyUrl) return null
		const sessionId = getSessionId()
		const headers = getExecutionStateHeaders()
		if (!headers) return null

		const response = await fetch(`${proxyUrl}/v1/execution_state/${sessionId}/restore_checkpoint`, {
			method: 'POST',
			headers: {
				...headers,
				Accept: 'application/json',
			},
		})

		if (!response.ok) {
			throw new Error(`Failed to restore checkpoint: ${response.statusText}`)
		}

		return (await response.json()) as ExecutionState
	} catch {
		return null
	}
}
