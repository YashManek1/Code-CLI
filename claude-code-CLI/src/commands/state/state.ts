import {
	applyPlan,
	isExecutionStateOrchestrationEnabled,
} from '../../orchestration/executionState.js'
import type { LocalCommandCall } from '../../types/command.js'
import { getPlan } from '../../utils/plans.js'

export const call: LocalCommandCall = async (_, _context) => {
	const planContent = getPlan()

	if (!planContent || !planContent.trim()) {
		return {
			type: 'text',
			value: 'No plan currently exists to lock. Create one in plan mode first.',
		}
	}

	if (!isExecutionStateOrchestrationEnabled()) {
		return {
			type: 'text',
			value:
				'Execution-state orchestration is disabled. Set CLAUDE_CODE_ENABLE_EXECUTION_STATE_ORCHESTRATION=1 to enable plan locking.',
		}
	}

	const state = await applyPlan(planContent)

	if (!state) {
		return {
			type: 'text',
			value: 'Failed to lock plan into orchestration state.',
		}
	}

	return {
		type: 'text',
		value: 'Plan successfully locked into persistent orchestration state.',
	}
}
