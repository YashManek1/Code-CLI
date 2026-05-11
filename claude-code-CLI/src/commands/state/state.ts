import type { LocalCommandCall } from '../../types/command.js'
import { getPlan } from '../../utils/plans.js'
import { applyPlan } from '../../orchestration/executionState.js'

// Adjust this import if your active plan state lives elsewhere
import { getCurrentPlanState } from '../../planMode/state.js'

export const call: LocalCommandCall = async (_, context) => {

  const activePlanState = getCurrentPlanState()

  const planContent =
    activePlanState?.content ??
    getPlan()

  if (!planContent || !planContent.trim()) {
    return {
      type: 'text',
      value:
        'No plan currently exists to lock. Create one in plan mode first.',
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
    value:
      'Plan successfully locked into persistent orchestration state.',
  }
}