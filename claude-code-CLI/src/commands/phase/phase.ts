import type { LocalCommandCall } from '../../types/command.js'
import { updateExecutionState, type ExecutionPhase } from '../../orchestration/executionState.js'

const VALID_PHASES: ExecutionPhase[] = [
  'idle',
  'backend_planning',
  'backend_execution',
  'backend_validation',
  'frontend_planning',
  'frontend_execution',
  'frontend_validation',
  'integration_check'
]

export const call: LocalCommandCall = async (args, context) => {
  const targetPhase = args.trim() as ExecutionPhase
  
  if (!targetPhase) {
    return { type: 'text', value: `Usage: /phase <phase>\nValid phases: ${VALID_PHASES.join(', ')}` }
  }
  
  if (!VALID_PHASES.includes(targetPhase)) {
    return { type: 'text', value: `Invalid phase. Valid phases: ${VALID_PHASES.join(', ')}` }
  }
  
  const state = await updateExecutionState({ implementation_phase: targetPhase })
  if (!state) {
    return { type: 'text', value: 'Failed to update phase.' }
  }
  
  return { type: 'text', value: `Orchestration phase updated to: ${targetPhase}` }
}
