import type { LocalCommandCall } from '../../types/command.js'
import { restoreCheckpoint } from '../../orchestration/executionState.js'

export const call: LocalCommandCall = async (_, context) => {
  const state = await restoreCheckpoint()
  if (!state) {
    return { type: 'text', value: 'Failed to restore checkpoint. Ensure a checkpoint was created.' }
  }
  
  return { type: 'text', value: `Checkpoint restored. Current Phase: ${state.implementation_phase}` }
}
