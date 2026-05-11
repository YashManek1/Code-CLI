import type { LocalCommandCall } from '../../types/command.js'
import { setCheckpoint } from '../../orchestration/executionState.js'

export const call: LocalCommandCall = async (args, context) => {
  const parts = args.trim().split(' ')
  const name = parts[0]
  if (!name) {
    return { type: 'text', value: 'Usage: /checkpoint <name> [description]' }
  }
  
  const description = parts.slice(1).join(' ')
  
  const state = await setCheckpoint(name, description)
  if (!state) {
    return { type: 'text', value: 'Failed to set checkpoint.' }
  }
  
  return { 
    type: 'text', 
    value: `Checkpoint '${name}' created successfully. Current Phase: ${state.implementation_phase}` 
  }
}
