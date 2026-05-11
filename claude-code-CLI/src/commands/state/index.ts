import type { Command } from '../../commands.js'

const state = {
  type: 'local',
  name: 'state',
  description: 'View current orchestration state',
  load: () => import('./state.js'),
} satisfies Command

export default state
