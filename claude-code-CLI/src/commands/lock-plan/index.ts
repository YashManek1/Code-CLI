import type { Command } from '../../commands.js'

const lockPlan = {
  type: 'local',
  name: 'lock-plan',
  description: 'Lock the current plan into persistent orchestration state',
  load: () => import('./lock-plan.js'),
} satisfies Command

export default lockPlan
