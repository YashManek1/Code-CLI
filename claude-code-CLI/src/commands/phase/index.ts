import type { Command } from '../../commands.js'

const phase = {
  type: 'local',
  name: 'phase',
  description: 'Update the orchestration phase',
  argumentHint: '<phase_name>',
  load: () => import('./phase.js'),
} satisfies Command

export default phase
