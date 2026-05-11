import type { Command } from '../../commands.js'

const checkpoint = {
  type: 'local',
  name: 'checkpoint',
  description: 'Create a new orchestration checkpoint',
  argumentHint: '<name> [description]',
  load: () => import('./checkpoint.js'),
} satisfies Command

export default checkpoint
