import type { Command } from '../../commands.js'

const resumeCheckpoint = {
  type: 'local',
  name: 'resume-checkpoint',
  description: 'Revert to a previously saved orchestration checkpoint',
  argumentHint: '<name>',
  load: () => import('./resume-checkpoint.js'),
} satisfies Command

export default resumeCheckpoint
