export const profileCheckpoint = (name: string) => {};
export const enableConfigs = () => {};
export const getMainLoopModel = () => 'claude-3-5-sonnet-20241022';
export const getSystemPrompt = async () => [];
export const runClaudeInChromeMcpServer = async () => {};
export const runChromeNativeHost = async () => {};
export const runComputerUseMcpServer = async () => {};
export const runDaemonWorker = async () => {};
export const getBridgeDisabledReason = async () => null;
export const checkBridgeMinVersion = () => null;
export const bridgeMain = async () => {};
export const exitWithError = (msg: string) => { console.error(msg); process.exit(1); };
export const getClaudeAIOAuthTokens = () => ({ accessToken: 'dummy' });
export const waitForPolicyLimitsToLoad = async () => {};
export const isPolicyAllowed = () => true;
export const initSinks = () => {};
export const daemonMain = async () => {};
export const psHandler = async () => {};
export const logsHandler = async () => {};
export const attachHandler = async () => {};
export const killHandler = async () => {};
export const handleBgFlag = async () => {};
export const templatesMain = async () => {};
export const environmentRunnerMain = async () => {};
export const selfHostedRunnerMain = async () => {};
export const isWorktreeModeEnabled = () => false;
export const execIntoTmuxWorktree = async () => ({ handled: false });
export const startCapturingEarlyInput = () => {};
export const stopUltraplan = () => {};
export const main = async () => { console.log("Dummy main loop started"); };

export default { 
  name: 'dummy', 
  description: 'dummy', 
  type: 'local', 
  run: () => {},
  psHandler,
  logsHandler,
  attachHandler,
  killHandler,
  handleBgFlag,
  main
};

// Common proactive/contextCollapse exports
export const isProactiveActive = () => false;
export const subscribeToProactiveChanges = () => () => {};
export const getNextTickAt = () => null;
export const isContextCollapseEnabled = () => false;
export const getStats = () => ({ health: { emptySpawnWarningEmitted: false, totalErrors: 0, totalEmptySpawns: 0 }, collapsedSpans: 0, stagedSpans: 0 });
export const subscribe = () => () => {};
export const getWorkflowCommands = async () => [];
export const clearSkillIndexCache = () => {};
export const DEFAULT_UPLOAD_CONCURRENCY = 5;
export const FILE_COUNT_LIMIT = 1000;
export const OUTPUTS_SUBDIR = 'outputs';
