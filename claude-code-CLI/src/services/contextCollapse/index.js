export function isContextCollapseEnabled() { return false; }
export function getStats() { return { health: { emptySpawnWarningEmitted: false, totalErrors: 0, totalEmptySpawns: 0 }, collapsedSpans: 0, stagedSpans: 0 }; }
export function subscribe() { return () => {}; }
