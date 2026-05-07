const fs = require('fs');
const path = require('path');
const os = require('os');

const configFile = path.join(os.homedir(), '.claude.json');
const projectPath = 'c:/Claude Code/claude-code-CLI';

let config = {};
if (fs.existsSync(configFile)) {
    config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
}

if (!config.projects) {
    config.projects = {};
}

config.projects[projectPath] = {
    hasTrustDialogAccepted: true
};

fs.writeFileSync(configFile, JSON.stringify(config, null, 2));
console.log('Updated config with trust for:', projectPath);
