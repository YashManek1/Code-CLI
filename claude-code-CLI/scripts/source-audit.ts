import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const ROOT = process.cwd()
const SRC_DIR = join(ROOT, 'src')
const STRICT = process.argv.includes('--strict')
const EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx']
const ALLOWED_SHORT_NAMES = new Set([
	'e',
	'i',
	'j',
	'k',
	'm',
	'n',
	'x',
	'y',
	'z',
	'id',
	'db',
	'fs',
	'io',
	'os',
	'ui',
])

type Finding = {
	file: string
	line: number
	rule: string
	text: string
}

function walk(dir: string, files: string[] = []): string[] {
	for (const entry of readdirSync(dir)) {
		if (entry === 'node_modules' || entry === 'dist') continue
		const path = join(dir, entry)
		const stat = statSync(path)
		if (stat.isDirectory()) {
			walk(path, files)
		} else if (EXTENSIONS.some((ext) => path.endsWith(ext))) {
			files.push(path)
		}
	}
	return files
}

function lineFindings(file: string, source: string): Finding[] {
	const findings: Finding[] = []
	const relativeFile = relative(ROOT, file).replaceAll('\\', '/')
	const lines = source.split(/\r?\n/)
	for (let index = 0; index < lines.length; index++) {
		const line = lines[index]!
		const lineNumber = index + 1
		if (/\bTODO\b/.test(line) && !/TODO\([^)]+\)|TODO:[^\n]+#\d+/.test(line)) {
			findings.push({
				file: relativeFile,
				line: lineNumber,
				rule: 'stale-todo',
				text: line.trim(),
			})
		}
		if (/console\.(log|debug)\(/.test(line) && !relativeFile.includes('/shims/')) {
			findings.push({
				file: relativeFile,
				line: lineNumber,
				rule: 'console-leftover',
				text: line.trim(),
			})
		}
		for (const match of line.matchAll(/\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\b/g)) {
			const name = match[1]!
			if (name.length < 3 && !ALLOWED_SHORT_NAMES.has(name)) {
				findings.push({
					file: relativeFile,
					line: lineNumber,
					rule: 'short-variable',
					text: line.trim(),
				})
			}
		}
	}
	return findings
}

const findings = walk(SRC_DIR).flatMap((file) =>
	lineFindings(file, readFileSync(file, 'utf8')),
)

const grouped = new Map<string, Finding[]>()
for (const finding of findings) {
	const group = grouped.get(finding.rule) ?? []
	group.push(finding)
	grouped.set(finding.rule, group)
}

for (const [rule, ruleFindings] of grouped) {
	console.log(`${rule}: ${ruleFindings.length}`)
	for (const finding of ruleFindings.slice(0, 20)) {
		console.log(`  ${finding.file}:${finding.line} ${finding.text}`)
	}
	if (ruleFindings.length > 20) {
		console.log(`  ... ${ruleFindings.length - 20} more`)
	}
}

if (STRICT && findings.length > 0) {
	process.exitCode = 1
}
