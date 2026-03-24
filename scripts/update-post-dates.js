/**
 * Batch update `published` date for all posts.
 * Usage: node scripts/update-post-dates.js [YYYY-MM-DD]
 *   - No arg: set all to today
 *   - With date: set all to given date (e.g. 2026-03-24)
 *
 * Both article pages and Archive read from the same `published` field—they stay in sync.
 */

import fs from "fs";
import path from "path";

function getTodayDate() {
	const d = new Date();
	const year = d.getFullYear();
	const month = String(d.getMonth() + 1).padStart(2, "0");
	const day = String(d.getDate()).padStart(2, "0");
	return `${year}-${month}-${day}`;
}

function collectMdFiles(dir, list = []) {
	const entries = fs.readdirSync(dir, { withFileTypes: true });
	for (const e of entries) {
		const full = path.join(dir, e.name);
		if (e.isDirectory()) {
			collectMdFiles(full, list);
		} else if (e.name.endsWith(".md")) {
			list.push(full);
		}
	}
	return list;
}

function updatePublishedInFile(filePath, newDate) {
	let content = fs.readFileSync(filePath, "utf-8");
	// Match published: YYYY-MM-DD or published: "YYYY-MM-DD"
	const publishedRe = /^(published:\s*)(["']?)(\d{4}-\d{2}-\d{2})(["']?)\s*$/m;
	if (!publishedRe.test(content)) {
		console.warn(`  Skip (no published field): ${filePath}`);
		return false;
	}
	content = content.replace(publishedRe, `$1$2${newDate}$4`);
	fs.writeFileSync(filePath, content);
	return true;
}

const args = process.argv.slice(2);
const targetDate = args[0] && /^\d{4}-\d{2}-\d{2}$/.test(args[0]) ? args[0] : getTodayDate();

const postsDir = "./src/content/posts";
const files = collectMdFiles(postsDir);

console.log(`Updating published to ${targetDate} for ${files.length} file(s)...`);
let updated = 0;
for (const f of files) {
	if (updatePublishedInFile(f, targetDate)) {
		console.log(`  OK: ${f}`);
		updated++;
	}
}
console.log(`Done. Updated ${updated} file(s).`);
