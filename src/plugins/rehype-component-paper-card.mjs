/// <reference types="mdast" />
import { h } from "hastscript";

/**
 * Rounded “paper note” card wrapper. Use in markdown as (four colons so inner `:::note` … `:::` does not close this block):
 * ::::paper{tone="act"}
 * ## ShortName
 * :::note[…]
 * :::
 * ::::
 *
 * Optional `tone` adds a left accent (act | dp | dp3 | default).
 *
 * @param {Record<string, unknown>} properties
 * @param {import("hast").ElementContent[]} children
 * @returns {import("hast").Element}
 */
export function PaperCardComponent(properties, children) {
	const kids = Array.isArray(children) ? children : [];
	const rawTone = properties?.tone ?? properties?.variant;
	const tone =
		typeof rawTone === "string" && rawTone.trim() !== ""
			? rawTone.trim().toLowerCase()
			: undefined;

	const attrs = { class: "paper-note-card" };
	if (tone) attrs["data-paper-tone"] = tone;

	return h("div", attrs, kids);
}
