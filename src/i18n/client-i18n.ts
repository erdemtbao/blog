/**
 * Client-side i18n utilities for runtime language switching.
 * Used by LangSwitch.svelte.
 */

import { getTranslation } from "./translation";

export type DisplayLang = "en" | "zh_CN";

const STORAGE_KEY = "fuwari-display-lang";

export const FUWARI_LANG_CHANGE = "fuwari-lang-change";

export function getStoredDisplayLang(): DisplayLang {
	if (typeof window === "undefined") return "en";
	const stored = localStorage.getItem(STORAGE_KEY);
	if (stored === "en" || stored === "zh_CN") return stored;
	return "en";
}

export function setStoredDisplayLang(lang: DisplayLang): void {
	if (typeof window === "undefined") return;
	localStorage.setItem(STORAGE_KEY, lang);
}

export function applyDomI18n(): void {
	if (typeof window === "undefined") return;
	const lang = getStoredDisplayLang();
	const langKey = lang === "zh_CN" ? "zh_cn" : "en";
	const translation = getTranslation(langKey);

	document.querySelectorAll<HTMLElement>("[data-i18n]").forEach((el) => {
		const key = el.getAttribute("data-i18n");
		if (key && Object.hasOwn(translation, key)) {
			el.textContent = translation[key as keyof typeof translation];
		}
	});
}
