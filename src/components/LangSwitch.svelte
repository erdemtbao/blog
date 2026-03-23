<script lang="ts">
import Icon from "@iconify/svelte";
import {
	applyDomI18n,
	FUWARI_LANG_CHANGE,
	getStoredDisplayLang,
	setStoredDisplayLang,
	type DisplayLang,
} from "@i18n/client-i18n";
import { onMount } from "svelte";

let current: DisplayLang = $state("en");

onMount(() => {
	current = getStoredDisplayLang();
});

function toggleLang(): void {
	const next: DisplayLang = current === "en" ? "zh_CN" : "en";
	setStoredDisplayLang(next);
	current = next;
	applyDomI18n();
	window.dispatchEvent(new CustomEvent(FUWARI_LANG_CHANGE));
}
</script>

<button
	type="button"
	class="btn-plain scale-animation rounded-lg h-11 px-2.5 active:scale-90 font-bold text-sm flex items-center gap-1 text-black/70 dark:text-white/70"
	aria-label={current === "en" ? "Switch to Chinese" : "Switch to English"}
	onclick={toggleLang}
>
	<Icon icon="material-symbols:translate-rounded" class="text-[1.2rem] opacity-80"></Icon>
	<span class:opacity-100={current === "en"} class:opacity-40={current !== "en"}>EN</span>
	<span class="opacity-30 text-xs">|</span>
	<span class:opacity-100={current === "zh_CN"} class:opacity-40={current !== "zh_CN"}>中文</span>
</button>
