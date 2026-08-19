<script lang="ts">
  /** Панель источников: выписки из документов, а не ссылки на файлы.
   *
   * Появляется до первого символа ответа - человек видит, на чём ответ
   * основан, ещё до того как ответ написан. Пустой список это сигнал
   * «убрать прошлые выдержки», а не отсутствие данных.
   */
  import { api, type Source } from "./api";

  let { sources, searching, talkMode }: {
    sources: Source[];
    searching: boolean;
    talkMode: boolean;
  } = $props();

  // На узком экране панель не помещается третьей колонкой. Прятать её нельзя:
  // выписки - главное, ради чего интерфейс существует. Поэтому внизу остаётся
  // полоса со счётчиком, а по нажатию поднимается шторка.
  let sheetOpen = $state(false);

  let openCode = $state("");
  let reading = $state<{ title: string; chunks: { clause: string | null; breadcrumb: string; text: string }[] } | null>(null);
  let readingError = $state("");

  async function openDocument(code: string) {
    openCode = code;
    reading = null;
    readingError = "";
    try {
      const document = await api.document(code);
      reading = { title: document.doc_title, chunks: document.chunks };
    } catch (error) {
      readingError = error instanceof Error ? error.message : "Документ не открылся";
    }
  }

  function close() {
    openCode = "";
    reading = null;
    readingError = "";
  }
</script>

<button
  class="bar"
  onclick={() => (sheetOpen = true)}
  aria-expanded={sheetOpen}
  aria-label="Показать источники"
>
  <span>Источники</span>
  {#if sources.length}<span class="bar-badge">{sources.length}</span>{/if}
  <span class="bar-chevron" aria-hidden="true">^</span>
</button>

{#if sheetOpen}
  <div class="sheet-scrim" role="presentation" onclick={() => (sheetOpen = false)}></div>
{/if}

<aside class="panel" class:sheet={sheetOpen} aria-label="Источники">
  {#if sheetOpen}
    <span class="grabber" aria-hidden="true"></span>
  {/if}
  <header>
    <h2>Источники</h2>
    {#if sources.length}
      <span class="badge">{sources.length}</span>
      <span class="note mono">отобрано до ответа</span>
    {/if}
  </header>

  <div class="body">
    {#if sources.length}
      {#each sources as source (source.n)}
        <article class="card">
          <div class="top">
            <span class="num">{source.n}</span>
            <span class="code mono">{source.doc_code}</span>
            {#if typeof source.score === "number"}
              <span class="score mono">{source.score.toFixed(2)}</span>
            {/if}
          </div>
          <h3>{source.doc_title}</h3>
          <p class="crumb mono">
            {source.breadcrumb || [source.section_no, source.section_title].filter(Boolean).join(" › ")}
          </p>
          {#if !source.clause}
            <span class="chip">без пункта</span>
          {/if}
          {#if source.snippet || source.text}
            <blockquote>«{source.snippet || source.text}»</blockquote>
          {/if}
          <button class="open" onclick={() => openDocument(source.doc_code)}>
            Открыть документ
          </button>
        </article>
      {/each}
    {:else if searching}
      <p class="empty-title">Отбираем фрагменты…</p>
      <div class="skeleton"></div>
      <div class="skeleton"></div>
    {:else if talkMode}
      <p class="empty-title">Обычная реплика</p>
      <p class="empty">
        В базу не ходили — источников нет. Так и должно быть: цитировать
        регламент незачем.
      </p>
    {:else}
      <p class="empty">
        Здесь появятся выписки из документов — до того, как ответ будет написан.
      </p>
      <p class="empty dim">
        Код документа, путь до пункта и цитата. Ссылок на файлы нет: документы
        живут внутри системы.
      </p>
    {/if}
  </div>
</aside>

{#if openCode}
  <div
    class="overlay"
    role="dialog"
    aria-modal="true"
    aria-label="Документ {openCode}"
  >
    <div class="reader">
      <header class="reader-head">
        <div>
          <span class="code mono">{openCode}</span>
          <h3>{reading?.title ?? "Загружаем…"}</h3>
        </div>
        <button class="close" onclick={close} aria-label="Закрыть документ">×</button>
      </header>
      <div class="reader-body">
        {#if readingError}
          <p class="empty">{readingError}</p>
        {:else if reading}
          {#each reading.chunks as chunk (chunk.text.slice(0, 40) + chunk.breadcrumb)}
            <section>
              <p class="crumb mono">{chunk.clause ?? chunk.breadcrumb}</p>
              <p>{chunk.text}</p>
            </section>
          {/each}
        {:else}
          <div class="skeleton"></div>
          <div class="skeleton"></div>
          <div class="skeleton"></div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .panel {
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: var(--panel);
    border-left: 1px solid var(--line);
  }

  header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--line);
  }

  h2 {
    margin: 0;
    font-size: 14px;
    font-weight: 640;
    letter-spacing: -0.015em;
  }

  .badge {
    min-width: 20px;
    padding: 1px 6px;
    text-align: center;
    font-size: 11px;
    background: var(--pri-soft);
    color: var(--pri);
    border-radius: 6px;
  }

  .note {
    margin-left: auto;
    font-size: 10.5px;
    color: var(--text3);
  }

  .body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: grid;
    gap: 10px;
    align-content: start;
  }

  .card {
    padding: 12px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 11px;
    box-shadow: var(--sh1);
  }

  .top {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .num {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    font-size: 11px;
    background: var(--pri-soft);
    color: var(--pri);
    border-radius: 6px;
  }

  .code {
    font-size: 11.5px;
    color: var(--text2);
  }

  .score {
    margin-left: auto;
    font-size: 10.5px;
    color: var(--text3);
  }

  h3 {
    margin: 8px 0 4px;
    font-size: 13px;
    font-weight: 620;
    line-height: 1.35;
  }

  .crumb {
    margin: 0;
    font-size: 10.5px;
    color: var(--text3);
  }

  .chip {
    display: inline-block;
    margin-top: 6px;
    padding: 1px 7px;
    font-size: 10.5px;
    color: var(--warn);
    background: var(--warn-soft);
    border-radius: 5px;
  }

  blockquote {
    margin: 8px 0 0;
    font-size: 12px;
    line-height: 1.5;
    color: var(--text2);
  }

  .open {
    margin-top: 10px;
    padding: 5px 9px;
    font-size: 11.5px;
    background: transparent;
    color: var(--pri);
    border: 1px solid var(--pri-line);
    border-radius: 7px;
  }

  .open:hover {
    background: var(--pri-soft);
  }

  .empty,
  .empty-title {
    margin: 0;
    font-size: 12px;
    color: var(--text2);
    line-height: 1.55;
  }

  .empty-title {
    font-weight: 620;
    color: var(--text);
  }

  .dim {
    color: var(--text3);
  }

  .skeleton {
    height: 54px;
    border-radius: 10px;
    background: linear-gradient(90deg, var(--surface2), var(--hover), var(--surface2));
    background-size: 200% 100%;
    animation: shimmer 1.4s linear infinite;
  }

  @keyframes shimmer {
    to {
      background-position: -200% 0;
    }
  }

  .overlay {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 24px;
    background: rgba(15, 18, 22, 0.38);
    z-index: 20;
  }

  .reader {
    width: min(760px, 100%);
    max-height: 84vh;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: var(--sh2);
  }

  .reader-head {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px 18px;
    border-bottom: 1px solid var(--line);
  }

  .reader-head h3 {
    margin: 4px 0 0;
    font-size: 16.5px;
  }

  .close {
    margin-left: auto;
    width: 30px;
    height: 30px;
    font-size: 18px;
    line-height: 1;
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 8px;
    color: var(--text2);
  }

  .reader-body {
    overflow-y: auto;
    padding: 18px;
    display: grid;
    gap: 16px;
  }

  .reader-body p {
    margin: 4px 0 0;
    font-size: 13.5px;
    line-height: 1.6;
  }

  .bar,
  .sheet-scrim,
  .grabber {
    display: none;
  }

  @media (max-width: 1100px) {
    .panel {
      display: none;
    }

    /* Кнопки и полосы не меньше 44 px: интерфейс открывают в цеху,
       нередко в перчатках. */
    .bar {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 15;
      display: flex;
      align-items: center;
      gap: 8px;
      height: 52px;
      padding: 0 16px;
      background: var(--panel);
      border: none;
      border-top: 1px solid var(--line);
      color: var(--text2);
      font-weight: 580;
    }

    .bar-badge {
      min-width: 20px;
      padding: 1px 6px;
      font-size: 11px;
      background: var(--pri-soft);
      color: var(--pri);
      border-radius: 6px;
    }

    .bar-chevron {
      margin-left: auto;
      color: var(--text3);
    }

    .sheet-scrim {
      position: fixed;
      inset: 0;
      display: block;
      background: rgba(15, 18, 22, 0.38);
      z-index: 16;
    }

    .panel.sheet {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 17;
      display: flex;
      height: 74%;
      border-left: none;
      border-radius: 20px 20px 0 0;
      box-shadow: 0 -18px 40px -20px rgba(0, 0, 0, 0.4);
    }

    .grabber {
      position: absolute;
      top: 7px;
      left: 50%;
      transform: translateX(-50%);
      display: block;
      width: 38px;
      height: 4px;
      background: var(--line);
      border-radius: 2px;
    }

    .panel.sheet header {
      padding-top: 20px;
    }
  }
</style>
