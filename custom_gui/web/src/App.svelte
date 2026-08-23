<script lang="ts">
  /** Оболочка приложения: три колонки, лента этапов и поле ввода. */
  import Login from "./lib/Login.svelte";
  import logo from "./assets/logo-bird.png";
  import Help from "./lib/Help.svelte";
  import Sources from "./lib/Sources.svelte";
  import { renderMarkdown } from "./lib/markdown";
  import {
    appearance,
    applyAnswerSize,
    applyTheme,
    catalog,
    chat,
    deleteConversation,
    loadCatalog,
    loadConversations,
    loadSession,
    logout,
    newConversation,
    openConversation,
    renameConversation,
    restoreAppearance,
    sendQuestion,
    session,
    stopStreaming,
  } from "./lib/state.svelte";

  /** Стартовые подсказки. Области взяты из реального состава корпуса. */
  const STARTERS = [
    { area: "Охрана труда", q: "Сколько дней дополнительного отпуска положено за вредность?" },
    { area: "Работы повышенной опасности", q: "Какой порядок допуска к газоопасным работам?" },
    { area: "Пожарная безопасность", q: "Как часто проверяют огнетушители в цеху?" },
    { area: "Промышленная безопасность", q: "Кто проводит техническое освидетельствование оборудования?" },
    { area: "ГО и ЧС", q: "Что делать при сигнале «Внимание всем»?" },
    { area: "Производственная безопасность", q: "Какие требования к средствам индивидуальной защиты?" },
  ];

  let draft = $state("");
  let alwaysSearch = $state(false);
  let treeOpen = $state(false);
  let treeQuery = $state("");
  let feed: HTMLDivElement | null = $state(null);
  // Следовать за ответом, только пока человек внизу. Иначе он не сможет
  // перечитать начало длинного регламента: лента дёргала бы его обратно
  // на каждом токене.
  let stickToBottom = $state(true);
  let renamingId = $state("");
  let renameDraft = $state("");
  let confirmingId = $state("");
  let helpOpen = $state(false);

  const areaChip = $derived(
    catalog.selectedAreas.length
      ? `${catalog.selectedAreas.length} из ${catalog.areas.length || "…"}`
      : "вся база",
  );

  /** Короткая сводка ленты этапов для узкого экрана.
   *
   * Пять строк на телефоне вытесняют ответ за пределы экрана, а нужны они
   * там редко: человек смотрит, что система работает, а не разбирает ход.
   */
  const stagesSummary = $derived.by(() => {
    const total = chat.stages.length;
    const branches = chat.stages.filter((s) => s.key.startsWith("search_agent:")).length;
    const parts = [`${total} ${plural(total, "этап", "этапа", "этапов")}`];
    if (branches) {
      parts.push(`${branches} ${plural(branches, "подзапрос", "подзапроса", "подзапросов")}`);
    }
    return parts.join(" · ");
  });

  const talkMode = $derived(
    !chat.streaming &&
      chat.messages.length > 0 &&
      chat.stages.every((s) => s.key !== "finalize") &&
      chat.liveSources.length === 0,
  );

  const visibleAreas = $derived(
    treeQuery.trim()
      ? catalog.areas
          .map((area) => ({
            ...area,
            documents: area.documents.filter((d) =>
              `${d.doc_code} ${d.doc_title}`
                .toLowerCase()
                .includes(treeQuery.trim().toLowerCase()),
            ),
          }))
          .filter((a) => a.documents.length || a.title.toLowerCase().includes(treeQuery.trim().toLowerCase()))
      : catalog.areas,
  );

  // Начальная загрузка ровно один раз. В $effect она перезапускала сама себя:
  // эффект пишет состояние, которое сам же и читает, и запрос «кто вошёл»
  // уходил трижды подряд. Одноразовая инициализация к реактивности отношения
  // не имеет, ей место здесь, а не в эффекте.
  let booted = false;
  if (!booted) {
    booted = true;
    restoreAppearance();
    void loadSession().then(async () => {
      if (session.username) {
        await loadConversations();
        await loadCatalog();
      }
    });
  }

  // Лента прокручивается к последнему сообщению по мере печати ответа.
  $effect(() => {
    void chat.draftAnswer;
    void chat.messages.length;
    if (feed && stickToBottom) feed.scrollTop = feed.scrollHeight;
  });

  function onFeedScroll() {
    if (!feed) return;
    // 24 px запаса: точное равенство не срабатывает из-за дробных высот.
    stickToBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 24;
  }

  // Esc закрывает панель областей откуда угодно, а не только из поля ввода.
  function onWindowKey(event: KeyboardEvent) {
    if (event.key === "Escape" && treeOpen) {
      treeOpen = false;
      event.stopPropagation();
    }
  }

  function send() {
    const text = draft;
    draft = "";
    void sendQuestion(text, alwaysSearch);
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  function toggleArea(area: string) {
    catalog.selectedAreas = catalog.selectedAreas.includes(area)
      ? catalog.selectedAreas.filter((a) => a !== area)
      : [...catalog.selectedAreas, area];
  }

  function toggleDoc(code: string) {
    catalog.selectedDocs = catalog.selectedDocs.includes(code)
      ? catalog.selectedDocs.filter((c) => c !== code)
      : [...catalog.selectedDocs, code];
  }

  function resetTree() {
    catalog.selectedAreas = [];
    catalog.selectedDocs = [];
  }

  /** Ставит курсор в поле сразу после появления.
   *
   * Действием, а не атрибутом ``autofocus``: атрибут срабатывает и при
   * загрузке страницы, уводя фокус у людей с экранным диктором. Здесь поле
   * появляется по осознанному действию человека, и фокус в нём уместен.
   */
  function focusOnMount(node: HTMLInputElement) {
    node.focus();
    node.select();
  }

  function startRename(id: string, title: string) {
    confirmingId = "";
    renamingId = id;
    renameDraft = title;
  }

  function commitRename() {
    const value = renameDraft.trim();
    if (renamingId && value) void renameConversation(renamingId, value);
    renamingId = "";
  }

  function onRenameKey(event: KeyboardEvent) {
    if (event.key === "Enter") {
      event.preventDefault();
      commitRename();
    } else if (event.key === "Escape") {
      event.stopPropagation();
      renamingId = "";
    }
  }

  /** Отмечен ли документ области хотя бы частично. */
  function areaMark(area: (typeof catalog.areas)[number]): "none" | "some" | "all" {
    const codes = area.documents.map((d) => d.doc_code);
    const picked = codes.filter((c) => catalog.selectedDocs.includes(c)).length;
    if (catalog.selectedAreas.includes(area.area) && !picked) return "all";
    if (!picked) return "none";
    return picked === codes.length ? "all" : "some";
  }

  /** Склоняет существительное при числительном.
   *
   * Args:
   *   count: Число.
   *   one: Форма для «один».
   *   few: Форма для «два-четыре».
   *   many: Форма для «пять и больше».
   */
  function plural(count: number, one: string, few: string, many: string): string {
    const mod100 = count % 100;
    if (mod100 >= 11 && mod100 <= 14) return many;
    const mod10 = count % 10;
    if (mod10 === 1) return one;
    if (mod10 >= 2 && mod10 <= 4) return few;
    return many;
  }

  function initials(name: string): string {
    return name.slice(0, 2).toUpperCase();
  }
</script>

<svelte:window onkeydown={onWindowKey} />

{#if !session.ready}
  <div class="boot" aria-busy="true">Загружаем…</div>
{:else if !session.username}
  <Login />
{:else}
  <div class="shell">
    <nav class="side" aria-label="Диалоги">
      <div class="side-head">
        <img class="mark" src={logo} alt="" width="41" height="34" />
        <span class="brand">AIst</span>
        <button
          class="theme"
          onclick={() => applyTheme(appearance.theme === "light" ? "dark" : "light")}
          aria-label="Переключить тему"
        >
          {appearance.theme === "light" ? "Светлая" : "Тёмная"}
        </button>
      </div>

      <button class="new" onclick={() => void newConversation()}>+ Новый диалог</button>

      <p class="section mono">Диалоги · {chat.conversations.length}</p>

      <ul class="list">
        {#if !chat.conversations.length}
          <li class="list-empty">Диалогов пока нет</li>
        {/if}
        {#each chat.conversations as item (item.id)}
          <li>
            {#if renamingId === item.id}
              <input
                class="rename"
                bind:value={renameDraft}
                onkeydown={onRenameKey}
                onblur={commitRename}
                aria-label="Название диалога"
                use:focusOnMount
              />
              <span class="rename-hint mono">ENTER</span>
            {:else}
              <button
                class="row"
                class:active={item.id === chat.currentId}
                onclick={() => void openConversation(item.id)}
                ondblclick={() => startRename(item.id, item.title)}
                title={item.title || "Новый диалог"}
              >
                <span class="row-title">{item.title || "Новый диалог"}</span>
              </button>
              <button
                class="drop"
                onclick={() => startRename(item.id, item.title)}
                aria-label="Переименовать диалог"
                title="Переименовать"
              >✎</button>
              <button
                class="drop"
                onclick={() => (confirmingId = confirmingId === item.id ? "" : item.id)}
                aria-label="Удалить диалог"
                title="Удалить"
              >×</button>
            {/if}
          </li>
          {#if confirmingId === item.id}
            <li class="confirm">
              <p>Удалить «{item.title || "Новый диалог"}»? Отменить нельзя.</p>
              <div class="confirm-row">
                <button class="confirm-no" onclick={() => (confirmingId = "")}>Оставить</button>
                <button
                  class="confirm-yes"
                  onclick={() => { confirmingId = ""; void deleteConversation(item.id); }}
                >Удалить</button>
              </div>
            </li>
          {/if}
        {/each}
      </ul>

      <div class="side-foot">
        <span class="avatar" aria-hidden="true">{initials(session.username)}</span>
        <span class="who">{session.username}</span>
        <div class="sizes" role="group" aria-label="Размер текста ответа">
          {#each [["s", "А"], ["m", "А"], ["l", "А"]] as [key, label] (key)}
            <button
              class="size size-{key}"
              class:on={appearance.answerSize === key}
              onclick={() => applyAnswerSize(key as "s" | "m" | "l")}
              aria-pressed={appearance.answerSize === key}
              aria-label={key === "s" ? "Мелкий" : key === "m" ? "Обычный" : "Крупный"}
            >{label}</button>
          {/each}
        </div>
        <button
          class="exit"
          onclick={() => (helpOpen = !helpOpen)}
          aria-pressed={helpOpen}
        >{helpOpen ? "К диалогу" : "Помощь"}</button>
        <button class="exit" onclick={() => void logout()}>Выйти</button>
      </div>
    </nav>

    <main class="main">
      <div class="feed" bind:this={feed} onscroll={onFeedScroll}>
        {#if helpOpen}
          <Help />
        {:else}
        {#if !chat.messages.length && !chat.streaming}
          <div class="empty-state">
            <h1>Спросите живым языком - отвечу с пунктом регламента.</h1>
            <p class="sub">Готовые вопросы по темам базы. Клик - отправляет вопрос.</p>
            <div class="cards">
              {#each STARTERS as card (card.q)}
                <button class="starter" onclick={() => void sendQuestion(card.q, alwaysSearch)}>
                  <span class="starter-area mono">{card.area}</span>
                  <span class="starter-q">{card.q}</span>
                </button>
              {/each}
            </div>
          </div>
        {/if}

        {#each chat.messages as message, index (index)}
          {#if message.role === "user"}
            <div class="bubble user">{message.content}</div>
          {:else}
            <!-- Разметка разбирается своим кодом, HTML из текста
                 экранируется до вывода: см. lib/markdown.ts -->
            <div class="answer" data-answer="1">{@html renderMarkdown(message.content)}</div>
            {#if message.followups?.length}
              <div class="followups">
                {#each message.followups as tip (tip)}
                  <button class="tip" onclick={() => void sendQuestion(tip, alwaysSearch)}>{tip}</button>
                {/each}
              </div>
            {/if}
          {/if}
        {/each}

        {#if chat.stages.length}
          <section class="stages" aria-label="Этапы выполнения" aria-live="polite">
            <p class="stages-head mono">Этапы выполнения</p>
            <p class="stages-summary mono">{stagesSummary}</p>
            {#each chat.stages as stage (stage.key)}
              <div class="stage" class:run={stage.running} class:fail={stage.failed}>
                <span class="tick" aria-hidden="true"></span>
                <span class="stage-title">{stage.title}</span>
                <span class="stage-detail mono">{stage.detail}</span>
              </div>
            {/each}
          </section>
        {/if}

        {#if chat.draftAnswer}
          <div class="answer" data-answer="1" aria-live="polite" aria-atomic="false">
            {@html renderMarkdown(chat.draftAnswer)}
          </div>
        {/if}

        {#if chat.error}
          <p class="error" role="alert">{chat.error}</p>
        {/if}
        {/if}
      </div>

      {#if !helpOpen}
      <div class="composer">
        {#if treeOpen}
          <div class="tree" role="group" aria-label="Области знаний">
            <div class="tree-head">
              <strong>Области знаний</strong>
              <span class="tree-count mono">
                {catalog.selectedAreas.length || catalog.selectedDocs.length
                  ? `выбрано областей: ${catalog.selectedAreas.length} из ${catalog.areas.length}`
                  : "ничего не выбрано - искать по всей базе"}
              </span>
              <button class="tree-reset" onclick={resetTree}>Сбросить</button>
              <button class="tree-close" onclick={() => (treeOpen = false)} aria-label="Закрыть">×</button>
            </div>
            <input
              class="tree-search"
              bind:value={treeQuery}
              placeholder="Поиск по областям и документам"
            />
            <div class="tree-body">
              {#if catalog.error}
                <p class="empty">{catalog.error}</p>
              {:else if !catalog.loaded}
                <p class="empty">Загружаем состав базы…</p>
              {:else}
                {#if !visibleAreas.length}
                  <p class="empty">Ничего не нашлось. Уточните запрос или сбросьте поиск.</p>
                {/if}
                {#each visibleAreas as area (area.area)}
                  <div class="tree-area">
                    <label class="tree-row">
                      <input
                        type="checkbox"
                        checked={areaMark(area) === "all"}
                        indeterminate={areaMark(area) === "some"}
                        aria-checked={areaMark(area) === "some" ? "mixed" : undefined}
                        onchange={() => toggleArea(area.area)}
                      />
                      <span>{area.title}</span>
                      <span class="tree-num mono">{area.documents.length}</span>
                    </label>
                    {#each area.documents as doc (doc.doc_code + area.area)}
                      <label class="tree-row nested">
                        <input
                          type="checkbox"
                          checked={catalog.selectedDocs.includes(doc.doc_code)}
                          onchange={() => toggleDoc(doc.doc_code)}
                        />
                        <span class="mono doc-code">{doc.doc_code}</span>
                        <span class="doc-title">{doc.doc_title}</span>
                      </label>
                    {/each}
                  </div>
                {/each}
              {/if}
            </div>
          </div>
        {/if}

        <div class="input-box">
          <textarea
            bind:value={draft}
            onkeydown={onKey}
            rows="2"
            placeholder="Спросите о регламенте, инструкции, порядке допуска…"
            aria-label="Вопрос"
          ></textarea>
          <div class="controls">
            <button
              class="areas"
              aria-expanded={treeOpen}
              onclick={() => { treeOpen = !treeOpen; void loadCatalog(); }}
            >
              Области: {areaChip}
            </button>
            <label class="switch">
              <input type="checkbox" bind:checked={alwaysSearch} />
              <span>Всегда искать в базе</span>
            </label>
            <span class="hint mono">ENTER - отправить</span>
            {#if chat.streaming}
              <button class="stop" onclick={stopStreaming}>Остановить</button>
            {:else}
              <button class="send" onclick={send} disabled={!draft.trim()} aria-label="Отправить">↑</button>
            {/if}
          </div>
        </div>
      </div>
      {/if}
    </main>

    <Sources sources={chat.liveSources} searching={chat.streaming} {talkMode} />
  </div>
{/if}

<style>
  .boot {
    display: grid;
    place-items: center;
    min-height: 100dvh;
    color: var(--text3);
  }

  .shell {
    display: grid;
    grid-template-columns: 260px minmax(0, 1fr) 340px;
    height: 100dvh;
    background: var(--bg);
  }

  .side {
    display: flex;
    flex-direction: column;
    background: var(--panel);
    border-right: 1px solid var(--line);
    min-width: 0;
  }

  .side-head {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 14px 14px 10px;
  }

  .mark {
    /* Высота ниже 34px съедает тонкие рёбра сети, и птица мажется. */
    width: auto;
    height: 34px;
    flex: none;
  }

  .brand {
    font-size: 15px;
    font-weight: 640;
    letter-spacing: -0.015em;
  }

  .theme {
    margin-left: auto;
    padding: 3px 8px;
    font-size: 11px;
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--text2);
  }

  .new {
    margin: 0 12px 10px;
    padding: 9px 12px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 9px;
    font-weight: 580;
    box-shadow: var(--sh1);
  }

  .new:hover {
    background: var(--hover);
  }

  .section {
    margin: 0 14px 6px;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text3);
  }

  .list {
    flex: 1;
    overflow-y: auto;
    margin: 0;
    padding: 0 8px;
    list-style: none;
  }

  .list li {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .row {
    flex: 1;
    min-width: 0;
    padding: 8px 8px;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text2);
  }

  .row:hover {
    background: var(--hover);
  }

  .row.active {
    background: var(--surface);
    color: var(--text);
    box-shadow: var(--sh1);
  }

  .row-title {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .drop {
    padding: 2px 7px;
    font-size: 15px;
    line-height: 1;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text3);
    opacity: 0;
  }

  .list li:hover .drop,
  .drop:focus-visible {
    opacity: 1;
  }

  .drop:hover {
    background: var(--hover);
    color: var(--acc);
  }

  .list-empty {
    margin: 4px 6px;
    padding: 14px 10px;
    text-align: center;
    font-size: 12px;
    color: var(--text3);
    border: 1px dashed var(--line);
    border-radius: 10px;
  }

  .rename {
    flex: 1;
    min-width: 0;
    height: 30px;
    padding: 0 8px;
    background: var(--surface);
    border: none;
    border-radius: 8px;
    box-shadow: inset 0 0 0 2px var(--pri);
  }

  .rename-hint {
    padding: 0 5px;
    font-size: 9.5px;
    letter-spacing: 0.06em;
    color: var(--text3);
  }

  .confirm {
    margin: 4px 6px 8px;
    padding: 10px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: var(--sh1);
  }

  .confirm p {
    margin: 0 0 8px;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text2);
  }

  .confirm-row {
    display: flex;
    gap: 6px;
  }

  .confirm-no,
  .confirm-yes {
    flex: 1;
    height: 32px;
    font-size: 12px;
    border-radius: 8px;
  }

  .confirm-no {
    background: transparent;
    border: 1px solid var(--line);
    color: var(--text2);
  }

  .confirm-yes {
    background: var(--acc);
    border: none;
    color: #ffffff;
  }

  .sizes {
    display: flex;
    gap: 1px;
    padding: 2px;
    background: var(--line2);
    border-radius: 7px;
  }

  .size {
    width: 22px;
    height: 20px;
    padding: 0;
    background: transparent;
    border: none;
    border-radius: 5px;
    color: var(--text3);
    line-height: 1;
  }

  .size-s {
    font-size: 9px;
  }

  .size-m {
    font-size: 11px;
  }

  .size-l {
    font-size: 13px;
  }

  .size.on {
    background: var(--surface);
    color: var(--text);
    box-shadow: var(--sh1);
  }

  .side-foot {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 14px;
    border-top: 1px solid var(--line);
  }

  .avatar {
    display: grid;
    place-items: center;
    width: 26px;
    height: 26px;
    font-size: 11px;
    background: var(--pri-soft);
    color: var(--pri);
    border-radius: 7px;
  }

  .who {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text2);
  }

  .exit {
    padding: 3px 8px;
    font-size: 11px;
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--text2);
  }

  .main {
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
    min-width: 0;
  }

  .feed {
    overflow-y: auto;
    padding: 22px clamp(16px, 4vw, 44px) 8px;
    display: grid;
    gap: 14px;
    align-content: start;
  }

  .empty-state h1 {
    margin: 8px 0 4px;
    font-size: clamp(21px, 2.2vw, 27px);
    font-weight: 640;
    line-height: 1.2;
    letter-spacing: -0.025em;
  }

  .sub {
    margin: 0 0 16px;
    color: var(--text3);
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 10px;
  }

  .starter {
    display: grid;
    gap: 6px;
    padding: 12px;
    text-align: left;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 11px;
    box-shadow: var(--sh1);
  }

  .starter:hover {
    border-color: var(--pri-line);
  }

  .starter-area {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--proc);
  }

  .starter-q {
    font-size: 13px;
    line-height: 1.4;
  }

  .bubble.user {
    justify-self: end;
    max-width: 72%;
    padding: 10px 13px;
    background: var(--pri);
    color: var(--pri-ink);
    border-radius: 12px 12px 4px 12px;
  }

  .answer {
    font-size: 15px;
    line-height: 1.65;
    text-wrap: pretty;
  }

  .answer :global(p) {
    margin: 0 0 10px;
  }

  .answer :global(p:last-child) {
    margin-bottom: 0;
  }

  .answer :global(h3),
  .answer :global(h4),
  .answer :global(h5) {
    margin: 16px 0 8px;
    font-size: 1.03em;
    font-weight: 620;
    letter-spacing: -0.01em;
  }

  .answer :global(ul),
  .answer :global(ol) {
    margin: 0 0 10px;
    padding-left: 22px;
    display: grid;
    gap: 6px;
  }

  .answer :global(li) {
    padding-left: 2px;
  }

  .answer :global(blockquote) {
    margin: 10px 0;
    padding: 8px 12px;
    border-left: 3px solid var(--pri-line);
    background: var(--surface2);
    color: var(--text2);
  }

  .answer :global(code) {
    padding: 1px 5px;
    font-family: var(--mono);
    font-size: 0.88em;
    background: var(--surface2);
    border: 1px solid var(--line2);
    border-radius: 5px;
  }

  /* Маркер источника: тот же номер, что на карточке справа. */
  .answer :global(sup.cite) {
    display: inline-block;
    min-width: 15px;
    margin: 0 1px;
    padding: 0 3px;
    text-align: center;
    font-size: 0.68em;
    line-height: 1.5;
    vertical-align: super;
    background: var(--pri-soft);
    color: var(--pri);
    border-radius: 4px;
  }

  :global([data-atext="s"]) .answer {
    font-size: 13.5px;
  }

  :global([data-atext="l"]) .answer {
    font-size: 17.5px;
  }

  .followups {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .tip {
    padding: 6px 11px;
    font-size: 12px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--text2);
  }

  .tip:hover {
    border-color: var(--pri-line);
    color: var(--pri);
  }

  .stages {
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--line2);
    border-radius: 10px;
  }

  .stages-head {
    margin: 0 0 8px;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text3);
  }

  .stage {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
    font-size: 12px;
    color: var(--text2);
  }

  .tick {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ok);
    flex: none;
  }

  .stage.run .tick {
    background: var(--proc);
    animation: pulse 1s ease-in-out infinite;
  }

  .stage.fail .tick {
    background: var(--warn);
  }

  @keyframes pulse {
    50% {
      opacity: 0.35;
    }
  }

  .stages-summary {
    display: none;
    margin: 0;
    font-size: 11px;
    color: var(--text3);
  }

  .stage-detail {
    margin-left: auto;
    font-size: 10.5px;
    color: var(--text3);
  }

  .error {
    margin: 0;
    padding: 9px 12px;
    font-size: 12.5px;
    color: var(--warn);
    background: var(--warn-soft);
    border-radius: 9px;
  }

  .composer {
    position: relative;
    padding: 10px clamp(16px, 4vw, 44px) 16px;
  }

  .input-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 13px;
    box-shadow: var(--sh2);
    overflow: hidden;
  }

  textarea {
    width: 100%;
    padding: 12px 14px 4px;
    background: transparent;
    border: none;
    resize: none;
    outline: none;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px 10px 12px;
  }

  .areas {
    padding: 4px 10px;
    font-size: 11.5px;
    background: var(--pri-soft);
    color: var(--pri);
    border: 1px solid var(--pri-line);
    border-radius: 999px;
  }

  .switch {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    color: var(--text2);
    cursor: pointer;
  }

  .hint {
    margin-left: auto;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text3);
  }

  .send {
    width: 30px;
    height: 30px;
    font-size: 15px;
    background: var(--acc);
    color: #ffffff;
    border: none;
    border-radius: 9px;
  }

  .send:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .stop {
    padding: 5px 10px;
    font-size: 11.5px;
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 8px;
    color: var(--text2);
  }

  .tree {
    position: absolute;
    left: clamp(16px, 4vw, 44px);
    right: clamp(16px, 4vw, 44px);
    bottom: calc(100% - 6px);
    max-height: 52vh;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 13px;
    box-shadow: var(--sh2);
    z-index: 10;
  }

  .tree-head {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--line2);
  }

  .tree-count {
    font-size: 10.5px;
    color: var(--text3);
  }

  .tree-reset,
  .tree-close {
    padding: 3px 8px;
    font-size: 11px;
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--text2);
  }

  .tree-reset {
    margin-left: auto;
  }

  .tree-search {
    margin: 10px 14px;
    padding: 7px 10px;
    background: var(--surface2);
    border: 1px solid var(--line);
    border-radius: 8px;
  }

  .tree-body {
    overflow-y: auto;
    padding: 0 10px 12px;
  }

  .tree-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 6px;
    border-radius: 7px;
    cursor: pointer;
  }

  .tree-row:hover {
    background: var(--hover);
  }

  .tree-row.nested {
    padding-left: 26px;
    font-size: 12px;
    color: var(--text2);
  }

  .tree-num {
    margin-left: auto;
    font-size: 10.5px;
    color: var(--text3);
  }

  .doc-code {
    font-size: 11px;
    color: var(--text3);
  }

  .doc-title {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .empty {
    margin: 8px 6px;
    font-size: 12px;
    color: var(--text3);
  }

  @media (max-width: 1100px) {
    .shell {
      grid-template-columns: 232px minmax(0, 1fr);
    }

    /* Снизу появляется полоса источников высотой 52 px - освобождаем под неё
       место, иначе она накрывает кнопку отправки. */
    .composer {
      padding-bottom: 68px;
    }
  }

  @media (max-width: 780px) {
    /* Пять строк этапов вытесняют ответ за пределы экрана. Оставляем одну
       сводку: она отвечает на единственный вопрос, который здесь задают, -
       работает ли система. */
    .stages .stage {
      display: none;
    }

    .stages-summary {
      display: block;
    }

    .stages-head {
      margin-bottom: 4px;
    }
  }

  @media (max-width: 780px) {
    .shell {
      grid-template-columns: minmax(0, 1fr);
    }

    .side {
      display: none;
    }
  }
</style>
