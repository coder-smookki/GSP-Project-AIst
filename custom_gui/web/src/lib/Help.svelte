<script lang="ts">
  /** Экран «Помощь»: вопросы, которые задают чаще всего.
   *
   * Текст перенесён из прототипа дословно - это авторский продуктовый
   * материал, а не заполнитель. Первый пункт объясняет честный отказ:
   * без объяснения его читают как поломку, хотя это правильный ответ.
   */

  /** Раскрыт может быть только один пункт: иначе список превращается в
   *  сплошную простыню, и найти нужное сложнее, чем прокрутить. */
  let openIndex = $state(-1);

  const FAQ = [
    {
      q: "Почему система пишет «В базе знаний информации нет»?",
      a: "Потому что подходящего пункта в загруженных документах не нашлось. Это правильный ответ: альтернатива — выдуманный. Попробуйте расширить области поиска или переформулировать вопрос словами, близкими к тексту регламента.",
    },
    {
      q: "Можно ли открыть сам документ?",
      a: "Документы хранятся внутри системы, внешнего адреса у них нет. Открыть документ можно прямо в интерфейсе — кнопкой в карточке источника: он собирается из фрагментов и показывается по порядку чтения. Сама карточка сделана как выписка: код документа, путь до пункта и цитата — этого достаточно, чтобы найти место в бумажном или учётном экземпляре.",
    },
    {
      q: "Что означают номера в тексте ответа?",
      a: "Каждый номер — конкретная выписка справа. Нумерацию задаёт система в порядке отбора; она не меняется, пока вы читаете ответ.",
    },
    {
      q: "Почему в карточке иногда нет номера пункта?",
      a: "В некоторых документах фрагмент относится к разделу целиком. Тогда вместо пункта показывается раздел и его название — это запасной ориентир.",
    },
    {
      q: "Зачем система разбивает вопрос на несколько запросов?",
      a: "Сложный вопрос ищется частями одновременно: так находятся все нужные документы, а не только самый очевидный. В ленте этапов видно каждый подзапрос и сколько фрагментов он нашёл.",
    },
    {
      q: "Что делает переключатель «Всегда искать в базе»?",
      a: "Включённым он заставляет систему идти в документы даже на короткие реплики. Выключенным — на «спасибо» или «привет» система просто ответит, не тратя время на поиск.",
    },
    {
      q: "Нужен ли интернет?",
      a: "Нет. Система полностью работает на этом ноутбуке: поиск, ответ и база документов локальные. Отключение сети ничего не меняет.",
    },
    {
      q: "Как завести нового пользователя?",
      a: "Самостоятельной регистрации нет. Учётные записи и права выдаёт администратор системы.",
    },
  ];

  function toggle(index: number) {
    openIndex = openIndex === index ? -1 : index;
  }
</script>

<div class="help">
  <h1>Помощь</h1>
  <p class="lead">Восемь вопросов, которые задают чаще всего.</p>

  <div class="faq">
    {#each FAQ as item, index (item.q)}
      <div class="item">
        <button
          class="q"
          onclick={() => toggle(index)}
          aria-expanded={openIndex === index}
        >
          <span>{item.q}</span>
          <span class="chevron" class:open={openIndex === index} aria-hidden="true"></span>
        </button>
        {#if openIndex === index}
          <p class="a">{item.a}</p>
        {/if}
      </div>
    {/each}
  </div>

  <div class="keys">
    <h2>Клавиатура</h2>
    <ul>
      <li><span class="key mono">Enter</span> отправить вопрос</li>
      <li><span class="key mono">Shift+Enter</span> перенос строки</li>
      <li><span class="key mono">Esc</span> закрыть панель областей</li>
      <li><span class="key mono">Tab</span> обход интерфейса с клавиатуры</li>
    </ul>
  </div>
</div>

<style>
  .help {
    max-width: 64ch;
    padding: 26px clamp(16px, 4vw, 44px);
  }

  h1 {
    margin: 0 0 4px;
    font-size: clamp(21px, 2.2vw, 27px);
    font-weight: 640;
    line-height: 1.2;
    letter-spacing: -0.025em;
  }

  .lead {
    margin: 0 0 20px;
    color: var(--text3);
  }

  .faq {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface);
    overflow: hidden;
  }

  /* У последнего пункта своей черты нет: иначе она сложилась бы с рамкой
     контейнера в заметный двойной волосок. */
  .item:not(:last-child) {
    border-bottom: 1px solid var(--line2);
  }

  .q {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 13px 16px;
    text-align: left;
    background: transparent;
    border: none;
    font-size: 13.5px;
    font-weight: 580;
  }

  .q:hover {
    background: var(--hover);
  }

  .chevron {
    margin-left: auto;
    width: 7px;
    height: 7px;
    border-right: 1.6px solid var(--text3);
    border-bottom: 1.6px solid var(--text3);
    transform: rotate(45deg);
    transform-origin: center;
  }

  .chevron.open {
    transform: rotate(225deg);
  }

  .a {
    margin: 0;
    padding: 0 16px 15px;
    max-width: 64ch;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text2);
  }

  .keys {
    margin-top: 24px;
  }

  h2 {
    margin: 0 0 10px;
    font-size: 15px;
    font-weight: 620;
    letter-spacing: -0.015em;
  }

  ul {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 8px;
  }

  li {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12.5px;
    color: var(--text2);
  }

  .key {
    min-width: 92px;
    padding: 2px 7px;
    text-align: center;
    font-size: 10.5px;
    background: var(--surface2);
    border: 1px solid var(--line);
    border-radius: 6px;
    color: var(--text3);
  }
</style>
