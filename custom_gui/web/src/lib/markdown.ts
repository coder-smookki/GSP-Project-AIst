/** Разбор разметки ответа в HTML.
 *
 * Модель отвечает разметкой: заголовки, нумерованные списки, выделения.
 * Без разбора всё это видно как звёздочки и решётки посреди текста.
 *
 * Почему свой разбор, а не библиотека. Нужен узкий набор - тот, что реально
 * встречается в ответах по регламентам, - и полная невозможность вставить
 * свою разметку. Здесь HTML экранируется ПЕРВЫМ делом, поэтому ни один тег
 * из текста документа не доживает до вывода: в результате остаются только
 * теги, которые построил этот файл. Готовые разборщики пропускают HTML
 * насквозь и требуют отдельной очистки, а это ещё шестьдесят килобайт.
 */

/** Экранирует символы, из которых можно собрать тег. */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Размечает выделения и маркеры цитат внутри строки.
 *
 * Маркер `[1]` становится ссылкой на карточку источника: по нему
 * подсвечивается нужная выписка справа.
 */
function inline(text: string): string {
  return (
    escapeHtml(text)
      // Порядок важен: тройные звёздочки разбираются до двойных и одиночных.
      .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*(?!\s)(.+?)(?<!\s)\*/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d{1,2})\]/g, '<sup class="cite" data-n="$1">$1</sup>')
  );
}

/** Закрывает открытый список, если он был. */
function closeList(out: string[], state: { list: "ul" | "ol" | null }): void {
  if (state.list) {
    out.push(`</${state.list}>`);
    state.list = null;
  }
}

/** Открывает список нужного вида, закрыв прежний. */
function openList(
  out: string[],
  state: { list: "ul" | "ol" | null },
  kind: "ul" | "ol",
): void {
  if (state.list !== kind) {
    closeList(out, state);
    out.push(`<${kind}>`);
    state.list = kind;
  }
}

/**
 * Превращает разметку ответа в безопасный HTML.
 *
 * Поддерживается то, что встречается в ответах по регламентам: заголовки,
 * маркированные и нумерованные списки, цитаты, выделения, моноширинные
 * вставки и маркеры источников. Таблицы и ссылки не поддерживаются
 * намеренно: первых модель не даёт, вторых в замкнутом контуре не бывает.
 *
 * @param source Текст ответа.
 * @returns HTML, в котором нет ни одного тега из исходного текста.
 */
export function renderMarkdown(source: string): string {
  const out: string[] = [];
  const state: { list: "ul" | "ol" | null } = { list: null };
  let paragraph: string[] = [];

  const flush = () => {
    if (paragraph.length) {
      out.push(`<p>${inline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  for (const raw of source.split("\n")) {
    const line = raw.trimEnd();

    if (!line.trim()) {
      flush();
      closeList(out, state);
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      closeList(out, state);
      // Заголовки уровнями ниже: h1 на слайде ответа был бы крупнее вопроса.
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const ordered = /^\s*(\d{1,2})[.)]\s+(.*)$/.exec(line);
    if (ordered) {
      flush();
      openList(out, state, "ol");
      out.push(`<li>${inline(ordered[2])}</li>`);
      continue;
    }

    const bullet = /^\s*[-*•]\s+(.*)$/.exec(line);
    if (bullet) {
      flush();
      openList(out, state, "ul");
      out.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }

    const quote = /^\s*>\s?(.*)$/.exec(line);
    if (quote) {
      flush();
      closeList(out, state);
      out.push(`<blockquote>${inline(quote[1])}</blockquote>`);
      continue;
    }

    paragraph.push(line.trim());
  }

  flush();
  closeList(out, state);
  return out.join("");
}
