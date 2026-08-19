/** Общее состояние приложения.
 *
 * Экспортируются объекты, а не переменные: у Svelte 5 переприсваивание
 * экспортированного `$state` через границу модуля не отслеживается, а правка
 * поля объекта - отслеживается.
 */

import {
  api,
  ask,
  ApiError,
  type AreaNode,
  type Conversation,
  type Message,
  type Source,
  type Stage,
} from "./api";

type Theme = "light" | "dark";
type AnswerSize = "s" | "m" | "l";

const THEME_KEY = "aist:theme";
const SIZE_KEY = "aist:answer-size";

/** Кто вошёл. */
export const session = $state<{
  username: string;
  role: string;
  ready: boolean;
}>({ username: "", role: "", ready: false });

/** Оформление: тема и размер текста ответа. */
export const appearance = $state<{ theme: Theme; answerSize: AnswerSize }>({
  theme: "light",
  answerSize: "m",
});

/** Диалоги и открытая переписка. */
export const chat = $state<{
  conversations: Conversation[];
  currentId: string;
  messages: Message[];
  streaming: boolean;
  stages: Stage[];
  draftAnswer: string;
  liveSources: Source[];
  followups: string[];
  error: string;
}>({
  conversations: [],
  currentId: "",
  messages: [],
  streaming: false,
  stages: [],
  draftAnswer: "",
  liveSources: [],
  followups: [],
  error: "",
});

/** Дерево областей знаний и выбор в нём. */
export const catalog = $state<{
  areas: AreaNode[];
  loaded: boolean;
  error: string;
  selectedAreas: string[];
  selectedDocs: string[];
}>({
  areas: [],
  loaded: false,
  error: "",
  selectedAreas: [],
  selectedDocs: [],
});

let controller: AbortController | null = null;

/** Применяет тему к документу и запоминает выбор. */
export function applyTheme(theme: Theme): void {
  appearance.theme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Приватный режим запрещает запись - тема просто не переживёт перезагрузку.
  }
}

/** Применяет размер текста ответа и запоминает выбор. */
export function applyAnswerSize(size: AnswerSize): void {
  appearance.answerSize = size;
  document.documentElement.setAttribute("data-atext", size);
  try {
    localStorage.setItem(SIZE_KEY, size);
  } catch {
    // См. выше.
  }
}

/** Восстанавливает оформление из прошлого сеанса. */
export function restoreAppearance(): void {
  let theme: Theme = "light";
  let size: AnswerSize = "m";
  try {
    const savedTheme = localStorage.getItem(THEME_KEY);
    if (savedTheme === "dark" || savedTheme === "light") theme = savedTheme;
    const savedSize = localStorage.getItem(SIZE_KEY);
    if (savedSize === "s" || savedSize === "m" || savedSize === "l") size = savedSize;
  } catch {
    // Нет доступа к хранилищу - остаются умолчания.
  }
  applyTheme(theme);
  applyAnswerSize(size);
}

/** Проверяет, есть ли действующая сессия. */
export async function loadSession(): Promise<void> {
  try {
    const who = await api.me();
    session.username = who.username;
    session.role = who.role;
  } catch {
    session.username = "";
    session.role = "";
  } finally {
    session.ready = true;
  }
}

/** Входит в систему и подтягивает список диалогов. */
export async function login(username: string, password: string): Promise<void> {
  const who = await api.login(username, password);
  session.username = who.username;
  session.role = who.role;
  await loadConversations();
}

/** Выходит из системы и очищает состояние. */
export async function logout(): Promise<void> {
  await api.logout();
  session.username = "";
  session.role = "";
  chat.conversations = [];
  chat.messages = [];
  chat.currentId = "";
  catalog.selectedAreas = [];
  catalog.selectedDocs = [];
}

/** Загружает список диалогов. */
export async function loadConversations(): Promise<void> {
  try {
    chat.conversations = await api.conversations();
  } catch (error) {
    chat.error = describe(error);
  }
}

/** Создаёт диалог и делает его текущим. */
export async function newConversation(): Promise<void> {
  try {
    const created = await api.createConversation();
    chat.conversations = [created, ...chat.conversations];
    await openConversation(created.id);
  } catch (error) {
    chat.error = describe(error);
  }
}

/** Открывает диалог и подтягивает его сообщения. */
export async function openConversation(id: string): Promise<void> {
  stopStreaming();
  chat.currentId = id;
  chat.stages = [];
  chat.draftAnswer = "";
  chat.liveSources = [];
  chat.followups = [];
  chat.error = "";
  try {
    const loaded = await api.conversation(id);
    chat.messages = loaded.messages;
    const last = loaded.messages[loaded.messages.length - 1];
    if (last?.role === "assistant") {
      chat.liveSources = last.sources ?? [];
      chat.followups = last.followups ?? [];
    }
  } catch (error) {
    chat.error = describe(error);
    chat.messages = [];
  }
}

/** Переименовывает диалог. */
export async function renameConversation(id: string, title: string): Promise<void> {
  try {
    const saved = await api.rename(id, title);
    const found = chat.conversations.find((item) => item.id === id);
    if (found) found.title = saved.title;
  } catch (error) {
    chat.error = describe(error);
  }
}

/** Удаляет диалог; если он был открыт, открывает соседний. */
export async function deleteConversation(id: string): Promise<void> {
  try {
    await api.remove(id);
    chat.conversations = chat.conversations.filter((item) => item.id !== id);
    if (chat.currentId === id) {
      chat.currentId = "";
      chat.messages = [];
      chat.liveSources = [];
      chat.followups = [];
      const next = chat.conversations[0];
      if (next) await openConversation(next.id);
    }
  } catch (error) {
    chat.error = describe(error);
  }
}

/** Загружает дерево областей знаний из индекса. */
export async function loadCatalog(): Promise<void> {
  if (catalog.loaded) return;
  try {
    catalog.areas = await api.tree();
    catalog.error = "";
    catalog.loaded = true;
  } catch (error) {
    catalog.error = describe(error);
  }
}

/** Прерывает текущий ответ. */
export function stopStreaming(): void {
  controller?.abort();
  controller = null;
  chat.streaming = false;
}

/**
 * Задаёт вопрос и наполняет состояние по мере ответа.
 *
 * Если диалога ещё нет, он заводится: человек может начать печатать сразу,
 * не нажимая «Новый диалог».
 */
export async function sendQuestion(
  question: string,
  alwaysSearch: boolean,
): Promise<void> {
  const text = question.trim();
  if (!text || chat.streaming) return;

  if (!chat.currentId) {
    await newConversation();
    if (!chat.currentId) return;
  }

  const conversationId = chat.currentId;
  chat.error = "";
  chat.stages = [];
  chat.draftAnswer = "";
  chat.liveSources = [];
  chat.followups = [];
  chat.messages = [...chat.messages, { role: "user", content: text }];
  chat.streaming = true;

  controller = new AbortController();
  const sources: Source[] = [];
  let followups: string[] = [];

  try {
    for await (const event of ask(conversationId, text, {
      areas: catalog.selectedAreas,
      docs: catalog.selectedDocs,
      alwaysSearch,
      signal: controller.signal,
    })) {
      switch (event.type) {
        case "title": {
          const found = chat.conversations.find((c) => c.id === conversationId);
          if (found && !found.title) found.title = event.title;
          break;
        }
        case "stage": {
          const at = chat.stages.findIndex((s) => s.key === event.stage.key);
          if (at === -1) chat.stages = [...chat.stages, event.stage];
          else chat.stages[at] = event.stage;
          break;
        }
        case "token":
          chat.draftAnswer += event.text;
          break;
        case "sources":
          // Пустой список - это сигнал «убрать прошлые источники», а не
          // отсутствие события: иначе панель показывала бы выдержки к
          // предыдущему вопросу.
          sources.length = 0;
          sources.push(...event.items);
          chat.liveSources = [...sources];
          break;
        case "followups":
          followups = event.items;
          chat.followups = followups;
          break;
        case "error":
          chat.error = event.message;
          break;
        case "done":
          chat.liveSources = event.sources;
          chat.followups = event.followups;
          break;
      }
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      chat.error = describe(error);
    }
  } finally {
    chat.streaming = false;
    controller = null;
    if (chat.draftAnswer || chat.liveSources.length) {
      chat.messages = [
        ...chat.messages,
        {
          role: "assistant",
          content: chat.draftAnswer,
          sources: chat.liveSources,
          followups: chat.followups,
          stages: chat.stages,
        },
      ];
    }
    chat.draftAnswer = "";
    await loadConversations();
  }
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Непредвиденная ошибка";
}
