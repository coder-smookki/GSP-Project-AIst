/** Клиент бэкенда интерфейса.
 *
 * Браузер знает только про эти маршруты. К агенту он не ходит: токен и
 * личность подставляет сервер, поэтому подменить пользователя из вкладки
 * нельзя (см. deploy/auth.py).
 */

export interface Conversation {
  id: string;
  title: string;
  updated_at: string;
  messages: number;
}

export interface Source {
  n: number;
  doc_code: string;
  doc_title: string;
  clause: string | null;
  section_no?: string;
  section_title?: string;
  breadcrumb?: string;
  score?: number;
  snippet?: string;
  text?: string;
}

export interface Stage {
  key: string;
  title: string;
  running: boolean;
  failed: boolean;
  detail: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  followups?: string[];
  stages?: Stage[];
  created_at?: string;
}

export interface AreaNode {
  area: string;
  title: string;
  documents: DocumentNode[];
}

export interface DocumentNode {
  doc_code: string;
  doc_title: string;
  chunks: number;
  section_types: { key: string; title: string }[];
}

export interface DocumentChunk {
  chunk_id: string;
  order_no: number;
  section_no: string;
  section_title: string;
  clause: string | null;
  breadcrumb: string;
  text: string;
  is_table_region: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `Запрос не выполнен (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // Тело может быть не JSON - тогда остаётся код ответа.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  me: () => request<{ username: string; role: string }>("/api/me"),

  login: (username: string, password: string) =>
    request<{ username: string; role: string }>("/api/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  logout: () => request<{ ok: boolean }>("/api/logout", { method: "POST" }),

  conversations: () =>
    request<{ items: Conversation[] }>("/api/conversations").then((r) => r.items),

  createConversation: () =>
    request<Conversation>("/api/conversations", { method: "POST" }),

  conversation: (id: string) =>
    request<{ id: string; messages: Message[] }>(
      `/api/conversations/${encodeURIComponent(id)}`,
    ),

  rename: (id: string, title: string) =>
    request<{ title: string }>(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),

  remove: (id: string) =>
    request<{ ok: boolean }>(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  tree: () =>
    request<{ areas: AreaNode[] }>("/api/knowledge/tree").then((r) => r.areas),

  document: (code: string) =>
    request<{ doc_code: string; doc_title: string; chunks: DocumentChunk[] }>(
      `/api/knowledge/documents/${encodeURIComponent(code)}`,
    ),
};

/** Событие потока ответа. */
export type AskEvent =
  | { type: "title"; title: string }
  | { type: "stage"; stage: Stage }
  | { type: "token"; text: string }
  | { type: "sources"; items: Source[] }
  | { type: "followups"; items: string[] }
  | { type: "error"; message: string }
  | { type: "done"; sources: Source[]; followups: string[] };

export interface AskOptions {
  areas?: string[];
  docs?: string[];
  sections?: string[];
  alwaysSearch?: boolean;
  signal?: AbortSignal;
}

/**
 * Задаёт вопрос и отдаёт события по мере поступления.
 *
 * Используется fetch со чтением тела, а не EventSource: тот умеет только GET,
 * не передаёт тело запроса и не позволяет прервать ответ на полуслове.
 */
export async function* ask(
  conversationId: string,
  question: string,
  options: AskOptions = {},
): AsyncGenerator<AskEvent> {
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        areas: options.areas ?? [],
        docs: options.docs ?? [],
        sections: options.sections ?? [],
        always_search: options.alwaysSearch ?? false,
      }),
      signal: options.signal,
    },
  );

  if (!response.ok || !response.body) {
    throw new ApiError(`Агент не ответил (${response.status})`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // События разделены пустой строкой. Хвост буфера может быть неполным
    // событием - он остаётся до следующего куска.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseEvent(raw);
      if (parsed) yield parsed;
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseEvent(raw: string): AskEvent | null {
  let name = "";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!name || !data) return null;

  let payload: any;
  try {
    payload = JSON.parse(data);
  } catch {
    return null;
  }

  switch (name) {
    case "title":
      return { type: "title", title: String(payload.title ?? "") };
    case "stage":
      return { type: "stage", stage: payload as Stage };
    case "token":
      return { type: "token", text: String(payload.text ?? "") };
    case "sources":
      return { type: "sources", items: (payload.items ?? []) as Source[] };
    case "followups":
      return { type: "followups", items: (payload.items ?? []) as string[] };
    case "error":
      return { type: "error", message: String(payload.message ?? "Сбой") };
    case "done":
      return {
        type: "done",
        sources: (payload.sources ?? []) as Source[],
        followups: (payload.followups ?? []) as string[],
      };
    default:
      return null;
  }
}
