<script lang="ts">
  /** Экран входа. Учётные записи заводит администратор, регистрации нет. */
  import { login } from "./state.svelte";

  let username = $state("");
  let password = $state("");
  let error = $state("");
  let busy = $state(false);

  async function submit(event: SubmitEvent) {
    event.preventDefault();
    if (busy) return;
    error = "";
    busy = true;
    try {
      await login(username.trim(), password);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Не удалось войти";
    } finally {
      busy = false;
    }
  }
</script>

<div class="screen">
  <section class="promo">
    <div class="brand">
      <div class="mark" aria-hidden="true"></div>
      <div>
        <div class="name">АИст</div>
        <div class="tagline">поиск по нормативной базе предприятия</div>
      </div>
    </div>
    <h1>Ответ на рабочий вопрос — со ссылкой на пункт регламента.</h1>
    <ul>
      <li>Регламенты, охрана труда, пожарная и промышленная безопасность, ГО и ЧС.</li>
      <li>Работает на одном ноутбуке, полностью без интернета.</li>
      <li>Если сведений в базе нет — так и скажет.</li>
    </ul>
  </section>

  <section class="form-side">
    <form class="card" onsubmit={submit}>
      <h2>Вход в систему</h2>
      <p class="sub">Учётные записи заводит администратор</p>

      <label for="login-user">Логин</label>
      <input
        id="login-user"
        bind:value={username}
        autocomplete="username"
        required
        disabled={busy}
      />

      <label for="login-pass">Пароль</label>
      <input
        id="login-pass"
        type="password"
        bind:value={password}
        autocomplete="current-password"
        required
        disabled={busy}
      />

      {#if error}
        <p class="error" role="alert">{error}</p>
      {/if}

      <button class="primary" type="submit" disabled={busy}>
        {busy ? "Проверяем…" : "Войти"}
      </button>

      <p class="status"><span class="dot" aria-hidden="true"></span>Локальный контур · сеть не используется</p>
    </form>
  </section>
</div>

<style>
  .screen {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    min-height: 100vh;
    background: var(--bg);
  }

  .promo {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 28px;
    padding: 48px clamp(32px, 5vw, 72px);
    background: var(--panel);
    border-right: 1px solid var(--line);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .mark {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(140deg, var(--proc), var(--pri));
    box-shadow: var(--sh1);
  }

  .name {
    font-size: 19px;
    font-weight: 640;
    letter-spacing: -0.02em;
  }

  .tagline {
    font-size: 12px;
    color: var(--text3);
  }

  h1 {
    margin: 0;
    font-size: clamp(25px, 2.8vw, 36px);
    font-weight: 640;
    line-height: 1.15;
    letter-spacing: -0.028em;
    max-width: 16ch;
  }

  ul {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 10px;
    max-width: 46ch;
  }

  li {
    position: relative;
    padding-left: 20px;
    color: var(--text2);
  }

  li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 8px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--pri-line);
  }

  .form-side {
    display: grid;
    place-items: center;
    padding: 32px;
  }

  .card {
    width: min(360px, 100%);
    display: grid;
    gap: 8px;
    padding: 28px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: var(--sh2);
  }

  h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 640;
    letter-spacing: -0.02em;
  }

  .sub {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--text3);
  }

  label {
    font-size: 12px;
    color: var(--text2);
    margin-top: 6px;
  }

  input {
    padding: 9px 11px;
    background: var(--surface2);
    border: 1px solid var(--line);
    border-radius: 9px;
  }

  input:focus {
    border-color: var(--pri-line);
    box-shadow: 0 0 0 3px var(--focus);
    outline: none;
  }

  .primary {
    margin-top: 14px;
    padding: 10px 14px;
    background: var(--pri);
    color: var(--pri-ink);
    border: none;
    border-radius: 9px;
    font-weight: 580;
  }

  .primary:disabled {
    opacity: 0.6;
    cursor: progress;
  }

  .error {
    margin: 6px 0 0;
    font-size: 12px;
    color: var(--acc);
  }

  .status {
    margin: 14px 0 0;
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11.5px;
    color: var(--text3);
  }

  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ok);
  }

  @media (max-width: 880px) {
    .screen {
      grid-template-columns: 1fr;
    }

    .promo {
      border-right: none;
      border-bottom: 1px solid var(--line);
      padding: 32px 24px;
      gap: 18px;
    }

    ul {
      display: none;
    }
  }
</style>
