/** Точка входа приложения. */
import { mount } from "svelte";
import "./app.css";
import App from "./App.svelte";

document.documentElement.setAttribute("data-aist", "1");

export default mount(App, { target: document.getElementById("app")! });
