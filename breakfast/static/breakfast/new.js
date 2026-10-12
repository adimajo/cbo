// Frontend of the "new breakfast" page. Plain ES modules, no dependency: state lives on the server and every
// change is sent to it, the response being the new state to render.

const app = document.getElementById("breakfast-app");
const $ = (selector) => app.querySelector(selector);
const message = (key) => document.querySelector(`[data-msg="${key}"]`).textContent;

const dateInput = $("#date");
const teamInput = $("#include-optional-team");
const countOutput = $("#participant-count");
const absenteesBox = $("#absentees");
const hostsList = $("#hosts");
const noHosts = $("#no-hosts");
const designated = $("#designated");
const statusLine = $("#status");
const another = $("#another");

let state = null;
let queue = Promise.resolve();

function setStatus(text, kind = "") {
    statusLine.textContent = text;
    statusLine.className = kind;
}

async function post(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": app.dataset.csrf },
        body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || message("error"));
    return data;
}

function payload(action, extra) {
    return {
        action,
        date: dateInput.value,
        include_optional_team: teamInput ? teamInput.checked : true,
        absentees: [...absenteesBox.querySelectorAll("input:checked")].map((input) => input.value),
        hosts: state.hosts.map(({ email, homemade }) => ({ email, homemade })),
        ...extra,
    };
}

// Requests are chained so that a slow response never overwrites a newer one.
function send(action = "update", extra = {}) {
    queue = queue.then(async () => {
        setStatus(message("saving"));
        try {
            state = await post(app.dataset.updateUrl.replace("TOKEN", state.token), payload(action, extra));
            render();
            setStatus(state.confirmed ? message("confirmed") : message("saved"), state.confirmed ? "success" : "");
        } catch (error) {
            setStatus(error.message, "error");
        }
    });
    return queue;
}

function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}

let absenteesSignature = "";
function renderAbsentees() {
    const signature = JSON.stringify(state.possible_users.map((user) => user.email));
    if (signature !== absenteesSignature) {
        absenteesSignature = signature;
        absenteesBox.replaceChildren(
            ...state.possible_users.map((user) => {
                const input = el("input", { type: "checkbox", value: user.email });
                input.addEventListener("change", () => send());
                return el("label", {}, input, user.name);
            }),
        );
    }
    for (const input of absenteesBox.querySelectorAll("input")) {
        input.checked = state.absentees.includes(input.value);
    }
}

function renderHosts() {
    hostsList.replaceChildren(
        ...state.hosts.map((host) => {
            const homemade = el("input", { type: "checkbox", checked: host.homemade });
            homemade.addEventListener("change", () => {
                host.homemade = homemade.checked;
                send();
            });
            const remove = el("button", { type: "button", textContent: message("remove") });
            remove.addEventListener("click", () => {
                state.hosts = state.hosts.filter((other) => other.email !== host.email);
                send();
            });
            return el("li", {}, el("span", { className: "name" }, host.name), el("label", {}, homemade, message("homemade")), remove);
        }),
    );
    noHosts.hidden = state.hosts.length > 0;
}

function renderDesignated() {
    const taken = new Set([...state.absentees, ...state.hosts.map((host) => host.email)]);
    designated.replaceChildren(
        el("option", { value: "", textContent: message("choose") }),
        ...state.possible_users
            .filter((user) => !taken.has(user.email))
            .map((user) => el("option", { value: user.email, textContent: user.name })),
    );
}

function render() {
    dateInput.value = state.date;
    if (teamInput) teamInput.checked = state.include_optional_team;
    countOutput.textContent = state.participant_count;
    renderAbsentees();
    renderHosts();
    renderDesignated();
    for (const control of app.querySelectorAll("input, select, button")) control.disabled = state.confirmed;
    another.hidden = !state.confirmed;
}

async function start() {
    try {
        state = await post(app.dataset.createUrl, {});
    } catch (error) {
        app.hidden = false;
        setStatus(error.message, "error");
        return;
    }
    render();
    app.hidden = false;

    dateInput.addEventListener("change", () => send());
    teamInput?.addEventListener("change", () => send());
    $("#draw").addEventListener("click", () => send("draw"));
    $("#add-designated").addEventListener("click", () => {
        if (designated.value) send("add_host", { email: designated.value });
    });
    $("#confirm").addEventListener("click", () => send("confirm"));
}

start();
