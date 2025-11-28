(function () {
    const panel = document.getElementById("wandb-visualizer-panel");
    if (!panel) {
        return;
    }

    const commandEl = document.getElementById("wandb-visualizer-command");
    const outputEl = document.getElementById("wandb-visualizer-output");
    const alertEl = document.getElementById("wandb-visualizer-alert");
    const copyBtn = document.getElementById("wandb-copy-command");
    const helpers = window.WandbPanelHelpers || {};
    const state = {
        lastCommand: "",
    };

    function setCommand(command) {
        if (!command) {
            return;
        }
        state.lastCommand = command;
        commandEl.textContent = command;
    }

    function setOutput(text) {
        outputEl.textContent = text;
    }

    function showPanel() {
        panel.classList.remove("hidden");
    }

    function hidePanel() {
        panel.classList.add("hidden");
    }

    function setAlert(message) {
        if (message) {
            alertEl.textContent = message;
            alertEl.classList.remove("hidden");
        } else {
            alertEl.textContent = "";
            alertEl.classList.add("hidden");
        }
    }

    async function pollStatus() {
        try {
            const response = await fetch("/ui/wandb_visualizer/status");
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            const status = helpers.deriveStatus
                ? helpers.deriveStatus(data)
                : {
                      command: data && data.command ? data.command : "",
                      text: Array.isArray(data && data.lines) ? data.lines.join("\n") : data.message || "",
                      error: data && data.error ? data.error : null,
                      active: Boolean(data && data.active),
                  };

            if (!status || (!status.command && !status.text && !status.error)) {
                if (data && data.message) {
                    showPanel();
                    setOutput(data.message);
                    setCommand(status && status.command ? status.command : "");
                    setAlert(null);
                } else {
                    hidePanel();
                }
                return;
            }

            showPanel();
            if (status.command) {
                setCommand(status.command);
            }
            setOutput(status.text || "Waiting for W&B visualizer output...");
            setAlert(status.error ? "W&B visualizer error: " + status.error : null);
        } catch (error) {
            // Silent failure; UI polling will retry
        }
    }

    if (copyBtn) {
        copyBtn.addEventListener("click", async function () {
            if (!state.lastCommand) {
                return;
            }
            try {
                await navigator.clipboard.writeText(state.lastCommand);
                setAlert("Command copied to clipboard.");
                setTimeout(() => setAlert(null), 2000);
            } catch (err) {
                setAlert("Failed to copy command.");
            }
        });
    }

    window.WandbPanel = {
        updateCommand(command, error) {
            if (command) {
                showPanel();
                setCommand(command);
            }
            if (error) {
                setAlert("W&B visualizer error: " + error);
            } else if (command) {
                setAlert(null);
            }
        },
    };

    setInterval(pollStatus, 5000);
    pollStatus();
})();

