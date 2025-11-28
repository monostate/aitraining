(function (root, factory) {
    if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.WandbPanelHelpers = factory();
    }
})(typeof self !== "undefined" ? self : this, function () {
    function sanitizePath(runDir) {
        if (!runDir || typeof runDir !== "string") {
            return "";
        }
        return runDir.trim();
    }

    function formatCommand(runDir) {
        const resolved = sanitizePath(runDir);
        if (!resolved) {
            return "";
        }
        return `WANDB_DIR="${resolved}" wandb beta leet "${resolved}"`;
    }

    function deriveStatus(response) {
        const lines = Array.isArray(response && response.lines) ? response.lines : [];
        const command =
            (response && response.command) ||
            (response && response.run_dir ? formatCommand(response.run_dir) : "");
        const message = response && response.message ? response.message : null;
        const text =
            lines.length > 0
                ? lines.join("\n")
                : message || "Waiting for W&B visualizer output...";

        return {
            active: Boolean(response && response.active),
            command,
            text,
            error: response && response.error ? response.error : null,
        };
    }

    return {
        formatCommand,
        deriveStatus,
    };
});

