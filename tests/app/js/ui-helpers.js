/**
 * UI Helper functions extracted from listeners.js for testing.
 * These are the core logic functions that power the grouped form renderer.
 */

/**
 * Group parameters by their group field.
 * @param {Object} params - Parameter configuration object
 * @returns {Object} - Object with groupedParams and ungroupedParams
 */
export function groupParameters(params) {
    const groupedParams = {};
    const ungroupedParams = {};

    Object.keys(params).forEach(param => {
        const config = params[param];
        const group = config.group || 'Other';

        if (config.group) {
            if (!groupedParams[group]) {
                groupedParams[group] = {};
            }
            groupedParams[group][param] = config;
        } else {
            ungroupedParams[param] = config;
        }
    });

    return { groupedParams, ungroupedParams };
}

/**
 * Determine which groups should be in the advanced panel.
 * @param {Set<string>} allGroups - All available groups
 * @returns {Array<string>} - Groups that belong in advanced panel
 */
export function getAdvancedGroups(allGroups) {
    const advancedGroups = [
        'Knowledge Distillation',
        'Hyperparameter Sweep',
        'Enhanced Evaluation',
        'Reinforcement Learning (PPO)',
        'Advanced Features'
    ];

    return advancedGroups.filter(g => allGroups.has(g));
}

/**
 * Get the standard group ordering.
 * @returns {Array<string>} - Ordered list of group names
 */
export function getGroupOrder() {
    return [
        'Training Hyperparameters',
        'Training Configuration',
        'Data Processing',
        'PEFT/LoRA',
        'DPO/ORPO',
        'Hub Integration',
        'Knowledge Distillation',
        'Hyperparameter Sweep',
        'Enhanced Evaluation',
        'Reinforcement Learning (PPO)',
        'Advanced Features'
    ];
}

/**
 * Check if a task is PPO.
 * @param {string} taskValue - The task value (e.g., 'llm:ppo')
 * @returns {boolean} - True if task is PPO
 */
export function isPPOTask(taskValue) {
    return Boolean(taskValue && taskValue.includes(':ppo'));
}

/**
 * Determine if PPO controls should be enabled based on requirement field.
 * @param {boolean} isPPO - Whether PPO trainer is selected
 * @param {string} requirementValue - Value of the requirement field (reward model path)
 * @returns {boolean} - True if PPO controls should be enabled
 */
export function shouldEnablePPOControls(isPPO, requirementValue) {
    if (!isPPO) {
        return true; // Not PPO, everything enabled
    }
    return Boolean(requirementValue && requirementValue.trim() !== '');
}

/**
 * Get PPO control state for a specific control.
 * @param {HTMLElement} control - The control element
 * @param {boolean} isRequirementFilled - Whether the requirement is filled
 * @param {boolean} isPPO - Whether PPO trainer is selected
 * @returns {Object} - Object with { disabled, opacity, title }
 */
export function getPPOControlState(control, isRequirementFilled, isPPO) {
    // Don't disable the requirement field itself
    const isRequirementField = control.hasAttribute('data-is-ppo-requirement');

    if (!isPPO || isRequirementField) {
        return {
            disabled: false,
            opacity: '1',
            title: ''
        };
    }

    if (!isRequirementFilled) {
        return {
            disabled: true,
            opacity: '0.5',
            title: 'Fill in Reward Model Path first'
        };
    }

    return {
        disabled: false,
        opacity: '1',
        title: ''
    };
}

/**
 * Create HTML for a parameter element with help text and markers.
 * @param {string} param - Parameter name
 * @param {Object} config - Parameter configuration
 * @returns {string} - HTML string
 */
export function createParameterHTML(param, config) {
    const helpIcon = config.help ? `<span class="help-icon" title="${config.help}">ⓘ</span>` : '';
    const requiredMarker = config.is_ppo_requirement ? `<span class="required-marker" title="Required for PPO trainer">*</span>` : '';

    const label = `${config.label}${requiredMarker}${helpIcon}`;

    switch (config.type) {
        case 'number':
            return `<input type="number" id="param_${param}" value="${config.default || ''}" data-label="${label}">`;
        case 'string':
            return `<input type="text" id="param_${param}" value="${config.default || ''}" data-label="${label}">`;
        case 'dropdown':
            const options = config.options.map(opt => `<option value="${opt}">${opt}</option>`).join('');
            return `<select id="param_${param}" data-label="${label}">${options}</select>`;
        case 'checkbox':
            return `<input type="checkbox" id="param_${param}" ${config.default ? 'checked' : ''} data-label="${label}">`;
        case 'textarea':
            return `<textarea id="param_${param}" data-label="${label}">${config.default || ''}</textarea>`;
        default:
            return '';
    }
}
