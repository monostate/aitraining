document.addEventListener('DOMContentLoaded', function () {
    const dataSource = document.getElementById("dataset_source");
    const uploadDataTabContent = document.getElementById("upload-data-tab-content");
    const hubDataTabContent = document.getElementById("hub-data-tab-content");
    const uploadDataTabs = document.getElementById("upload-data-tabs");

    const jsonCheckbox = document.getElementById('show-json-parameters');
    const jsonParametersDiv = document.getElementById('json-parameters');
    const dynamicUiDiv = document.getElementById('dynamic-ui');

    const paramsTextarea = document.getElementById('params_json');

    const updateTextarea = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        const params = {};
        paramElements.forEach(el => {
            const key = el.id.replace('param_', '');
            params[key] = el.value;
        });
        paramsTextarea.value = JSON.stringify(params, null, 2);
        //paramsTextarea.className = 'p-2.5 w-full text-sm text-gray-600 border-white border-transparent focus:border-transparent focus:ring-0'
        paramsTextarea.style.height = '600px';
    };
    const observeParamChanges = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        paramElements.forEach(el => {
            el.addEventListener('input', updateTextarea);
        });
    };
    const updateParamsFromTextarea = () => {
        try {
            const params = JSON.parse(paramsTextarea.value);
            Object.keys(params).forEach(key => {
                const el = document.getElementById('param_' + key);
                if (el) {
                    el.value = params[key];
                }
            });
        } catch (e) {
            console.error('Invalid JSON:', e);
        }
    };
    function switchToJSON() {
        if (jsonCheckbox.checked) {
            dynamicUiDiv.style.display = 'none';
            jsonParametersDiv.style.display = 'block';
        } else {
            dynamicUiDiv.style.display = 'block';
            jsonParametersDiv.style.display = 'none';
        }
    }

    function handleDataSource() {
        if (dataSource.value === "hub") {
            uploadDataTabContent.style.display = "none";
            uploadDataTabs.style.display = "none";
            hubDataTabContent.style.display = "block";
        } else if (dataSource.value === "local") {
            uploadDataTabContent.style.display = "block";
            uploadDataTabs.style.display = "block";
            hubDataTabContent.style.display = "none";
        }
    }

    async function fetchParams() {
        const taskValue = document.getElementById('task').value;
        const parameterMode = document.getElementById('parameter_mode').value;
        const response = await fetch(`/ui/params/${taskValue}/${parameterMode}`);
        const params = await response.json();
        return params;
    }

    function createElement(param, config) {
        const helpIcon = config.help ? `<span class="ml-1 text-gray-400 cursor-help" title="${config.help}">ⓘ</span>` : '';
        const requiredMarker = config.is_ppo_requirement ? `<span class="ml-1 text-red-500" title="Required for PPO trainer">*</span>` : '';

        let element = '';
        switch (config.type) {
            case 'number':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">
                        ${config.label}${requiredMarker}${helpIcon}
                    </label>
                    <input type="number" name="param_${param}" id="param_${param}" value="${config.default}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        ${config.required_for_ppo ? 'data-required-for-ppo="true"' : ''}
                        ${config.is_ppo_requirement ? 'data-is-ppo-requirement="true"' : ''}>
                </div>`;
                break;
            case 'dropdown':
                let options = config.options.map(option => `<option value="${option}" ${option === config.default ? 'selected' : ''}>${option}</option>`).join('');
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">
                        ${config.label}${requiredMarker}${helpIcon}
                    </label>
                    <select name="param_${param}" id="param_${param}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        ${config.required_for_ppo ? 'data-required-for-ppo="true"' : ''}
                        ${config.is_ppo_requirement ? 'data-is-ppo-requirement="true"' : ''}>
                        ${options}
                    </select>
                </div>`;
                break;
            case 'checkbox':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">
                        ${config.label}${requiredMarker}${helpIcon}
                    </label>
                    <input type="checkbox" name="param_${param}" id="param_${param}" ${config.default ? 'checked' : ''}
                        class="mt-1 text-xs font-medium border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        ${config.required_for_ppo ? 'data-required-for-ppo="true"' : ''}
                        ${config.is_ppo_requirement ? 'data-is-ppo-requirement="true"' : ''}>
                </div>`;
                break;
            case 'string':
            case 'textarea':
                const inputElement = config.type === 'textarea'
                    ? `<textarea name="param_${param}" id="param_${param}" rows="3"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        ${config.required_for_ppo ? 'data-required-for-ppo="true"' : ''}
                        ${config.is_ppo_requirement ? 'data-is-ppo-requirement="true"' : ''}>${config.default || ''}</textarea>`
                    : `<input type="text" name="param_${param}" id="param_${param}" value="${config.default || ''}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        ${config.required_for_ppo ? 'data-required-for-ppo="true"' : ''}
                        ${config.is_ppo_requirement ? 'data-is-ppo-requirement="true"' : ''}>`;

                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">
                        ${config.label}${requiredMarker}${helpIcon}
                    </label>
                    ${inputElement}
                </div>`;
                break;
        }
        return element;
    }

    function renderUI(params) {
        const uiContainer = document.getElementById('dynamic-ui');
        uiContainer.innerHTML = ''; // Clear existing content

        // Define group order and which groups are collapsible (Advanced)
        const groupOrder = [
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

        const advancedGroups = [
            'Knowledge Distillation',
            'Hyperparameter Sweep',
            'Enhanced Evaluation',
            'Reinforcement Learning (PPO)',
            'Advanced Features'
        ];

        // Group parameters by their group field
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

        // Render basic groups first
        groupOrder.forEach(groupName => {
            if (groupedParams[groupName] && !advancedGroups.includes(groupName)) {
                renderGroup(uiContainer, groupName, groupedParams[groupName], false);
            }
        });

        // Create collapsible advanced panel
        if (advancedGroups.some(g => groupedParams[g])) {
            const advancedPanel = document.createElement('div');
            advancedPanel.className = 'mb-4 border border-gray-300 dark:border-gray-600 rounded-md';

            const advancedHeader = document.createElement('button');
            advancedHeader.type = 'button';
            advancedHeader.className = 'w-full px-4 py-2 text-left font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-t-md flex justify-between items-center';
            advancedHeader.innerHTML = `
                <span>Advanced / RL / Evaluation / Sweeps</span>
                <span id="advanced-toggle-icon">▼</span>
            `;

            const advancedContent = document.createElement('div');
            advancedContent.id = 'advanced-content';
            advancedContent.className = 'hidden p-4';

            advancedHeader.addEventListener('click', () => {
                advancedContent.classList.toggle('hidden');
                const icon = document.getElementById('advanced-toggle-icon');
                icon.textContent = advancedContent.classList.contains('hidden') ? '▼' : '▲';
            });

            advancedPanel.appendChild(advancedHeader);
            advancedPanel.appendChild(advancedContent);
            uiContainer.appendChild(advancedPanel);

            // Render advanced groups inside the collapsible panel
            advancedGroups.forEach(groupName => {
                if (groupedParams[groupName]) {
                    renderGroup(advancedContent, groupName, groupedParams[groupName], true);
                }
            });
        }

        // Render ungrouped parameters at the end
        if (Object.keys(ungroupedParams).length > 0) {
            renderGroup(uiContainer, 'Other', ungroupedParams, false);
        }

        // Set up PPO validation
        setupPPOValidation();
    }

    function renderGroup(container, groupName, params, isInAdvanced) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'mb-4';

        const groupHeader = document.createElement('h3');
        groupHeader.className = 'text-md font-semibold text-gray-800 dark:text-gray-200 mb-2 pb-1 border-b border-gray-300 dark:border-gray-600';
        groupHeader.textContent = groupName;
        groupDiv.appendChild(groupHeader);

        const gridDiv = document.createElement('div');
        gridDiv.className = 'grid grid-cols-3 gap-2';

        Object.keys(params).forEach(param => {
            const config = params[param];
            const paramDiv = document.createElement('div');
            paramDiv.innerHTML = createElement(param, config);
            gridDiv.appendChild(paramDiv.firstChild);
        });

        groupDiv.appendChild(gridDiv);
        container.appendChild(groupDiv);
    }

    function setupPPOValidation() {
        // Get the task selector to determine if PPO is selected
        const taskSelector = document.getElementById('task');

        function updatePPOValidation() {
            const taskValue = taskSelector ? taskSelector.value : '';
            const isPPO = taskValue.includes(':ppo');

            // Find all PPO-specific controls and the requirement field
            const ppoControls = document.querySelectorAll('[data-required-for-ppo="true"]');
            const requirementField = document.querySelector('[data-is-ppo-requirement="true"]');

            if (isPPO) {
                // Check if the requirement field (reward model path) is filled
                const isRequirementFilled = requirementField && requirementField.value.trim() !== '';

                // Enable/disable PPO controls based on whether the requirement is met
                ppoControls.forEach(control => {
                    // Don't disable the requirement field itself
                    if (!control.hasAttribute('data-is-ppo-requirement')) {
                        if (!isRequirementFilled) {
                            control.disabled = true;
                            control.style.opacity = '0.5';
                            control.title = 'Fill in Reward Model Path first';
                        } else {
                            control.disabled = false;
                            control.style.opacity = '1';
                            control.title = '';
                        }
                    }
                });

                // Add validation on requirement field input
                if (requirementField) {
                    requirementField.removeEventListener('input', updatePPOValidation);  // Prevent duplicate listeners
                    requirementField.addEventListener('input', updatePPOValidation);
                }
            } else {
                // Not PPO, enable all controls
                ppoControls.forEach(control => {
                    control.disabled = false;
                    control.style.opacity = '1';
                    control.title = '';
                });
            }
        }

        // Run validation on task change
        if (taskSelector) {
            taskSelector.addEventListener('change', updatePPOValidation);
        }

        // Run initial validation
        updatePPOValidation();
    }


    fetchParams().then(params => {
        renderUI(params);
        observeParamChanges();
        updateTextarea();
    });

    document.getElementById('task').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckBoxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckBoxFlag = true;
            }
            renderUI(params);
            if (jsonCheckBoxFlag) {
                jsonCheckbox.checked = true;
            }
            updateTextarea();
            observeParamChanges();
        });
    });

    document.getElementById('parameter_mode').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckBoxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckBoxFlag = true;
            }
            renderUI(params);
            if (jsonCheckBoxFlag) {
                jsonCheckbox.checked = true;
            }
            updateTextarea();
            observeParamChanges();
        });
    });

    jsonCheckbox.addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });
    document.getElementById('task').addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });
    // Attach event listeners to dataset_source dropdown
    dataSource.addEventListener("change", handleDataSource);
    jsonCheckbox.addEventListener('change', switchToJSON);
    paramsTextarea.addEventListener('input', updateParamsFromTextarea);

    // Trigger the event listener to set the initial state
    handleDataSource();
    observeParamChanges();
    updateTextarea();
});