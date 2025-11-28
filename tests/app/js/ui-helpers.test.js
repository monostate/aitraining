/**
 * Unit tests for UI helper functions.
 * Tests the core logic of the grouped form renderer.
 */

import { describe, it, expect, jest } from '@jest/globals';
import {
    groupParameters,
    getAdvancedGroups,
    getGroupOrder,
    isPPOTask,
    shouldEnablePPOControls,
    getPPOControlState,
    createParameterHTML
} from './ui-helpers.js';

describe('groupParameters', () => {
    it('should group parameters by their group field', () => {
        const params = {
            'lr': { type: 'number', label: 'Learning Rate', group: 'Training Hyperparameters', default: 0.001 },
            'epochs': { type: 'number', label: 'Epochs', group: 'Training Hyperparameters', default: 3 },
            'peft': { type: 'checkbox', label: 'PEFT', group: 'PEFT/LoRA', default: true },
            'no_group': { type: 'string', label: 'No Group', default: '' }
        };

        const { groupedParams, ungroupedParams } = groupParameters(params);

        expect(groupedParams['Training Hyperparameters']).toBeDefined();
        expect(groupedParams['Training Hyperparameters']['lr']).toBeDefined();
        expect(groupedParams['Training Hyperparameters']['epochs']).toBeDefined();
        expect(groupedParams['PEFT/LoRA']['peft']).toBeDefined();
        expect(ungroupedParams['no_group']).toBeDefined();
    });

    it('should handle empty parameters', () => {
        const { groupedParams, ungroupedParams } = groupParameters({});

        expect(Object.keys(groupedParams).length).toBe(0);
        expect(Object.keys(ungroupedParams).length).toBe(0);
    });

    it('should not create empty groups', () => {
        const params = {
            'param1': { type: 'number', group: 'Group1', default: 1 }
        };

        const { groupedParams } = groupParameters(params);

        expect(Object.keys(groupedParams)).toEqual(['Group1']);
    });
});

describe('getAdvancedGroups', () => {
    it('should return only advanced groups that exist', () => {
        const allGroups = new Set([
            'Training Hyperparameters',
            'Knowledge Distillation',
            'Reinforcement Learning (PPO)',
            'Advanced Features'
        ]);

        const advanced = getAdvancedGroups(allGroups);

        expect(advanced).toContain('Knowledge Distillation');
        expect(advanced).toContain('Reinforcement Learning (PPO)');
        expect(advanced).toContain('Advanced Features');
        expect(advanced).not.toContain('Training Hyperparameters');
    });

    it('should return empty array if no advanced groups exist', () => {
        const allGroups = new Set(['Training Hyperparameters', 'PEFT/LoRA']);

        const advanced = getAdvancedGroups(allGroups);

        expect(advanced).toEqual([]);
    });

    it('should filter out non-existent advanced groups', () => {
        const allGroups = new Set(['Knowledge Distillation']);

        const advanced = getAdvancedGroups(allGroups);

        expect(advanced).toEqual(['Knowledge Distillation']);
        expect(advanced).not.toContain('Hyperparameter Sweep');
    });
});

describe('getGroupOrder', () => {
    it('should return a consistent group order', () => {
        const order1 = getGroupOrder();
        const order2 = getGroupOrder();

        expect(order1).toEqual(order2);
    });

    it('should include all standard groups', () => {
        const order = getGroupOrder();

        expect(order).toContain('Training Hyperparameters');
        expect(order).toContain('Training Configuration');
        expect(order).toContain('Data Processing');
        expect(order).toContain('PEFT/LoRA');
        expect(order).toContain('Reinforcement Learning (PPO)');
    });

    it('should place basic groups before advanced groups', () => {
        const order = getGroupOrder();

        const hyperparametersIndex = order.indexOf('Training Hyperparameters');
        const distillationIndex = order.indexOf('Knowledge Distillation');
        const rlIndex = order.indexOf('Reinforcement Learning (PPO)');

        expect(hyperparametersIndex).toBeLessThan(distillationIndex);
        expect(hyperparametersIndex).toBeLessThan(rlIndex);
    });
});

describe('isPPOTask', () => {
    it('should return true for PPO tasks', () => {
        expect(isPPOTask('llm:ppo')).toBe(true);
        expect(isPPOTask('custom:ppo')).toBe(true);
    });

    it('should return false for non-PPO tasks', () => {
        expect(isPPOTask('llm:sft')).toBe(false);
        expect(isPPOTask('llm:dpo')).toBe(false);
        expect(isPPOTask('llm:orpo')).toBe(false);
        expect(isPPOTask('')).toBe(false);
    });

    it('should handle null/undefined gracefully', () => {
        expect(isPPOTask(null)).toBe(false);
        expect(isPPOTask(undefined)).toBe(false);
    });
});

describe('shouldEnablePPOControls', () => {
    it('should enable controls when not PPO', () => {
        expect(shouldEnablePPOControls(false, '')).toBe(true);
        expect(shouldEnablePPOControls(false, null)).toBe(true);
    });

    it('should disable controls when PPO and requirement is empty', () => {
        expect(shouldEnablePPOControls(true, '')).toBe(false);
        expect(shouldEnablePPOControls(true, null)).toBe(false);
        expect(shouldEnablePPOControls(true, '   ')).toBe(false);
    });

    it('should enable controls when PPO and requirement is filled', () => {
        expect(shouldEnablePPOControls(true, 'reward-model-path')).toBe(true);
        expect(shouldEnablePPOControls(true, 'huggingface/model')).toBe(true);
    });
});

describe('getPPOControlState', () => {
    let mockControl, mockRequirementControl;

    beforeEach(() => {
        // Create mock DOM elements
        mockControl = {
            hasAttribute: jest.fn(() => false)
        };
        mockRequirementControl = {
            hasAttribute: jest.fn((attr) => attr === 'data-is-ppo-requirement')
        };
    });

    it('should enable all controls when not PPO', () => {
        const state = getPPOControlState(mockControl, false, false);

        expect(state.disabled).toBe(false);
        expect(state.opacity).toBe('1');
        expect(state.title).toBe('');
    });

    it('should always enable the requirement field', () => {
        const state = getPPOControlState(mockRequirementControl, false, true);

        expect(state.disabled).toBe(false);
        expect(state.opacity).toBe('1');
        expect(state.title).toBe('');
    });

    it('should disable controls when PPO and requirement not filled', () => {
        const state = getPPOControlState(mockControl, false, true);

        expect(state.disabled).toBe(true);
        expect(state.opacity).toBe('0.5');
        expect(state.title).toBe('Fill in Reward Model Path first');
    });

    it('should enable controls when PPO and requirement is filled', () => {
        const state = getPPOControlState(mockControl, true, true);

        expect(state.disabled).toBe(false);
        expect(state.opacity).toBe('1');
        expect(state.title).toBe('');
    });
});

describe('createParameterHTML', () => {
    it('should create number input HTML', () => {
        const config = {
            type: 'number',
            label: 'Learning Rate',
            default: 0.001
        };

        const html = createParameterHTML('lr', config);

        expect(html).toContain('type="number"');
        expect(html).toContain('id="param_lr"');
        expect(html).toContain('value="0.001"');
    });

    it('should create dropdown HTML with options', () => {
        const config = {
            type: 'dropdown',
            label: 'Optimizer',
            options: ['adam', 'sgd', 'adamw'],
            default: 'adam'
        };

        const html = createParameterHTML('optimizer', config);

        expect(html).toContain('<select');
        expect(html).toContain('id="param_optimizer"');
        expect(html).toContain('<option value="adam">adam</option>');
        expect(html).toContain('<option value="sgd">sgd</option>');
    });

    it('should include help icon when help text is provided', () => {
        const config = {
            type: 'string',
            label: 'Model',
            help: 'Select a base model',
            default: ''
        };

        const html = createParameterHTML('model', config);

        expect(html).toContain('help-icon');
        expect(html).toContain('Select a base model');
    });

    it('should include required marker for PPO requirement', () => {
        const config = {
            type: 'string',
            label: 'Reward Model',
            is_ppo_requirement: true,
            default: ''
        };

        const html = createParameterHTML('rl_reward_model_path', config);

        expect(html).toContain('required-marker');
        expect(html).toContain('Required for PPO trainer');
    });

    it('should create checkbox HTML', () => {
        const config = {
            type: 'checkbox',
            label: 'Enable PEFT',
            default: true
        };

        const html = createParameterHTML('peft', config);

        expect(html).toContain('type="checkbox"');
        expect(html).toContain('checked');
    });

    it('should create textarea HTML', () => {
        const config = {
            type: 'textarea',
            label: 'Custom Code',
            default: 'print("hello")'
        };

        const html = createParameterHTML('custom_code', config);

        expect(html).toContain('<textarea');
        expect(html).toContain('print("hello")');
    });

    it('should not include help icon when help is not provided', () => {
        const config = {
            type: 'string',
            label: 'Simple Param',
            default: ''
        };

        const html = createParameterHTML('simple', config);

        expect(html).not.toContain('help-icon');
    });

    it('should not include required marker for non-requirement fields', () => {
        const config = {
            type: 'string',
            label: 'Normal Field',
            required_for_ppo: true,  // Not the requirement itself
            default: ''
        };

        const html = createParameterHTML('rl_gamma', config);

        expect(html).not.toContain('required-marker');
    });
});

describe('Integration: Group Ordering and Advanced Panel', () => {
    it('should correctly categorize all groups into basic and advanced', () => {
        const allGroups = new Set(getGroupOrder());
        const advancedGroups = new Set(getAdvancedGroups(allGroups));

        const basicGroups = new Set([...allGroups].filter(g => !advancedGroups.has(g)));

        // Basic groups should come first
        expect(basicGroups.has('Training Hyperparameters')).toBe(true);
        expect(basicGroups.has('Training Configuration')).toBe(true);
        expect(basicGroups.has('PEFT/LoRA')).toBe(true);

        // Advanced groups should be separate
        expect(advancedGroups.has('Knowledge Distillation')).toBe(true);
        expect(advancedGroups.has('Reinforcement Learning (PPO)')).toBe(true);
        expect(advancedGroups.has('Advanced Features')).toBe(true);

        // No overlap
        const overlap = [...basicGroups].filter(g => advancedGroups.has(g));
        expect(overlap.length).toBe(0);
    });

    it('should maintain correct order for rendering', () => {
        const order = getGroupOrder();
        const advancedSet = new Set(getAdvancedGroups(new Set(order)));

        // Find index of first advanced group
        const firstAdvancedIndex = order.findIndex(g => advancedSet.has(g));

        // All basic groups should come before first advanced group
        for (let i = 0; i < firstAdvancedIndex; i++) {
            expect(advancedSet.has(order[i])).toBe(false);
        }

        // All advanced groups should come after last basic group
        for (let i = firstAdvancedIndex; i < order.length; i++) {
            if (advancedSet.has(order[i])) {
                expect(i).toBeGreaterThanOrEqual(firstAdvancedIndex);
            }
        }
    });
});
