import { describe, it, expect } from '@jest/globals';
import helpers from '../../../src/autotrain/app/static/scripts/wandb_helpers.js';

describe('formatCommand', () => {
    it('returns empty string for falsy input', () => {
        expect(helpers.formatCommand(null)).toBe('');
        expect(helpers.formatCommand('')).toBe('');
    });

    it('formats absolute paths correctly', () => {
        const cmd = helpers.formatCommand('/tmp/run');
        expect(cmd).toBe('WANDB_DIR="/tmp/run" wandb beta leet "/tmp/run"');
    });
});

describe('deriveStatus', () => {
    it('combines lines into text output', () => {
        const status = helpers.deriveStatus({
            active: true,
            run_dir: '/tmp/run',
            lines: ['line1', 'line2'],
        });
        expect(status.command).toContain('/tmp/run');
        expect(status.text).toBe('line1\nline2');
        expect(status.active).toBe(true);
    });

    it('falls back to message when no lines are present', () => {
        const status = helpers.deriveStatus({
            message: 'Visualizer unavailable',
        });
        expect(status.text).toBe('Visualizer unavailable');
        expect(status.command).toBe('');
    });
});

