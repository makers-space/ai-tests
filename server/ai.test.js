import fs from 'node:fs';
import { describe, test, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import TestStartup from '../utils/test.startup.js';
import { ROLES, RIGHTS, hasRight } from '../../server/config/rights.js';
import {
    setOllamaClientFactory, resetOllamaClientFactory, createOllamaClient,
    mapOllamaError, OllamaError, ERROR_CODES,
    parseNvidiaSmi, detectHardware,
    fitModel, rankCatalogue, memoryNeededMb, VERDICTS
} from '../../server/middleware/ai.middleware.js';
import { Device, Setting, Download, PersonalModel, deviceSchemas, modelSchemas } from '../../server/models/ai.model.js';
import { downloads, health } from '../../server/controllers/ai.controller.js';

const MB = 1024 * 1024;

let testStartup;
let client;

beforeAll(async () => {
    testStartup = new TestStartup('ai');
    await testStartup.initialize();
    client = testStartup.getClient();
}, 60000);

afterAll(async () => {
    resetOllamaClientFactory();
    await testStartup.cleanup();
}, 30000);

describe('Model rights', () => {
    test('MANAGE_MODELS and USE_MODELS exist', () => {
        expect(RIGHTS.MANAGE_MODELS).toBe('MANAGE_MODELS');
        expect(RIGHTS.USE_MODELS).toBe('USE_MODELS');
    });

    test.each([ROLES.OWNER, ROLES.ADMIN])('%s can manage models', (role) => {
        expect(hasRight([role], RIGHTS.MANAGE_MODELS)).toBe(true);
    });

    test.each([ROLES.SUPER_CREATOR, ROLES.CREATOR, ROLES.USER])('%s cannot manage models', (role) => {
        expect(hasRight([role], RIGHTS.MANAGE_MODELS)).toBe(false);
    });

    test.each(Object.values(ROLES))('%s can use models', (role) => {
        expect(hasRight([role], RIGHTS.USE_MODELS)).toBe(true);
    });

    test('a user with several roles gets the highest right', () => {
        expect(hasRight([ROLES.USER, ROLES.ADMIN], RIGHTS.MANAGE_MODELS)).toBe(true);
    });

    test('an unknown or missing role has neither right', () => {
        expect(hasRight(['GUEST'], RIGHTS.USE_MODELS)).toBe(false);
        expect(hasRight([], RIGHTS.MANAGE_MODELS)).toBe(false);
    });

    test('existing rights are unchanged for USER', () => {
        expect(hasRight([ROLES.USER], RIGHTS.MANAGE_OWN_ACCOUNT)).toBe(true);
        expect(hasRight([ROLES.USER], RIGHTS.MANAGE_ALL_USERS)).toBe(false);
        expect(hasRight([ROLES.USER], RIGHTS.CREATE_CONTENT)).toBe(false);
    });
});

const connectionRefusedError = () => Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED' } });

const simpleOllama = ({ up = true, models = ['small:1b', 'medium:7b'] } = {}) => () => ({
    version: async () => {
        if (!up) throw connectionRefusedError();
        return { version: '9.9.9-test' };
    },
    list: async () => {
        if (!up) throw connectionRefusedError();
        return { models: models.map((name) => ({ name })) };
    }
});

describe('Ollama error mapping', () => {
    test.each([
        [connectionRefusedError(), 'OLLAMA_DOWN', 503],
        [{ error: "model 'x:1' not found", status_code: 404 }, 'MODEL_MISSING', 404],
        [{ error: 'something else', status_code: 404 }, 'MODEL_MISSING', 404],
        [{ error: 'model requires more system memory (9.1 GiB) than is available (4.0 GiB)' }, 'OUT_OF_MEMORY', 422],
        [{ error: 'write /models/blob: no space left on device' }, 'DISK_FULL', 422],
        [{ error: 'registry.ollama.ai/library/tiny does not support tools' }, 'NO_TOOL_SUPPORT', 422],
        [{ error: 'input exceeds the maximum context length' }, 'TOO_LONG', 422],
        [{ error: 'pull failed: unexpected EOF' }, 'DOWNLOAD_INTERRUPTED', 502],
        [{ error: 'boom' }, 'UNKNOWN', 502],
        [undefined, 'UNKNOWN', 502]
    ])('maps %o to %s', (input, code, status) => {
        const mapped = mapOllamaError(input);
        expect(mapped).toBeInstanceOf(OllamaError);
        expect(mapped.code).toBe(code);
        expect(mapped.status).toBe(status);
        expect(mapped.message.length).toBeGreaterThan(10);
        expect(mapped.action.length).toBeGreaterThan(5);
    });

    test('every code has a plain message that hides the raw error', () => {
        for (const code of ERROR_CODES) {
            const error = new OllamaError(code, 'raw internal text');
            expect(error.toJSON()).toEqual({ code, message: error.message, action: error.action });
            expect(JSON.stringify(error.toJSON())).not.toContain('raw internal text');
        }
    });

    test('an already mapped error is returned unchanged', () => {
        const original = new OllamaError('DISK_FULL');
        expect(mapOllamaError(original)).toBe(original);
    });
});

describe('Ollama client', () => {
    afterAll(() => resetOllamaClientFactory());

    test('status reports the version when Ollama answers', async () => {
        setOllamaClientFactory(simpleOllama());
        expect(await createOllamaClient('http://fake').status()).toEqual({ available: true, version: '9.9.9-test' });
    });

    test('status reports unavailable instead of throwing', async () => {
        setOllamaClientFactory(simpleOllama({ up: false }));
        expect(await createOllamaClient('http://fake').status()).toEqual({ available: false, version: null });
    });

    test('other calls throw a mapped error', async () => {
        setOllamaClientFactory(simpleOllama({ up: false }));
        await expect(createOllamaClient('http://fake').list()).rejects.toMatchObject({ code: 'OLLAMA_DOWN' });
    });
});

describe('Hardware reading', () => {
    test('parses one graphics card', () => {
        expect(parseNvidiaSmi('NVIDIA GeForce RTX 4060 Ti, 8188\r\n')).toEqual([
            { name: 'NVIDIA GeForce RTX 4060 Ti', memoryMb: 8188 }
        ]);
    });

    test('parses several graphics cards', () => {
        const cards = parseNvidiaSmi('NVIDIA RTX A6000, 49140\nNVIDIA GeForce RTX 3090, 24576\n');
        expect(cards).toHaveLength(2);
        expect(cards[1]).toEqual({ name: 'NVIDIA GeForce RTX 3090', memoryMb: 24576 });
    });

    test.each(['', null, undefined, 'not a csv line', 'Card, N/A', ', 8000'])('ignores unusable output %o', (text) => {
        expect(parseNvidiaSmi(text)).toEqual([]);
    });

    test('reads this machine without failing', async () => {
        const hardware = await detectHardware();
        expect(hardware.memoryMb).toBeGreaterThan(256);
        expect(hardware.cpuThreads).toBeGreaterThan(0);
        expect(typeof hardware.hostname).toBe('string');
        expect(Array.isArray(hardware.gpus)).toBe(true);
        expect(hardware.gpuMemoryMb).toBeGreaterThanOrEqual(0);
        expect(hardware.detectedAt).toBeInstanceOf(Date);
    });
});

describe('Device and settings validation', () => {
    const valid = (schema, value) => schema.validate(value).error === undefined;

    test('settings accept sensible values', () => {
        expect(valid(deviceSchemas.updateSettings, { ollamaBaseUrl: 'http://127.0.0.1:11434' })).toBe(true);
        expect(valid(deviceSchemas.updateSettings, { ollamaBaseUrl: '' })).toBe(true);
        expect(valid(deviceSchemas.updateSettings, { defaultModel: 'hf.co/unsloth/x:Q4_K_M' })).toBe(true);
        expect(valid(deviceSchemas.updateSettings, { maxConcurrentGenerations: 8, keepAliveMinutes: 0 })).toBe(true);
    });

    test.each([
        [{}],
        [{ ollamaBaseUrl: 'ftp://host' }],
        [{ ollamaBaseUrl: 'not a url' }],
        [{ defaultModel: 'bad name!' }],
        [{ maxConcurrentGenerations: 0 }],
        [{ maxConcurrentGenerations: 9 }],
        [{ keepAliveMinutes: -1 }],
        [{ unknownField: true }]
    ])('settings reject %o', (value) => {
        expect(valid(deviceSchemas.updateSettings, value)).toBe(false);
    });

    test.each([
        [{}],
        [{ name: '' }],
        [{ overrides: { memoryMb: 10 } }],
        [{ overrides: { gpuMemoryMb: -1 } }],
        [{ overrides: { somethingElse: 1 } }]
    ])('device edits reject %o', (value) => {
        expect(valid(deviceSchemas.updateDevice, value)).toBe(false);
    });
});

describe('Device, settings and setup endpoints', () => {
    let deviceId;

    beforeEach(() => {
        setOllamaClientFactory(simpleOllama());
    });

    describe('GET /api/v1/devices', () => {
        test('refuses a visitor who is not signed in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            const response = await client.get('/api/v1/devices');
            expect(response.status).toBe(401);
        });

        test('shows this computer to a regular user, without manage rights', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');

            expect(response.status).toBe(200);
            expect(response.data.devices).toHaveLength(1);
            expect(response.data.permissions.manage).toBe(false);

            const [device] = response.data.devices;
            deviceId = device.id;
            expect(device.kind).toBe('local');
            expect(device.service).toBe('ai-base');
            expect(device.online).toBe(true);
            expect(device.hardware.memoryMb).toBeGreaterThan(256);
            expect(typeof device.hardware.hasGpu).toBe('boolean');
            expect(device.edited).toBe(false);
            expect(device.ollama).toEqual({ available: true, version: '9.9.9-test', modelCount: 2 });
        });

        test('gives manage rights to admin and owner', async () => {
            for (const role of ['admin', 'owner']) {
                await testStartup.loginAsUser(role);
                const response = await client.get('/api/v1/devices');
                expect(response.status).toBe(200);
                expect(response.data.permissions.manage).toBe(true);
            }
        });

        test('still answers when Ollama is not running', async () => {
            setOllamaClientFactory(simpleOllama({ up: false }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');

            expect(response.status).toBe(200);
            expect(response.data.devices[0].ollama).toEqual({ available: false, version: null, modelCount: 0 });
        });

        test('is never cached', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');
            expect(String(response.headers['cache-control'] || '')).toMatch(/no-store|no-cache/);
        });

        test('stores the device once, marked with the service', async () => {
            await client.get('/api/v1/devices');
            const stored = await Device.find({ kind: 'local' });
            expect(stored).toHaveLength(1);
            expect(stored[0].service).toBe('ai-base');
            expect(stored[0].hardware.memoryMb).toBeGreaterThan(256);
        });
    });

    describe('GET /api/v1/devices/:id', () => {
        test('returns the device', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get(`/api/v1/devices/${deviceId}`);
            expect(response.status).toBe(200);
            expect(response.data.device.id).toBe(deviceId);
        });

        test('rejects an id that is not a database id', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices/not-an-id');
            expect(response.status).toBe(400);
        });

        test('answers 404 for an unknown device', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices/64b7f0c2a1b2c3d4e5f60718');
            expect(response.status).toBe(404);
        });
    });

    describe('PUT /api/v1/devices/:id', () => {
        test('refuses a regular user', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.put(`/api/v1/devices/${deviceId}`, { name: 'Renamed' });
            expect(response.status).toBe(403);
        });

        test('lets an admin rename the device and correct its hardware', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, {
                name: 'Studio PC',
                overrides: { memoryMb: 65536, gpuName: 'Test Card', gpuMemoryMb: 24576 }
            });

            expect(response.status).toBe(200);
            expect(response.data.device.name).toBe('Studio PC');
            expect(response.data.device.edited).toBe(true);
            expect(response.data.device.hardware).toMatchObject({
                memoryMb: 65536, gpuName: 'Test Card', gpuMemoryMb: 24576, hasGpu: true
            });
        });

        test('keeps the correction after a new detection', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.post(`/api/v1/devices/${deviceId}/detect`, {});
            expect(response.status).toBe(200);
            expect(response.data.device.hardware.gpuMemoryMb).toBe(24576);
            expect(response.data.device.detected.memoryMb).not.toBe(65536);
        });

        test('goes back to detected values when the correction is cleared', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, {
                overrides: { memoryMb: null, gpuName: '', gpuMemoryMb: null }
            });

            expect(response.status).toBe(200);
            expect(response.data.device.edited).toBe(false);
            expect(response.data.device.hardware.memoryMb).toBe(response.data.device.detected.memoryMb);
        });

        test('rejects values outside the allowed range', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, { overrides: { memoryMb: 10 } });
            expect(response.status).toBe(400);
        });

        test('rejects an empty change', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, {});
            expect(response.status).toBe(400);
        });
    });

    describe('POST /api/v1/devices/:id/detect', () => {
        test('refuses a regular user', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.post(`/api/v1/devices/${deviceId}/detect`, {});
            expect(response.status).toBe(403);
        });
    });

    describe('Settings', () => {
        test('a regular user can read settings but not change them', async () => {
            await testStartup.loginAsUser('user');
            const read = await client.get('/api/v1/settings');
            expect(read.status).toBe(200);
            expect(read.data.permissions.manage).toBe(false);
            expect(read.data.settings.ollamaBaseUrl).toBe(process.env.OLLAMA_BASE_URL);
            expect(read.data.settings.ollamaBaseUrlIsDefault).toBe(true);
            expect(read.data.settings.maxConcurrentGenerations).toBe(1);

            const write = await client.put('/api/v1/settings', { keepAliveMinutes: 10 });
            expect(write.status).toBe(403);
        });

        test('an admin can change settings and they take effect at once', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put('/api/v1/settings', {
                ollamaBaseUrl: 'http://192.168.50.10:11434',
                defaultModel: 'medium:7b',
                maxConcurrentGenerations: 2,
                keepAliveMinutes: 30
            });

            expect(response.status).toBe(200);
            expect(response.data.settings).toMatchObject({
                ollamaBaseUrl: 'http://192.168.50.10:11434',
                ollamaBaseUrlIsDefault: false,
                defaultModel: 'medium:7b',
                maxConcurrentGenerations: 2,
                keepAliveMinutes: 30
            });

            const read = await client.get('/api/v1/settings');
            expect(read.data.settings.ollamaBaseUrl).toBe('http://192.168.50.10:11434');
        });

        test('clearing the address goes back to the default', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put('/api/v1/settings', { ollamaBaseUrl: '' });
            expect(response.status).toBe(200);
            expect(response.data.settings.ollamaBaseUrl).toBe(process.env.OLLAMA_BASE_URL);
            expect(response.data.settings.ollamaBaseUrlIsDefault).toBe(true);
        });

        test('rejects a bad address and out-of-range numbers', async () => {
            await testStartup.loginAsUser('admin');
            expect((await client.put('/api/v1/settings', { ollamaBaseUrl: 'ftp://host' })).status).toBe(400);
            expect((await client.put('/api/v1/settings', { maxConcurrentGenerations: 0 })).status).toBe(400);
            expect((await client.put('/api/v1/settings', {})).status).toBe(400);
        });

        test('keeps one settings record, marked with the service', async () => {
            const stored = await Setting.find({});
            expect(stored).toHaveLength(1);
            expect(stored[0].service).toBe('ai-base');
        });
    });

    describe('POST /api/v1/settings/test-connection', () => {
        test('refuses a regular user', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://127.0.0.1:11434' });
            expect(response.status).toBe(403);
        });

        test('reports success with the version', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://127.0.0.1:11434' });
            expect(response.status).toBe(200);
            expect(response.data.ollama).toEqual({ available: true, version: '9.9.9-test' });
        });

        test('reports a plain error when nothing answers', async () => {
            setOllamaClientFactory(simpleOllama({ up: false }));
            await testStartup.loginAsUser('admin');
            const response = await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://127.0.0.1:11434' });
            expect(response.status).toBe(503);
            expect(response.data).toMatchObject({ success: false, code: 'OLLAMA_DOWN' });
            expect(response.data.action).toBeDefined();
        });

        test('does not save the address it tested', async () => {
            await testStartup.loginAsUser('admin');
            await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://10.9.8.7:11434' });
            const read = await client.get('/api/v1/settings');
            expect(read.data.settings.ollamaBaseUrl).not.toBe('http://10.9.8.7:11434');
        });

        test('rejects a missing or malformed address', async () => {
            await testStartup.loginAsUser('admin');
            expect((await client.post('/api/v1/settings/test-connection', {})).status).toBe(400);
            expect((await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'nope' })).status).toBe(400);
        });
    });

    describe('GET /api/v1/setup/status', () => {
        test('refuses a visitor who is not signed in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            const response = await client.get('/api/v1/setup/status');
            expect(response.status).toBe(401);
        });

        test('reports every step as ready when Ollama has models', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/setup/status');
            expect(response.status).toBe(200);
            expect(response.data.steps).toEqual({ ollama: true, device: true, model: true });
            expect(response.data.ollama.version).toBe('9.9.9-test');
        });

        test('reports no model yet when Ollama is empty', async () => {
            setOllamaClientFactory(simpleOllama({ models: [] }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/setup/status');
            expect(response.data.steps).toEqual({ ollama: true, device: true, model: false });
        });

        test('explains in plain words when Ollama is not running', async () => {
            setOllamaClientFactory(simpleOllama({ up: false }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/setup/status');

            expect(response.status).toBe(200);
            expect(response.data.steps.ollama).toBe(false);
            expect(response.data.ollama).toMatchObject({ available: false, code: 'OLLAMA_DOWN' });
            expect(response.data.ollama.message).toMatch(/not running/i);
            expect(response.data.ollama.action).toBeDefined();
        });
    });
});

const catalogue = JSON.parse(fs.readFileSync(new URL('../../server/config/ai.catalogue.json', import.meta.url), 'utf8'));
const entry = (tag) => catalogue.find((model) => model.tag === tag);

const notFound = (name) => Object.assign(new Error(`model '${name}' not found`), { error: `model '${name}' not found`, status_code: 404 });
const refused = () => Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED' } });

const installed = (name, sizeMb = 100) => ({
    name,
    size: sizeMb * MB,
    modified_at: '2026-11-01T10:00:00Z',
    details: { family: 'testfam', parameter_size: '7.6B', quantization_level: 'Q4_K_M' }
});

const state = {};
const resetState = () => {
    Object.assign(state, {
        up: true,
        models: [installed('qwen2.5-coder:7b', 4700), installed('custom-thing:latest', 50)],
        running: [],
        unloaded: [],
        created: [],
        pullSizeMb: 10,
        pullFailsMidway: false,
        hold: null,
        stall: false,
        silent: false
    });
};

const fakePull = (model) => {
    let aborted = false;
    let wake = () => {};
    const total = state.pullSizeMb * MB;
    const check = () => {
        if (aborted) throw Object.assign(new Error('The operation was aborted'), { name: 'AbortError' });
    };
    const pause = (ms) => new Promise((resolve) => {
        wake = resolve;
        setTimeout(resolve, ms);
    });

    return {
        abort() {
            aborted = true;
            wake();
        },
        async *[Symbol.asyncIterator]() {
            yield { status: 'pulling manifest' };
            while (state.silent) {
                await pause(20);
                check();
            }
            await pause(5);
            check();
            yield { status: 'pulling layer', digest: 'sha256:one', total, completed: total / 2 };
            if (state.pullFailsMidway) throw new Error('unexpected EOF');
            while (state.stall) {
                await pause(20);
                check();
                yield { status: 'pulling layer', digest: 'sha256:one', total, completed: total / 2 };
            }
            while (state.hold) {
                await pause(20);
                check();
            }
            await pause(5);
            check();
            yield { status: 'pulling layer', digest: 'sha256:one', total, completed: total };
            yield { status: 'success' };
            if (!state.models.some((existing) => existing.name === model)) {
                state.models.push(installed(model, state.pullSizeMb));
            }
        }
    };
};

const fakeOllama = () => ({
    version: async () => {
        if (!state.up) throw refused();
        return { version: '9.9.9-test' };
    },
    list: async () => {
        if (!state.up) throw refused();
        return { models: state.models };
    },
    ps: async () => {
        if (!state.up) throw refused();
        return { models: state.running };
    },
    show: async ({ model }) => {
        if (!state.models.some((existing) => existing.name === model)) throw notFound(model);
        return {
            modelfile: `FROM ${model}`,
            parameters: 'temperature 0.7',
            template: '{{ .Prompt }}',
            capabilities: ['completion', 'tools'],
            model_info: { 'testfam.context_length': 32768 },
            details: { family: 'testfam' }
        };
    },
    delete: async ({ model }) => {
        const before = state.models.length;
        state.models = state.models.filter((existing) => existing.name !== model);
        if (state.models.length === before) throw notFound(model);
        return { status: 'success' };
    },
    create: async (request) => {
        state.created.push(request);
        state.models.push(installed(`${request.model}:latest`, 1));
        return { status: 'success' };
    },
    generate: async ({ model, keep_alive }) => {
        if (!state.models.some((existing) => existing.name === model)) throw notFound(model);
        state.unloaded.push({ model, keep_alive });
        return { done: true };
    },
    pull: async ({ model }) => {
        if (!state.up) throw refused();
        if (model.startsWith('nosuch')) {
            throw Object.assign(new Error('pull model manifest: file does not exist'), { error: 'pull model manifest: file does not exist' });
        }
        return fakePull(model);
    }
});

describe('Fit rules', () => {
    const laptop = { memoryMb: 32768, gpuMemoryMb: 8192 };
    const workstation = { memoryMb: 65536, gpuMemoryMb: 24576 };
    const noCard = { memoryMb: 16384, gpuMemoryMb: 0 };

    test('8 GB of graphics memory runs 7 to 9B models fully on the GPU', () => {
        for (const tag of ['mistral:7b', 'qwen2.5-coder:7b', 'llama3.1:8b']) {
            expect(fitModel(entry(tag), laptop).verdict).toBe('gpu');
        }
    });

    test('8 GB plus 32 GB of system memory runs a 30B mixture-of-experts model with CPU help', () => {
        const fit = fitModel(entry('qwen3-coder:30b'), laptop);
        expect(fit.verdict).toBe('mixed');
        expect(fit.label).toBe(VERDICTS.mixed);
        expect(fit.tokensPerSecond[0]).toBeGreaterThanOrEqual(15);
    });

    test('a 70B model is too large for that machine', () => {
        const fit = fitModel(entry('llama3.3:70b'), laptop);
        expect(fit.verdict).toBe('too_large');
        expect(fit.fits).toBe(false);
        expect(fit.tokensPerSecond).toBeNull();
    });

    test('24 GB of graphics memory runs a 32B model fully on the GPU', () => {
        expect(fitModel(entry('qwen2.5-coder:32b'), workstation).verdict).toBe('gpu');
    });

    test('a machine without a graphics card runs small models on the processor', () => {
        expect(fitModel(entry('llama3.2:3b'), noCard).verdict).toBe('cpu');
        expect(fitModel(entry('qwen3:30b'), noCard).verdict).toBe('too_large');
    });

    test('headroom is kept for desktop apps', () => {
        const model = { sizeMb: 5000, parameters: 8, moe: false, goodFor: ['chat'] };
        const needed = memoryNeededMb(model);
        expect(fitModel(model, { memoryMb: 0, gpuMemoryMb: needed + 1536 }).verdict).toBe('gpu');
        expect(fitModel(model, { memoryMb: 0, gpuMemoryMb: needed + 1535 }).verdict).toBe('too_large');
    });

    test('missing or odd hardware never throws', () => {
        for (const hardware of [undefined, null, {}, { memoryMb: 'abc', gpuMemoryMb: -5 }]) {
            expect(fitModel(entry('llama3.2:3b'), hardware).verdict).toBe('too_large');
        }
    });

    test('three different models are recommended first', () => {
        const ranked = rankCatalogue(catalogue, laptop);
        expect(ranked.slice(0, 3).map((model) => model.pick)).toEqual(['fast', 'balanced', 'best']);
        expect(new Set(ranked.slice(0, 3).map((model) => model.tag)).size).toBe(3);
        expect(ranked.slice(3).every((model) => model.pick === null)).toBe(true);
        expect(ranked.find((model) => model.pick === 'best').fit.fits).toBe(true);
    });

    test('a very small machine gets no recommendation instead of a wrong one', () => {
        const ranked = rankCatalogue(catalogue, { memoryMb: 4096, gpuMemoryMb: 0 });
        expect(ranked.some((model) => model.pick)).toBe(false);
        expect(ranked).toHaveLength(catalogue.length);
    });

    test('the catalogue covers every size tier with complete entries', () => {
        expect(catalogue.length).toBeGreaterThanOrEqual(12);
        const sizes = catalogue.map((model) => model.parameters);
        expect(sizes.some((size) => size <= 4.5)).toBe(true);
        expect(sizes.some((size) => size > 4.5 && size <= 9)).toBe(true);
        expect(sizes.some((size) => size > 9 && size <= 15)).toBe(true);
        expect(catalogue.some((model) => model.moe && model.parameters >= 30)).toBe(true);
        expect(sizes.some((size) => size >= 70)).toBe(true);
        for (const model of catalogue) {
            expect(modelSchemas.named.validate({ name: model.tag }).error).toBeUndefined();
            expect(model.name && model.description).toBeTruthy();
            expect(model.goodFor.length).toBeGreaterThan(0);
            expect(model.sizeMb).toBeGreaterThan(0);
            expect(Array.isArray(model.capabilities)).toBe(true);
        }
        expect(new Set(catalogue.map((model) => model.tag)).size).toBe(catalogue.length);
    });
});

describe('Model validation', () => {
    const valid = (schema, value) => schema.validate(value).error === undefined;

    test.each(['llama3.1:8b', 'hf.co/unsloth/x:Q4_K_M', 'a'])('accepts the model name %s', (name) => {
        expect(valid(modelSchemas.named, { name })).toBe(true);
    });

    test.each(['', 'bad name', 'semi;colon', 'x'.repeat(201), 'quote"'])('rejects the model name %o', (name) => {
        expect(valid(modelSchemas.named, { name })).toBe(false);
    });

    test('personal models need a simple name, a display name and a base', () => {
        expect(valid(modelSchemas.create, { name: 'careful-reviewer', displayName: 'Careful reviewer', from: 'llama3.1:8b' })).toBe(true);
        expect(valid(modelSchemas.create, { name: 'Careful', displayName: 'x', from: 'llama3.1:8b' })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'has/slash', displayName: 'x', from: 'llama3.1:8b' })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x' })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x', from: 'a', system: 'y'.repeat(20001) })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x', from: 'a', parameters: { temperature: 0.2, stop: 'END', raw: true } })).toBe(true);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x', from: 'a', parameters: { nested: { no: 1 } } })).toBe(false);
    });
});

describe('Model endpoints', () => {
    const post = (path, body) => client.post(`/api/v1${path}`, body);
    const get = (path) => client.get(`/api/v1${path}`);
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };

    const BIG_DEVICE = { memoryMb: 262144, gpuMemoryMb: 98304, gpuName: 'Test card' };
    const setHardware = async (overrides) => {
        await testStartup.loginAsUser('admin');
        const { data } = await get('/devices');
        await client.put(`/api/v1/devices/${data.devices[0].id}`, { overrides });
    };

    beforeAll(async () => {
        resetState();
        setOllamaClientFactory(fakeOllama);
        await setHardware(BIG_DEVICE);
    });

    afterAll(async () => {
        state.hold = null;
        state.stall = false;
        state.silent = false;
        await downloads.idle();
        downloads.setDiskProbe(null);
        await setHardware({ memoryMb: null, gpuMemoryMb: null, gpuName: '' });
    });

    beforeEach(async () => {
        state.hold = null;
        state.stall = false;
        state.silent = false;
        await downloads.idle();
        resetState();
        Object.assign(downloads, { stallWarnMs: 20000, stallFailMs: 180000, retryMs: 5000, saveWaitMs: 3000 });
        downloads.setDiskProbe(async () => ({ diskFreeMb: 500000, diskTotalMb: 1000000 }));
        await Download.deleteMany({});
        await PersonalModel.deleteMany({});
    });

    describe('GET /models', () => {
        test('refuses a visitor who is not signed in', async () => {
            signOut();
            expect((await get('/models')).status).toBe(401);
        });

        test('lists installed models with plain details', async () => {
            await testStartup.loginAsUser('user');
            const response = await get('/models');

            expect(response.status).toBe(200);
            expect(response.data.ollamaAvailable).toBe(true);
            expect(response.data.ollamaVersion).toBe('9.9.9-test');
            expect(response.data.permissions.manage).toBe(false);
            expect(response.data.models).toHaveLength(2);

            const known = response.data.models.find((model) => model.name === 'qwen2.5-coder:7b');
            expect(known).toMatchObject({
                displayName: 'Qwen 2.5 Coder 7B',
                family: 'testfam',
                parameterSize: '7.6B',
                quantization: 'Q4_K_M',
                contextLength: 32768,
                personal: false,
                usedBy: []
            });
            expect(known.capabilities).toContain('tools');
            expect(known.sizeBytes).toBe(4700 * MB);

            const unknown = response.data.models.find((model) => model.name === 'custom-thing:latest');
            expect(unknown.displayName).toBe('custom-thing:latest');
            expect(response.data.disk).toEqual(expect.objectContaining({ usedByModelsMb: 4750 }));
        });

        test('answers with an empty list when Ollama is not running', async () => {
            state.up = false;
            await testStartup.loginAsUser('user');
            const response = await get('/models');
            expect(response.status).toBe(200);
            expect(response.data).toMatchObject({ ollamaAvailable: false, models: [] });
        });
    });

    describe('GET /models/running', () => {
        test('reports how much of each loaded model is on the graphics card', async () => {
            state.running = [
                { name: 'qwen2.5-coder:7b', size: 6000 * MB, size_vram: 6000 * MB },
                { name: 'big:30b', size: 20000 * MB, size_vram: 5000 * MB },
                { name: 'cpuonly:1b', size: 1000 * MB, size_vram: 0 }
            ];
            await testStartup.loginAsUser('user');
            const response = await get('/models/running');

            expect(response.status).toBe(200);
            expect(response.data.models.map((model) => [model.gpuPercent, model.healthy])).toEqual([
                [100, true], [25, false], [0, false]
            ]);
        });

        test('explains plainly when Ollama is not running', async () => {
            state.up = false;
            await testStartup.loginAsUser('user');
            const response = await get('/models/running');
            expect(response.status).toBe(503);
            expect(response.data.code).toBe('OLLAMA_DOWN');
        });
    });

    describe('POST /models/show', () => {
        test('returns the definition of an installed model', async () => {
            await testStartup.loginAsUser('user');
            const response = await post('/models/show', { name: 'qwen2.5-coder:7b' });
            expect(response.status).toBe(200);
            expect(response.data.model).toMatchObject({ definition: 'FROM qwen2.5-coder:7b', contextLength: 32768 });
        });

        test('answers 404 in plain words for a model that is not installed', async () => {
            await testStartup.loginAsUser('user');
            const response = await post('/models/show', { name: 'nope:1b' });
            expect(response.status).toBe(404);
            expect(response.data).toMatchObject({ code: 'MODEL_MISSING' });
        });

        test('rejects a malformed name', async () => {
            await testStartup.loginAsUser('user');
            expect((await post('/models/show', { name: 'bad name!' })).status).toBe(400);
            expect((await post('/models/show', {})).status).toBe(400);
        });
    });

    describe('Changing the device needs manage rights', () => {
        test.each([
            ['/models/delete', { name: 'qwen2.5-coder:7b' }],
            ['/models/unload', { name: 'qwen2.5-coder:7b' }],
            ['/models/pull', { name: 'llama3.2:3b' }],
            ['/models/create', { name: 'mine', displayName: 'Mine', from: 'qwen2.5-coder:7b' }],
            ['/models/downloads/cancel', { id: '64b7f0c2a1b2c3d4e5f60718', action: 'cancel' }]
        ])('a regular user is refused on %s', async (path, body) => {
            await testStartup.loginAsUser('user');
            expect((await post(path, body)).status).toBe(403);
            expect(state.models).toHaveLength(2);
        });

        test.each(['creator', 'superCreator'])('a %s is refused too', async (role) => {
            await testStartup.loginAsUser(role);
            expect((await post('/models/delete', { name: 'qwen2.5-coder:7b' })).status).toBe(403);
        });

        test('a visitor who is not signed in is refused', async () => {
            signOut();
            expect((await post('/models/delete', { name: 'qwen2.5-coder:7b' })).status).toBe(401);
        });
    });

    describe('POST /models/delete and /models/unload', () => {
        test('an admin can delete a model', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/delete', { name: 'custom-thing:latest' });
            expect(response.status).toBe(200);
            expect(state.models.map((model) => model.name)).toEqual(['qwen2.5-coder:7b']);
        });

        test('deleting a model that is not there answers 404', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/delete', { name: 'ghost:1b' });
            expect(response.status).toBe(404);
            expect(response.data.code).toBe('MODEL_MISSING');
        });

        test('an owner can unload a model from memory', async () => {
            await testStartup.loginAsUser('owner');
            const response = await post('/models/unload', { name: 'qwen2.5-coder:7b' });
            expect(response.status).toBe(200);
            expect(state.unloaded).toEqual([{ model: 'qwen2.5-coder:7b', keep_alive: 0 }]);
        });
    });

    describe('POST /models/create', () => {
        const careful = {
            name: 'careful-reviewer',
            displayName: 'Careful reviewer',
            from: 'qwen2.5-coder:7b',
            system: 'Check the work for mistakes.',
            parameters: { temperature: 0.2 }
        };

        test('an admin creates a personal model that shows in the library', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/create', careful);

            expect(response.status).toBe(201);
            expect(state.created[0]).toMatchObject({
                model: 'careful-reviewer', from: 'qwen2.5-coder:7b', system: 'Check the work for mistakes.',
                parameters: { temperature: 0.2 }, stream: false
            });

            const stored = await PersonalModel.findOne({ name: 'careful-reviewer' });
            expect(stored.service).toBe('ai-base');

            const list = await get('/models');
            const mine = list.data.models.find((model) => model.name === 'careful-reviewer:latest');
            expect(mine).toMatchObject({ personal: true, displayName: 'Careful reviewer', basedOn: 'qwen2.5-coder:7b' });
            const base = list.data.models.find((model) => model.name === 'qwen2.5-coder:7b');
            expect(base.usedBy).toEqual(['Careful reviewer']);
        });

        test('refuses a name that is already taken', async () => {
            await testStartup.loginAsUser('admin');
            await post('/models/create', careful);
            const again = await post('/models/create', careful);
            expect(again.status).toBe(409);
        });

        test('refuses a base model that is not installed', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/create', { ...careful, from: 'missing:7b' });
            expect(response.status).toBe(404);
            expect(response.data.code).toBe('MODEL_MISSING');
            expect(state.created).toHaveLength(0);
        });

        test('deleting a personal model removes its record', async () => {
            await testStartup.loginAsUser('admin');
            await post('/models/create', careful);
            await post('/models/delete', { name: 'careful-reviewer:latest' });
            expect(await PersonalModel.countDocuments({})).toBe(0);
        });

        test('rejects an invalid request', async () => {
            await testStartup.loginAsUser('admin');
            expect((await post('/models/create', { ...careful, name: 'Has Spaces' })).status).toBe(400);
            expect((await post('/models/create', { ...careful, system: 'x'.repeat(20001) })).status).toBe(400);
        });
    });

    describe('Catalogue', () => {
        test('is public and lists every model', async () => {
            signOut();
            const response = await get('/catalogue');
            expect(response.status).toBe(200);
            expect(response.data.models).toHaveLength(catalogue.length);
            expect(response.data.models[0].fit).toBeNull();
        });

        test('gives verdicts for the specs a visitor enters', async () => {
            signOut();
            const response = await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192');
            expect(response.status).toBe(200);
            const verdict = (tag) => response.data.models.find((model) => model.tag === tag).fit.verdict;
            expect(verdict('llama3.1:8b')).toBe('gpu');
            expect(verdict('qwen3-coder:30b')).toBe('mixed');
            expect(verdict('llama3.3:70b')).toBe('too_large');
            expect(response.data.models.slice(0, 3).map((model) => model.pick)).toEqual(['fast', 'balanced', 'best']);
        });

        test('rejects nonsense specs', async () => {
            signOut();
            expect((await get('/catalogue?memoryMb=-5')).status).toBe(400);
            expect((await get('/catalogue?memoryMb=abc')).status).toBe(400);
            expect((await get('/catalogue?somethingElse=1')).status).toBe(400);
        });

        test('the device catalogue needs sign-in and agrees with the public verdicts', async () => {
            signOut();
            const devicesAnon = await get('/devices/64b7f0c2a1b2c3d4e5f60718/catalogue');
            expect(devicesAnon.status).toBe(401);

            await testStartup.loginAsUser('admin');
            const { data: deviceList } = await get('/devices');
            const device = deviceList.devices[0];
            await client.put(`/api/v1/devices/${device.id}`, { overrides: { memoryMb: 32768, gpuMemoryMb: 8192, gpuName: 'Test' } });

            const mine = await get(`/devices/${device.id}/catalogue`);
            expect(mine.status).toBe(200);
            expect(mine.data.permissions.manage).toBe(true);
            expect(mine.data.models.find((model) => model.tag === 'qwen2.5-coder:7b').installed).toBe(true);
            expect(mine.data.models.find((model) => model.tag === 'llama3.2:3b').installed).toBe(false);

            signOut();
            const visitor = await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192');
            const pairs = (data) => data.models.map((model) => [model.tag, model.fit.verdict, model.pick]);
            expect(pairs(mine.data)).toEqual(pairs(visitor.data));

            await setHardware(BIG_DEVICE);
        });

        test('marks models that will not fit on the disk', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 500000, diskTotalMb: 1000000 }));
            await testStartup.loginAsUser('user');
            const { data: deviceList } = await get('/devices');
            const response = await get(`/devices/${deviceList.devices[0].id}/catalogue`);
            expect(response.data.models.every((model) => typeof model.fitsDisk === 'boolean')).toBe(true);
        });
    });

    describe('Downloads', () => {
        const finished = async (id, status = 'completed') => {
            for (let attempt = 0; attempt < 200; attempt++) {
                const job = (await get('/models/downloads')).data.downloads.find((download) => download.id === id);
                if (job?.status === status) return job;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            throw new Error(`download ${id} never became ${status}`);
        };

        const halfway = async (id) => {
            for (let attempt = 0; attempt < 200; attempt++) {
                const job = (await get('/models/downloads')).data.downloads.find((download) => download.id === id);
                if (job?.status === 'downloading' && job.percent >= 50) return job;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            throw new Error(`download ${id} never reached halfway`);
        };

        test('an admin starts a download that finishes in the background', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/pull', { name: 'llama3.2:3b' });

            expect(response.status).toBe(202);
            expect(response.data.download).toMatchObject({ model: 'llama3.2:3b' });
            expect(['queued', 'downloading']).toContain(response.data.download.status);

            const job = await finished(response.data.download.id);
            expect(job).toMatchObject({ percent: 100, completedBytes: 10 * MB, totalBytes: 10 * MB, error: null });
            expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(true);

            const stored = await Download.findById(job.id);
            expect(stored.service).toBe('ai-base');
            expect(stored.finishedAt).toBeInstanceOf(Date);
        });

        test('refuses a download that will not fit, before it starts', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 1500, diskTotalMb: 100000 }));
            await testStartup.loginAsUser('admin');
            const response = await post('/models/pull', { name: 'llama3.3:70b' });

            expect(response.status).toBe(422);
            expect(response.data).toMatchObject({ code: 'DISK_FULL', neededMb: 43000, freeMb: 1500 });
            expect(response.data.message).toMatch(/42 GB|43 GB/);
            expect(await Download.countDocuments({})).toBe(0);
        });

        test('stops an unlisted model that turns out to be too large', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 2, diskTotalMb: 100000 }));
            state.pullSizeMb = 50;
            await testStartup.loginAsUser('admin');
            const response = await post('/models/pull', { name: 'someone/unlisted:q4' });
            const job = await finished(response.data.download.id, 'failed');
            expect(job.error.code).toBe('DISK_FULL');
            expect(state.models.some((model) => model.name === 'someone/unlisted:q4')).toBe(false);
        });

        test('runs one download at a time and queues the next', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(first.id);
            const second = (await post('/models/pull', { name: 'mistral:7b' })).data.download;

            const during = (await get('/models/downloads')).data.downloads;
            expect(during.find((job) => job.id === first.id)).toMatchObject({ status: 'downloading', percent: 50 });
            expect(during.find((job) => job.id === second.id).status).toBe('queued');

            state.hold = null;
            await finished(first.id);
            await finished(second.id);
            expect(state.models.map((model) => model.name)).toEqual(expect.arrayContaining(['llama3.2:3b', 'mistral:7b']));
        });

        test('asking again for the same model does not start a second download', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const again = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(again.id).toBe(first.id);
            expect(await Download.countDocuments({})).toBe(1);
            state.hold = null;
            await finished(first.id);
        });

        test('pauses a download and resumes it', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            const paused = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
            expect(paused.status).toBe(200);
            expect(paused.data.download).toMatchObject({ status: 'paused', percent: 50 });
            expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(false);

            state.hold = null;
            const resumed = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(resumed.id).toBe(job.id);
            await finished(job.id);
        });

        test('cancels a download', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            const cancelled = await post('/models/downloads/cancel', { id: job.id, action: 'cancel' });
            expect(cancelled.data.download.status).toBe('cancelled');
            state.hold = null;
            await downloads.idle();
            expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(false);
        });

        test('cancels a download that is still waiting in the queue', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(first.id);
            const second = (await post('/models/pull', { name: 'mistral:7b' })).data.download;

            const cancelled = await post('/models/downloads/cancel', { id: second.id, action: 'cancel' });
            expect(cancelled.data.download.status).toBe('cancelled');
            state.hold = null;
            await finished(first.id);
            await downloads.idle();
            expect(state.models.some((model) => model.name === 'mistral:7b')).toBe(false);
        });

        test('reports an interrupted download in plain words and lets it resume', async () => {
            state.pullFailsMidway = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const failed = await finished(job.id, 'failed');
            expect(failed.error).toMatchObject({ code: 'DOWNLOAD_INTERRUPTED' });
            expect(failed.error.action).toMatch(/resume/i);

            state.pullFailsMidway = false;
            const retry = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(retry.id);
        });

        test('reports that Ollama is not running', async () => {
            state.up = false;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('OLLAMA_DOWN');
        });

        test('picks up a download that was interrupted by a restart', async () => {
            await testStartup.loginAsUser('admin');
            const stale = await Download.create({
                model: 'llama3.2:3b', status: 'downloading', completedBytes: 3 * MB, totalBytes: 10 * MB,
                requestedBy: testStartup.admin.id || testStartup.admin._id
            });
            await downloads.recover();
            const job = await finished(stale.id);
            expect(job.percent).toBe(100);
        });

        test('a regular user can watch downloads but not change them', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            await testStartup.loginAsUser('user');
            const list = await get('/models/downloads');
            expect(list.status).toBe(200);
            expect(list.data.permissions.manage).toBe(false);
            expect(list.data.downloads[0].id).toBe(job.id);
            expect((await post('/models/downloads/cancel', { id: job.id, action: 'cancel' })).status).toBe(403);

            state.hold = null;
            await testStartup.loginAsUser('admin');
            await finished(job.id);
        });

        test('rejects a malformed cancel request and an unknown download', async () => {
            await testStartup.loginAsUser('admin');
            expect((await post('/models/downloads/cancel', { id: 'nope', action: 'cancel' })).status).toBe(400);
            expect((await post('/models/downloads/cancel', { id: '64b7f0c2a1b2c3d4e5f60718', action: 'explode' })).status).toBe(400);
            expect((await post('/models/downloads/cancel', { id: '64b7f0c2a1b2c3d4e5f60718', action: 'cancel' })).status).toBe(404);
        });

        const until = async (id, check, what) => {
            for (let attempt = 0; attempt < 200; attempt++) {
                const job = (await get('/models/downloads')).data.downloads.find((download) => download.id === id);
                if (job && check(job)) return job;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            throw new Error(`download ${id} never ${what}`);
        };

        test('says so when no data is arriving, then carries on when it returns', async () => {
            downloads.stallWarnMs = 100;
            state.stall = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;

            const stalled = await until(job.id, (current) => current.stalled && current.percent >= 50, 'reported a stall');
            expect(stalled).toMatchObject({ status: 'downloading', percent: 50, speedBps: 0, error: null });

            state.stall = false;
            const done = await finished(job.id);
            expect(done.stalled).toBe(false);
        });

        test('a stalled download can still be paused and cancelled', async () => {
            downloads.stallWarnMs = 100;
            state.stall = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await until(job.id, (current) => current.stalled, 'reported a stall');

            const paused = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
            expect(paused.data.download).toMatchObject({ status: 'paused', stalled: false });

            const cancelled = await post('/models/downloads/cancel', { id: job.id, action: 'cancel' });
            expect(cancelled.data.download.status).toBe('cancelled');
        });

        test('gives up on a download that stays stalled, in plain words, and lets it resume', async () => {
            downloads.stallWarnMs = 100;
            downloads.stallFailMs = 400;
            state.stall = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;

            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('DOWNLOAD_INTERRUPTED');
            expect(failed.error.action).toMatch(/resume/i);
            expect(failed.stalled).toBe(false);

            state.stall = false;
            const retry = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(retry.id).toBe(job.id);
            expect(retry.error).toBeNull();
            await finished(retry.id);
            expect(await Download.countDocuments({ model: 'llama3.2:3b' })).toBe(1);
        });

        test('gives up when a download never gets going', async () => {
            downloads.stallWarnMs = 100;
            downloads.stallFailMs = 300;
            state.silent = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('DOWNLOAD_INTERRUPTED');
            state.silent = false;
        });

        test('keeps downloading when progress cannot be saved, and saves once it can', async () => {
            const realUpdate = Download.updateOne;
            downloads.retryMs = 50;
            downloads.saveWaitMs = 200;
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            try {
                Download.updateOne = () => Promise.reject(new Error('database is unreachable'));
                state.hold = null;

                const shown = await finished(job.id);
                expect(shown.percent).toBe(100);
                expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(true);
                expect((await Download.findById(job.id)).status).not.toBe('completed');

                const next = (await post('/models/pull', { name: 'mistral:7b' })).data.download;
                await finished(next.id);
            } finally {
                Download.updateOne = realUpdate;
            }

            for (let attempt = 0; attempt < 200; attempt++) {
                if ((await Download.findById(job.id)).status === 'completed') break;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            const stored = await Download.findById(job.id);
            expect(stored.status).toBe('completed');
            expect(stored.finishedAt).toBeInstanceOf(Date);
        });

        test('a download whose result could not be saved can still be paused', async () => {
            const realUpdate = Download.updateOne;
            downloads.retryMs = 50;
            downloads.saveWaitMs = 200;
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            try {
                Download.updateOne = () => Promise.reject(new Error('database is unreachable'));
                const paused = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
                expect(paused.data.download.status).toBe('paused');
            } finally {
                Download.updateOne = realUpdate;
            }

            state.hold = null;
            const resumed = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(resumed.id).toBe(job.id);
            await finished(job.id);
        });

        test('says plainly when the database cannot be reached', async () => {
            const realCheck = health.databaseReady;
            await testStartup.loginAsUser('admin');
            try {
                health.databaseReady = () => false;
                for (const response of [await get('/models/downloads'), await get('/devices'), await post('/models/pull', { name: 'llama3.2:3b' })]) {
                    expect(response.status).toBe(503);
                    expect(response.data).toMatchObject({ success: false, code: 'DATABASE_DOWN' });
                    expect(response.data.action).toMatch(/internet/i);
                }
            } finally {
                health.databaseReady = realCheck;
            }
            expect((await get('/models/downloads')).status).toBe(200);
        });

        test('refuses a listed model that is too large for the device, before it is queued', async () => {
            await setHardware({ memoryMb: 32768, gpuMemoryMb: 8192, gpuName: 'Test card' });
            const response = await post('/models/pull', { name: 'llama3.3:70b' });
            await setHardware(BIG_DEVICE);

            expect(response.status).toBe(422);
            expect(response.data).toMatchObject({ success: false, code: 'MODEL_TOO_LARGE' });
            expect(response.data.message).toMatch(/8 GB of graphics memory and 32 GB of system memory/);
            expect(response.data.action).toMatch(/smaller/i);
            expect(await Download.countDocuments({})).toBe(0);
        });

        test('still queues a listed model that fits', async () => {
            await setHardware({ memoryMb: 32768, gpuMemoryMb: 8192, gpuName: 'Test card' });
            const response = await post('/models/pull', { name: 'qwen3-coder:30b' });
            await setHardware(BIG_DEVICE);
            expect(response.status).toBe(202);
            await finished(response.data.download.id);
        });

        test('stops an unlisted model as soon as it proves too large for the device', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 50000000, diskTotalMb: 90000000 }));
            state.pullSizeMb = 4000000;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'someone/enormous:q8' })).data.download;

            const failed = await finished(job.id, 'failed');
            expect(failed.error).toMatchObject({ code: 'MODEL_TOO_LARGE' });
            expect(failed.error.message).toMatch(/too large for this device/i);
            expect(state.models.some((model) => model.name === 'someone/enormous:q8')).toBe(false);
        });

        test('says plainly when no model has that name', async () => {
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;

            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('MODEL_NOT_FOUND');
            expect(failed.error.message).toMatch(/no model with this name/i);
            expect(failed.error.message).not.toMatch(/not installed/i);
            expect(failed.error.action).toMatch(/spelling/i);
        });

        test('a failed download can be removed from the list', async () => {
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            await finished(job.id, 'failed');

            const removed = await post('/models/downloads/cancel', { id: job.id, action: 'cancel' });
            expect(removed.status).toBe(200);
            expect(removed.data.download).toMatchObject({ status: 'cancelled', error: expect.anything() });

            const listed = (await get('/models/downloads')).data.downloads.find((download) => download.id === job.id);
            expect(listed.status).toBe('cancelled');
            expect((await Download.findById(job.id)).status).toBe('cancelled');
        });

        test('a finished download clears older failed rows for the same model', async () => {
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(first.id);
            const old = await Download.create([
                { model: 'llama3.2:3b', status: 'failed', errorCode: 'OLLAMA_DOWN', requestedBy: first.requestedBy },
                { model: 'llama3.2:3b', status: 'failed', errorCode: 'DOWNLOAD_INTERRUPTED', requestedBy: first.requestedBy },
                { model: 'mistral:7b', status: 'failed', errorCode: 'OLLAMA_DOWN', requestedBy: first.requestedBy }
            ]);

            const retry = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(retry.id);

            const left = await Download.find({ model: 'llama3.2:3b', status: 'failed' });
            expect(left).toHaveLength(0);
            expect(old.map((doc) => doc.id)).toContain(retry.id);
            expect((await Download.findById(old[2].id)).status).toBe('failed');
        });

        test('a failed download cannot be paused', async () => {
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            await finished(job.id, 'failed');
            const response = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
            expect(response.data.download.status).toBe('failed');
        });

        test('a download that failed does not block the next one', async () => {
            await testStartup.loginAsUser('admin');
            const bad = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            const good = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(bad.id, 'failed');
            await finished(good.id);
        });

        test('streams progress as server-sent events', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const cookies = client.getCookiesForWebSocket();
            const controller = new AbortController();
            const response = await fetch(`${testStartup.baseURL}/api/v1/models/downloads/stream`, {
                headers: { Cookie: cookies }, signal: controller.signal
            });
            expect(response.status).toBe(200);
            expect(response.headers.get('content-type')).toMatch(/text\/event-stream/);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let text = '';
            const readUntil = async (marker) => {
                for (let attempt = 0; attempt < 100 && !text.includes(marker); attempt++) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    text += decoder.decode(value, { stream: true });
                }
            };

            await readUntil('event: snapshot');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await readUntil('"status":"downloading"');
            expect(text).toContain('event: snapshot');
            expect(text).toContain('event: progress');
            expect(text).toContain(job.id);

            controller.abort();
            state.hold = null;
            await finished(job.id);
        });
    });
});
