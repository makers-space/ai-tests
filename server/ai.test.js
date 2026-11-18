import fs from 'node:fs';
import { createRequire } from 'node:module';
import { Readable } from 'node:stream';
import { describe, test, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import TestStartup from '../utils/test.startup.js';
import { getNotificationService } from '../../server/middleware/app.middleware.js';
import { ROLES, RIGHTS, hasRight } from '../../server/config/rights.js';
import {
    setOllamaClientFactory, resetOllamaClientFactory, createOllamaClient, setLibraryFetch, parseLibrary, parseParameters, parseLibrarySizes, friendlyName,
    mapOllamaError, OllamaError, ERROR_CODES,
    parseNvidiaSmi, appleGpus, parseDisplayAdapters, detectHardware,
    fitModel, rankCatalogue, memoryNeededMb, VERDICTS,
    GenerationQueue, extractText, estimateTokens, setFilesFetch, MAX_ATTACHMENT_BYTES,
    isPublicAddress, setRemoteTransport, MAX_REMOTE_BYTES, setSearchFetch,
    setToolConnector, ToolSignInError, TOOL_HEALTH, TOOL_CATALOGUE, TOOL_LOCAL_CATALOGUE, CLOUD_PROVIDERS, PROVIDER_HEALTH, setCloudFetch, ollamaOnlineNames, runsOnline, seal, unseal, deviceGateway, DeviceGateway, cloudTitle
} from '../../server/middleware/ai.middleware.js';
import { Device, Setting, Download, PersonalModel, Chat, Prompt, Guide, GUIDE_STEPS, MANAGER_GUIDE_STEPS, deviceSchemas, modelSchemas, chatSchemas, promptSchemas, ToolServer, TOOL_HEALTH_STATES, toolSchemas, Provider, PROVIDER_HEALTH_STATES, providerSchemas } from '../../server/models/ai.model.js';
import { downloads, health, queue, COMPONENT_GUIDE, BASE_GUIDE, CLOSING_GUIDE, SKILLS, SEARCH_GUIDE, PLAN_GUIDE, RESULTS_GUIDE, REVIEW_GUIDE, TOUCH_GUIDE, writtenToolCalls, BUILTIN_TOOLS, waitingPermissions, todayLine, SKILL_SOURCES, MAX_CHAT_SKILLS } from '../../server/controllers/ai.controller.js';

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
        return { models: models.map((model) => (typeof model === 'string' ? { name: model } : model)) };
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

    test('an Apple-chip Mac reports its built-in graphics with a share of system memory', () => {
        const [gpu] = appleGpus({ platform: 'darwin', arch: 'arm64', cpu: 'Apple M2 Pro', memoryMb: 32768 });
        expect(gpu.name).toBe('Apple M2 Pro built-in graphics');
        expect(gpu.memoryMb).toBe(21845);
        expect(appleGpus({ platform: 'darwin', arch: 'arm64', cpu: '', memoryMb: 8192 })[0].name).toBe('Apple chip built-in graphics');
    });

    test.each([
        ['an Intel Mac', { platform: 'darwin', arch: 'x64', cpu: 'Intel Core i7', memoryMb: 16384 }],
        ['a Windows computer', { platform: 'win32', arch: 'x64', cpu: 'AMD Ryzen 7', memoryMb: 32768 }],
        ['a Linux computer on an ARM chip', { platform: 'linux', arch: 'arm64', cpu: 'Cortex', memoryMb: 8192 }],
        ['a Mac with unknown memory', { platform: 'darwin', arch: 'arm64', cpu: 'Apple M1', memoryMb: 0 }]
    ])('%s gets no Apple graphics', (_label, machine) => {
        expect(appleGpus(machine)).toEqual([]);
    });

    test('Windows display adapters: an AMD card with its own memory counts, others do not', () => {
        const GB = 1024 * 1024 * 1024;
        const adapters = JSON.stringify([
            { DriverDesc: 'AMD Radeon RX 7800 XT', 'HardwareInformation.qwMemorySize': 16 * GB },
            { DriverDesc: 'AMD Radeon(TM) Graphics', 'HardwareInformation.qwMemorySize': 0.5 * GB },
            { DriverDesc: 'Intel(R) UHD Graphics 770', 'HardwareInformation.qwMemorySize': 1 * GB },
            { DriverDesc: 'NVIDIA GeForce RTX 4060', 'HardwareInformation.qwMemorySize': 8 * GB },
            { DriverDesc: 'Microsoft Basic Display Adapter' }
        ]);
        expect(parseDisplayAdapters(adapters)).toEqual([{ name: 'AMD Radeon RX 7800 XT', memoryMb: 16384 }]);
    });

    test('Windows display adapters: one adapter arrives as a single object, and bad output is ignored', () => {
        const one = JSON.stringify({ DriverDesc: 'Radeon RX 6600', 'HardwareInformation.qwMemorySize': 8 * 1024 * 1024 * 1024 });
        expect(parseDisplayAdapters(one)).toEqual([{ name: 'Radeon RX 6600', memoryMb: 8192 }]);
        for (const bad of ['', null, undefined, 'not json', '[1, 2]', '{"DriverDesc": 5}']) {
            expect(parseDisplayAdapters(bad)).toEqual([]);
        }
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
            expect(device.ollama).toEqual({ available: true, version: '9.9.9-test', modelCount: 2, chatCount: 2 });
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
            expect(response.data.devices[0].ollama).toEqual({ available: false, version: null, modelCount: 0, chatCount: 0 });
        });

        test('models that only make embeddings are counted, but not as models that can chat', async () => {
            setOllamaClientFactory(simpleOllama({ models: [{ name: 'bge-large:335m', details: { family: 'bert' } }, { name: 'mxbai-embed-large:335m', details: { family: 'bert' } }] }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');
            expect(response.data.devices[0].ollama).toMatchObject({ available: true, modelCount: 2, chatCount: 0 });

            setOllamaClientFactory(simpleOllama({ models: ['mxbai-embed-large:335m', 'nomic-embed-text:latest', 'small:1b'] }));
            const mixed = await client.get('/api/v1/devices');
            expect(mixed.data.devices[0].ollama).toMatchObject({ modelCount: 3, chatCount: 1 });
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

// Sample models for the fit and ranking tests. The app itself reads the live list from ollama.com.
const catalogue = [
    {"tag":"llama3.2:3b","name":"Llama 3.2 3B","description":"Small and quick. Good for short questions, rewording and simple summaries.","goodFor":["chat","writing"],"parameters":3.2,"sizeMb":2000,"moe":false,"capabilities":["tools"]},
    {"tag":"phi4-mini:3.8b","name":"Phi-4 Mini","description":"A compact model that reasons well for its size. Good for quick answers on modest hardware.","goodFor":["chat","reasoning"],"parameters":3.8,"sizeMb":2500,"moe":false,"capabilities":["tools"]},
    {"tag":"gemma3:4b","name":"Gemma 3 4B","description":"A small general model with a friendly writing style. Good for drafting and everyday chat.","goodFor":["chat","writing"],"parameters":4.3,"sizeMb":3300,"moe":false,"capabilities":["vision"]},
    {"tag":"mistral:7b","name":"Mistral 7B","description":"A dependable all-rounder for chat, summaries and short documents.","goodFor":["chat","writing"],"parameters":7.2,"sizeMb":4400,"moe":false,"capabilities":["tools"]},
    {"tag":"qwen2.5-coder:7b","name":"Qwen 2.5 Coder 7B","description":"Tuned for programming. Good for explaining, writing and fixing code.","goodFor":["code"],"parameters":7.6,"sizeMb":4700,"moe":false,"capabilities":["tools"]},
    {"tag":"llama3.1:8b","name":"Llama 3.1 8B","description":"A well-rounded everyday assistant that can also use tools, so it suits simple agents.","goodFor":["chat","writing"],"parameters":8,"sizeMb":4900,"moe":false,"capabilities":["tools"]},
    {"tag":"deepseek-r1:8b","name":"DeepSeek R1 8B","description":"Thinks step by step before answering. Good for problems that need working out.","goodFor":["reasoning"],"parameters":8.2,"sizeMb":5200,"moe":false,"capabilities":["thinking"]},
    {"tag":"gemma3:12b","name":"Gemma 3 12B","description":"A stronger writer than the small models. Good for longer drafts and careful rewording.","goodFor":["chat","writing"],"parameters":12.2,"sizeMb":8100,"moe":false,"capabilities":["vision"]},
    {"tag":"qwen2.5:14b","name":"Qwen 2.5 14B","description":"A capable general model for documents, analysis and tool use.","goodFor":["chat","writing","reasoning"],"parameters":14.8,"sizeMb":9000,"moe":false,"capabilities":["tools"]},
    {"tag":"phi4:14b","name":"Phi-4 14B","description":"Careful and precise. Good for analysis, structured answers and checking work.","goodFor":["reasoning","writing"],"parameters":14.7,"sizeMb":9100,"moe":false,"capabilities":[]},
    {"tag":"deepseek-r1:14b","name":"DeepSeek R1 14B","description":"Step-by-step reasoning with more depth than the 8B version.","goodFor":["reasoning"],"parameters":14.8,"sizeMb":9000,"moe":false,"capabilities":["thinking"]},
    {"tag":"gpt-oss:20b","name":"GPT-OSS 20B","description":"A strong general model that reasons and uses tools. Only part of it is active at a time, so it runs on less graphics memory than its size suggests.","goodFor":["chat","reasoning","writing"],"parameters":20.9,"sizeMb":14000,"moe":true,"capabilities":["tools","thinking"]},
    {"tag":"gemma3:27b","name":"Gemma 3 27B","description":"The largest Gemma. Good for long-form writing and detailed answers on a large graphics card.","goodFor":["chat","writing"],"parameters":27.4,"sizeMb":17000,"moe":false,"capabilities":["vision"]},
    {"tag":"qwen3:30b","name":"Qwen 3 30B","description":"A large general model that thinks before answering and uses tools. Only part of it is active at a time, so system memory can carry most of it.","goodFor":["chat","reasoning","writing"],"parameters":30.5,"sizeMb":19000,"moe":true,"capabilities":["tools","thinking"]},
    {"tag":"qwen3-coder:30b","name":"Qwen 3 Coder 30B","description":"A large programming model for bigger coding tasks and agents. Only part of it is active at a time, so system memory can carry most of it.","goodFor":["code"],"parameters":30.5,"sizeMb":19000,"moe":true,"capabilities":["tools"]},
    {"tag":"qwen2.5-coder:32b","name":"Qwen 2.5 Coder 32B","description":"A top programming model for a 24 GB graphics card.","goodFor":["code"],"parameters":32.8,"sizeMb":20000,"moe":false,"capabilities":["tools"]},
    {"tag":"llama3.3:70b","name":"Llama 3.3 70B","description":"A very large general model for workstations with a lot of graphics memory.","goodFor":["chat","writing","reasoning"],"parameters":70.6,"sizeMb":43000,"moe":false,"capabilities":["tools"]}
];

// Stands in for ollama.com: a search page built from the sample models, and a tags page with each size on disk.
const library = { down: false, empty: false, asked: [], marked: [] };
const libraryNames = () => [...new Set(catalogue.map((model) => model.tag.split(':')[0]))];
const fakeLibrary = async (address, options = {}) => {
    const url = new URL(address);
    library.asked.push(url.pathname + url.search);
    if (options.headers?.['HX-Request']) library.marked.push(url.pathname + url.search);
    if (library.down) throw new TypeError('fetch failed');
    const page = (html) => ({ ok: true, status: 200, text: async () => html });
    const tags = url.pathname.match(/^\/library\/([^/]+)\/tags$/);
    if (tags) {
        return page(catalogue.filter((model) => model.tag.startsWith(`${tags[1]}:`)).map((model) => `<a href="/library/${model.tag}"><span>${model.tag}</span><span>abc123 • ${model.sizeMb}MB • 128K context window</span></a>`).join(''));
    }
    if (library.empty) return page('<html><body>Nothing here</body></html>');
    const wanted = (url.searchParams.get('q') || '').toLowerCase();
    const second = url.searchParams.get('page') === '2' && Boolean(options.headers?.['HX-Request']);
    if (Number(url.searchParams.get('page')) > 2) return page('<ul></ul>');
    const names = libraryNames().filter((name) => name.includes(wanted)).filter((name, index) => (wanted ? true : second ? index >= 8 : index < 8));
    const cloud = !second && 'kimi-k3'.includes(wanted) ? '<li><a href="/library/kimi-k3"><h2><span>kimi-k3</span></h2><span>tools</span><span>cloud</span><p>Runs on the maker\'s servers.</p></a></li>' : '';
    const items = names.map((name) => {
        const sizes = catalogue.filter((model) => model.tag.startsWith(`${name}:`));
        return `<li><a href="/library/${name}" class="group"><h2 title="${name}"><span>${name}</span></h2><span title="Updated October 1, 2026">1mo</span>${sizes[0].capabilities.map((ability) => `<span class="a">${ability}</span>`).join('')}${sizes.map((model) => `<span class="s">${model.tag.split(':')[1]}</span>`).join('')}<p class="d">${sizes[0].description}${sizes[0].moe ? ' A mixture of experts model.' : ''}</p><span title="1,234,567 downloads"><span>1.2M</span></span></a></li>`;
    });
    return page(`<ul>${cloud}${items.join('')}</ul>${!wanted && !second ? '<div hx-get="/search?page=2"></div>' : ''}`);
};
setLibraryFetch(fakeLibrary);
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
        silent: false,
        chatReply: 'A short answer',
        chatCalls: [],
        guidesSeen: 0,
        toolPlan: [],
        toolsAsText: false,
        plan: { searches: [] },
        planCalls: [],
        planAborted: 0,
        searchRequests: [],
        reviews: [],
        reviewCalls: [],
        freshResults: false,
        enginesLimited: false,
        junkResults: false,
        planHold: false,
        searches: [],
        fetches: [],
        searchFails: false,
        searchStatus: null,
        searchUrls: [],
        searchGuides: 0,
        noTools: false,
        chatAborted: 0,
        chatHold: false,
        chatError: null,
        chatFailsMidway: false,
        thinkingModel: false,
        contextLength: 32768,
        filesDown: false
    });
};

const fakeChat = (request) => {
    let aborted = false;
    let wake = () => {};
    const check = () => {
        if (aborted) throw Object.assign(new Error('The operation was aborted'), { name: 'AbortError' });
    };
    const pause = (ms) => new Promise((resolve) => {
        wake = resolve;
        setTimeout(resolve, ms);
    });
    const words = state.chatReply.split(' ');

    return {
        abort() {
            if (!aborted) state.chatAborted += 1;
            aborted = true;
            wake();
        },
        async *[Symbol.asyncIterator]() {
            if (request.think) yield { message: { role: 'assistant', content: '', thinking: 'Let me consider this.' } };
            for (let index = 0; index < words.length; index++) {
                await pause(2);
                check();
                yield { message: { role: 'assistant', content: index < words.length - 1 ? `${words[index]} ` : words[index] } };
                if (state.chatFailsMidway) throw refused();
                while (state.chatHold) {
                    await pause(10);
                    check();
                }
            }
            yield {
                message: { role: 'assistant', content: '' },
                done: true,
                eval_count: words.length,
                prompt_eval_count: 20,
                total_duration: 2e9,
                eval_duration: 1e9
            };
        }
    };
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
            capabilities: state.noTools ? ['completion'] : state.thinkingModel ? ['completion', 'tools', 'thinking'] : ['completion', 'tools'],
            model_info: { 'testfam.context_length': state.contextLength },
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
    generate: async ({ model, keep_alive, options }) => {
        if (!state.models.some((existing) => existing.name === model)) throw notFound(model);
        state.unloaded.push(options ? { model, keep_alive, num_ctx: options.num_ctx } : { model, keep_alive });
        return { done: true };
    },
    chat: async (request) => {
        if (!state.up) throw refused();
        if (!state.models.some((existing) => existing.name === request.model)) throw notFound(request.model);
        const lead = request.messages[0];
        if (lead?.role === 'system' && lead.content.startsWith(COMPONENT_GUIDE)) {
            state.guidesSeen += 1;
            let own = lead.content.slice(COMPONENT_GUIDE.length).replace(/^\n\n/, '');
            own = own.split('\n\n').filter((part) => part !== todayLine()).join('\n\n');
            if (own.endsWith(CLOSING_GUIDE)) own = own.slice(0, own.length - CLOSING_GUIDE.length).replace(/\n\n$/, '');
            if (own.endsWith(SEARCH_GUIDE)) {
                state.searchGuides += 1;
                own = own.slice(0, own.length - SEARCH_GUIDE.length).replace(/\n\n$/, '');
            }
            request = { ...request, messages: own ? [{ role: 'system', content: own }, ...request.messages.slice(1)] : request.messages.slice(1) };
        }
        state.chatCalls.push(request);
        if (state.chatError) throw Object.assign(new Error(state.chatError), { error: state.chatError });
        if (request.format) {
            const answer = (content) => {
                const result = { message: { role: 'assistant', content }, done: true };
                if (request.stream === false) return result;
                let aborted = false;
                return {
                    abort() {
                        aborted = true;
                        state.planAborted += 1;
                    },
                    async *[Symbol.asyncIterator]() {
                        while (state.planHold && !aborted) await new Promise((resolve) => setTimeout(resolve, 10));
                        if (aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
                        yield result;
                    }
                };
            };
            state.chatCalls.pop();
            if (request.format.properties?.enough) {
                state.reviewCalls.push(request);
                const review = state.reviews.shift() ?? { enough: true, searches: [], pages: [] };
                return answer(typeof review === 'string' ? review : JSON.stringify(review));
            }
            state.planCalls.push(request);
            return answer(typeof state.plan === 'string' ? state.plan : JSON.stringify(state.plan));
        }
        if (request.stream === false) return { message: { role: 'assistant', content: state.chatReply }, done: true };
        if (request.tools && state.toolPlan.length > 0) {
            const calls = state.toolPlan.shift().map(([name, args]) => ({ function: { name, arguments: args } }));
            if (state.toolsAsText) {
                const text = calls.map((call) => JSON.stringify({ name: call.function.name, arguments: call.function.arguments })).join('\n');
                return {
                    abort() {},
                    async *[Symbol.asyncIterator]() {
                        yield { message: { role: 'assistant', content: text.slice(0, 20) } };
                        yield { message: { role: 'assistant', content: text.slice(20) } };
                        yield { message: { role: 'assistant', content: '' }, done: true, eval_count: 3, prompt_eval_count: 20, total_duration: 2e8, eval_duration: 1e8 };
                    }
                };
            }
            return {
                abort() {},
                async *[Symbol.asyncIterator]() {
                    yield { message: { role: 'assistant', content: '', tool_calls: calls } };
                    yield { message: { role: 'assistant', content: '' }, done: true, eval_count: 3, prompt_eval_count: 20, total_duration: 2e8, eval_duration: 1e8 };
                }
            };
        }
        return fakeChat(request);
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
        expect([null, 'code', 'reasoning']).toContain(ranked[3].pick);
        expect(new Set(ranked.slice(0, 4).map((model) => model.tag)).size).toBe(4);
        expect(ranked.slice(4).every((model) => model.pick === null)).toBe(true);
        expect(ranked.find((model) => model.pick === 'best').fit.fits).toBe(true);
        for (const model of ranked.filter((entry) => entry.pick)) {
            expect(model.pickLabel.length).toBeGreaterThan(3);
            expect(model.pickWhy).toMatch(/\d+(\.\d)? GB/);
            expect(model.pickWhy).toMatch(/tokens a second|speed unknown/);
        }
        const fastest = ranked.find((model) => model.pick === 'fast');
        const capable = ranked.filter((model) => model.parameters >= 1 && ['gpu', 'cpu'].includes(model.fit.verdict));
        if (capable.length > 0) expect(fastest.parameters).toBeGreaterThanOrEqual(1);
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
            expect(unknown.displayName).toBe('Custom Thing');
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

        test('a personal model shows its instructions and can be edited', async () => {
            await testStartup.loginAsUser('admin');
            await post('/models/create', careful);
            const before = (await get('/models')).data.models.find((model) => model.name === 'careful-reviewer:latest');
            expect(before).toMatchObject({ system: 'Check the work for mistakes.', parameters: { temperature: 0.2 } });
            expect((await get('/models')).data.models.find((model) => model.name === 'qwen2.5-coder:7b')).toMatchObject({ system: null, parameters: null });

            const response = await post('/models/update', {
                name: 'careful-reviewer:latest', displayName: 'Very careful reviewer', system: 'Check twice.', parameters: { temperature: 0.7, num_ctx: 4096 }
            });
            expect(response.status).toBe(200);
            expect(state.created[1]).toMatchObject({
                model: 'careful-reviewer', from: 'qwen2.5-coder:7b', system: 'Check twice.', parameters: { temperature: 0.7, num_ctx: 4096 }, stream: false
            });
            const after = (await get('/models')).data.models.find((model) => model.name === 'careful-reviewer:latest');
            expect(after).toMatchObject({ displayName: 'Very careful reviewer', basedOn: 'qwen2.5-coder:7b', system: 'Check twice.', parameters: { temperature: 0.7, num_ctx: 4096 } });

            expect((await post('/models/update', { name: 'qwen2.5-coder:7b', system: 'x' })).status).toBe(404);
            expect((await post('/models/update', { name: 'careful-reviewer', from: 'missing:7b' })).status).toBe(404);
            expect((await post('/models/update', { name: 'careful-reviewer', from: 'careful-reviewer' })).status).toBe(404);
            expect((await post('/models/update', { name: 'careful-reviewer' })).status).toBe(400);
            expect(state.created).toHaveLength(2);

            await testStartup.loginAsUser('user');
            expect((await post('/models/update', { name: 'careful-reviewer', system: 'x' })).status).toBe(403);
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
        const everything = async (query = '') => {
            const first = await get(`/catalogue${query ? `?${query}` : ''}`);
            const second = await get(`/catalogue?${query ? `${query}&` : ''}page=2`);
            return { first, second, models: [...first.data.models, ...second.data.models] };
        };

        test('is public and lists the models on ollama.com, a page at a time', async () => {
            signOut();
            const { first, second, models } = await everything();
            expect(first.status).toBe(200);
            expect(first.data).toMatchObject({ source: 'ollama.com', page: 1, more: false, query: '', total: catalogue.length + 1, library: libraryNames().length + 1 });
            expect(second.data).toMatchObject({ page: 2, more: false, models: [] });
            expect(first.data.models.map((model) => model.tag).sort()).toEqual([...catalogue.map((model) => model.tag), 'kimi-k3'].sort());
            expect(first.data.models[first.data.models.length - 1]).toMatchObject({ tag: 'kimi-k3', name: 'Kimi K3', cloudOnly: true, sizeMb: 0 });
            expect((await get('/catalogue?all=true')).data).toMatchObject({ more: false, total: catalogue.length + 1 });
            expect(library.marked).toContain('/search?page=2');
            expect(library.asked).toContain('/search');
            expect(models[0].fit).toBeNull();
            const gemma = models.find((model) => model.tag === 'gemma3:12b');
            expect(gemma).toMatchObject({ name: 'Gemma 3 12B', parameters: 12, sizeEstimated: false, sizeMb: 8100, capabilities: ['vision'], pulls: 1234567, updated: 'October 1, 2026' });
            expect(models.find((model) => model.tag === 'qwen2.5-coder:7b').goodFor).toContain('code');
            expect(models.find((model) => model.tag === 'qwen3:30b').moe).toBe(true);
        });

        test('searches the whole library by name', async () => {
            signOut();
            const found = await get('/catalogue?q=gemma&memoryMb=32768&gpuMemoryMb=8192');
            expect(found.data.query).toBe('gemma');
            expect(found.data.models.map((model) => model.tag).sort()).toEqual(['gemma3:12b', 'gemma3:27b', 'gemma3:4b']);
            expect(found.data.models.every((model) => model.fit.verdict)).toBe(true);
            expect(found.data).toMatchObject({ total: 3, fitting: 3, library: libraryNames().length + 1 });
            const kimi = (await get('/catalogue?q=kimi&memoryMb=32768&gpuMemoryMb=8192')).data;
            expect(kimi.models).toHaveLength(1);
            expect(kimi.models[0]).toMatchObject({ cloudOnly: true, pick: null, fit: { verdict: 'cloud', fits: false, label: 'Cloud only' } });
            expect(kimi.fitting).toBe(0);
            expect((await get('/catalogue?q=coder%20qwen3')).data.models.map((model) => model.tag)).toEqual(['qwen3-coder:30b']);
            expect((await get('/catalogue?q=nothing-like-this')).data.models).toEqual([]);
            expect((await get('/catalogue?page=0')).status).toBe(400);
            expect((await get('/catalogue?size=10')).status).toBe(400);
            const small = await get('/catalogue?size=12');
            expect(small.status).toBe(200);
            expect(small.data.size).toBe(12);
            expect(small.data.models.filter((model) => !model.pick).length).toBeLessThanOrEqual(12);
            expect(small.data.models.filter((model) => model.pick).length).toBeLessThanOrEqual(4);
            expect(small.data.paged).toBe(small.data.total - small.data.models.filter((model) => model.pick).length);
            expect((await get('/catalogue')).data.size).toBe(48);
        });

        test('says plainly when ollama.com cannot be reached or its page cannot be read', async () => {
            signOut();
            setLibraryFetch(fakeLibrary);
            library.down = true;
            try {
                const response = await get('/catalogue');
                expect(response.status).toBe(502);
                expect(response.data).toMatchObject({ code: 'LIBRARY_UNREACHABLE' });
                expect(response.data.message).toMatch(/ollama\.com/);
                library.down = false;
                library.empty = true;
                expect((await get('/catalogue')).data).toMatchObject({ code: 'LIBRARY_UNREADABLE' });
            } finally {
                library.down = false;
                library.empty = false;
                setLibraryFetch(fakeLibrary);
            }
            expect((await get('/catalogue')).status).toBe(200);
        });

        test('reads sizes, abilities and counts from the library pages', () => {
            expect(parseParameters('12b')).toBe(12);
            expect(parseParameters('270m')).toBe(0.27);
            expect(parseParameters('e4b')).toBe(4);
            expect(parseParameters('8x7b')).toBe(56);
            expect(parseParameters('29.2M')).toBeNull();
            expect(parseParameters('latest')).toBeNull();
            expect(['gemma3:12b', 'qwen2.5-coder:7b', 'deepseek-r1:14b', 'gpt-oss:20b', 'phi4-mini:3.8b', 'mixtral:8x7b', 'glm-5.3:latest', 'someone/unlisted:q4'].map(friendlyName))
                .toEqual(['Gemma 3 12B', 'Qwen 2.5 Coder 7B', 'Deepseek R1 14B', 'GPT OSS 20B', 'Phi 4 Mini 3.8B', 'Mixtral 8x7B', 'GLM 5.3', 'someone/unlisted:q4']);
            const [model, other] = parseLibrary('<a href="/library/duck" class="x"><h2><span>duck</span></h2><span>tools</span><span>1b</span><span>7b</span><p>Quacks &amp; codes.</p><span title="2,500 downloads"><span>2.5K</span></span></a><a href="/library/duck:7b">tag</a><a href="/library/goose"><p>Honks.</p><span>cloud</span></a>');
            expect(model).toEqual({ name: 'duck', description: 'Quacks & codes.', capabilities: ['tools'], sizes: ['1b', '7b'], pulls: 2500, updated: null });
            expect(other).toMatchObject({ name: 'goose', capabilities: ['cloud'], sizes: [] });
            expect(parseLibrary('<html>nothing</html>')).toEqual([]);
            expect([...parseLibrarySizes('<a href="/library/duck:7b"><span>duck:7b</span> abc • 4.4GB • 8K</a><a href="/library/duck:1b">x • 815MB •</a>')]).toEqual([['7b', 4506], ['1b', 815]]);
        });

        test('gives verdicts for the specs a visitor enters', async () => {
            signOut();
            const { first: response, models } = await everything('memoryMb=32768&gpuMemoryMb=8192');
            expect(response.status).toBe(200);
            const verdict = (tag) => models.find((model) => model.tag === tag).fit.verdict;
            expect(verdict('llama3.1:8b')).toBe('gpu');
            expect(verdict('qwen3-coder:30b')).toBe('mixed');
            expect(verdict('llama3.3:70b')).toBe('too_large');
            expect(response.data.models.slice(0, 3).map((model) => model.pick)).toEqual(['fast', 'balanced', 'best']);
            expect(response.data.reserved).toEqual({ gpuMemoryMb: 1536, memoryMb: 4096 });
            const notes = (await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192&page=2')).data.notes.join(' ');
            expect(notes).toMatch(/mixture-of-experts/);
            expect(response.data.notes.join(' ')).not.toMatch(/two 7 to 8B/);
        });

        test('explains what the specs mean: two models at once, no graphics card, nothing fits', async () => {
            signOut();
            const notes = async (query) => (await get(`/catalogue?${query}`)).data.notes.join(' ');
            expect(await notes('memoryMb=65536&gpuMemoryMb=24576')).toMatch(/two 7 to 8B models at once/);
            const noCard = await get('/catalogue?memoryMb=16384&gpuMemoryMb=0');
            expect(noCard.data.notes.join(' ')).toMatch(/No graphics card/);
            expect(noCard.data.models.filter((model) => model.fit.fits).every((model) => model.fit.verdict === 'cpu')).toBe(true);
            const small = await get('/catalogue?memoryMb=8192&gpuMemoryMb=4096');
            expect(small.data.models.some((model) => model.pick)).toBe(true);
            expect(await notes('memoryMb=1024&gpuMemoryMb=0')).toMatch(/None of the listed models fits/);
            expect((await get('/catalogue')).data.notes).toEqual([]);
        });

        test('gives the home page a sample: models that run fully, some that need system memory, and a few too large', async () => {
            signOut();
            const response = await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192&sample=true');
            expect(response.status).toBe(200);
            const verdicts = response.data.models.map((model) => model.fit.verdict);
            const count = (verdict) => verdicts.filter((found) => found === verdict).length;
            expect(count('gpu')).toBeGreaterThan(0);
            expect(count('gpu')).toBeLessThanOrEqual(6);
            expect(count('mixed')).toBeGreaterThan(0);
            expect(count('mixed')).toBeLessThanOrEqual(3);
            expect(count('too_large')).toBeGreaterThan(0);
            expect(count('too_large')).toBeLessThanOrEqual(3);
            expect(new Set(response.data.models.map((model) => model.tag)).size).toBe(response.data.models.length);
            expect(response.data.models.some((model) => model.cloudOnly)).toBe(false);
            expect(response.data).toMatchObject({ more: false, total: catalogue.length + 1 });
            expect(response.data.fitting).toBeGreaterThan(count('gpu') + count('mixed') - 1);
            expect((await get('/catalogue?sample=true')).data.models.length).toBe(catalogue.length + 1);
        });

        test('a chip whose graphics share the system memory does not have that memory counted twice', async () => {
            const model = { sizeMb: 17000, parameters: 27 };
            const separate = { memoryMb: 24576, gpuMemoryMb: 16384 };
            expect(fitModel(model, separate).verdict).toBe('mixed');
            expect(fitModel(model, { ...separate, gpuShared: true }).verdict).toBe('too_large');
            expect(fitModel({ sizeMb: 8100, parameters: 12 }, { ...separate, gpuShared: true }).verdict).toBe('gpu');

            signOut();
            const apart = (await get('/catalogue?memoryMb=24576&gpuMemoryMb=16384&all=true')).data;
            const shared = (await get('/catalogue?memoryMb=24576&gpuMemoryMb=16384&shared=true&all=true')).data;
            expect(shared.hardware.gpuShared).toBe(true);
            expect(shared.fitting).toBeLessThan(apart.fitting);
            expect(shared.models.find((entry) => entry.tag === 'gemma3:27b').fit.verdict).toBe('too_large');
            expect(apart.models.find((entry) => entry.tag === 'gemma3:27b').fit.verdict).toBe('mixed');
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

        test('a finished download and a finished reply are announced over the notification socket', async () => {
            await testStartup.loginAsUser('admin');
            const token = (await client.get('/api/v1/auth/ws-token')).data.token;
            const messages = [];
            const ws = new WebSocket(`ws://localhost:${testStartup.port}/notifications?token=${encodeURIComponent(token)}`);
            ws.addEventListener('message', (event) => messages.push(JSON.parse(event.data)));
            await new Promise((resolve, reject) => {
                ws.addEventListener('open', resolve);
                ws.addEventListener('error', () => reject(new Error('socket error')));
            });
            const seen = async (type) => {
                for (let attempt = 0; attempt < 200; attempt++) {
                    const found = messages.find((message) => message.type === type);
                    if (found) return found;
                    await new Promise((resolve) => setTimeout(resolve, 25));
                }
                throw new Error(`no ${type}; saw ${messages.map((message) => message.type).join(', ')}`);
            };

            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await seen('activity:changed');
            const finished = await seen('download:finished');
            expect(finished.data).toMatchObject({ level: 'success', message: 'llama3.2:3b has finished downloading.', model: 'llama3.2:3b', id: job.id });
            expect(messages.filter((message) => message.type === 'download:finished')).toHaveLength(1);

            const bad = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            const failed = await seen('download:failed');
            expect(failed.data).toMatchObject({ level: 'error', id: bad.id });
            expect(failed.data.message).toMatch(/nosuch-model:9b could not be downloaded/);

            const chat = (await post('/chats', { model: 'qwen2.5-coder:7b' })).data.chat;
            const answer = await client.post(`/api/v1/chats/${chat.id}/messages`, { content: 'hello there' });
            expect(answer.status).toBe(200);
            const ready = await seen('reply:finished');
            expect(ready.data).toMatchObject({ level: 'info', chatId: chat.id, path: `/chat/${chat.id}` });
            expect(ready.data.message).toContain('hello there');
            ws.close();
        });

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
            expect(await downloads.idle()).toBe(true);
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

describe('Generation queue', () => {
    const gate = () => {
        let open;
        const waiting = new Promise((resolve) => {
            open = resolve;
        });
        return { open, waiting };
    };
    const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

    test('runs one job at a time by default and reports the place in line', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        const a = line.add({ userId: 'u1', run: async () => { ran.push('a'); await first.waiting; return 'A'; } });
        const b = line.add({ userId: 'u2', run: async () => { ran.push('b'); return 'B'; } });
        await settle();

        expect(ran).toEqual(['a']);
        expect(a.ahead()).toBe(0);
        expect(b.ahead()).toBe(1);

        first.open();
        expect(await a.done).toBe('A');
        expect(await b.done).toBe('B');
    });

    test('runs chat ahead of background work', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'u1', run: () => first.waiting });
        await settle();
        const background = line.add({ userId: 'u1', priority: 'background', run: async () => ran.push('background') });
        const chat = line.add({ userId: 'u2', priority: 'interactive', run: async () => ran.push('chat') });

        expect(chat.ahead()).toBe(1);
        expect(background.ahead()).toBe(2);
        first.open();
        await Promise.all([background.done, chat.done]);
        expect(ran).toEqual(['chat', 'background']);
    });

    test('alternates between two users', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'blocker', run: () => first.waiting });
        await settle();
        const jobs = [
            line.add({ userId: 'a', run: async () => ran.push('a1') }),
            line.add({ userId: 'a', run: async () => ran.push('a2') }),
            line.add({ userId: 'a', run: async () => ran.push('a3') }),
            line.add({ userId: 'b', run: async () => ran.push('b1') }),
            line.add({ userId: 'b', run: async () => ran.push('b2') })
        ];
        first.open();
        await Promise.all(jobs.map((job) => job.done));
        expect(ran).toEqual(['a1', 'b1', 'a2', 'b2', 'a3']);
    });

    test('refuses a sixth waiting job from one user, but not from another', async () => {
        const line = new GenerationQueue();
        const first = gate();
        line.add({ userId: 'a', run: () => first.waiting });
        await settle();
        for (let count = 0; count < 5; count++) line.add({ userId: 'a', run: async () => {} });

        expect(() => line.add({ userId: 'a', run: async () => {} })).toThrow(OllamaError);
        try {
            line.add({ userId: 'a', run: async () => {} });
        } catch (error) {
            expect(error.code).toBe('QUEUE_FULL');
            expect(error.status).toBe(429);
        }
        expect(() => line.add({ userId: 'b', run: async () => {} })).not.toThrow();
        first.open();
    });

    test('runs several at once when allowed', async () => {
        const line = new GenerationQueue({ maxConcurrent: 2 });
        const first = gate();
        const ran = [];
        line.add({ userId: 'a', run: async () => { ran.push('a'); await first.waiting; } });
        line.add({ userId: 'b', run: async () => { ran.push('b'); await first.waiting; } });
        const third = line.add({ userId: 'c', run: async () => ran.push('c') });
        await settle();
        expect(ran).toEqual(['a', 'b']);
        expect(third.ahead()).toBe(2);
        first.open();
        await third.done;
    });

    test('each computer has its own line, so one never waits for another', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'a', lane: 'mac', run: async () => { ran.push('mac'); await first.waiting; } });
        const other = line.add({ userId: 'a', lane: 'pc', run: async () => ran.push('pc') });
        const behind = line.add({ userId: 'b', lane: 'mac', run: async () => ran.push('mac again') });
        await other.done;
        expect(ran).toEqual(['mac', 'pc']);
        expect(behind.ahead()).toBe(1);
        expect(line.runningIn('mac')).toBe(1);
        await settle();
        expect(line.busyIn('pc')).toBe(false);
        first.open();
        await behind.done;
        expect(ran).toEqual(['mac', 'pc', 'mac again']);
    });

    test('a cancelled job never runs, and a failed job does not block the next', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'a', run: () => first.waiting });
        await settle();
        const dropped = line.add({ userId: 'b', run: async () => ran.push('dropped') });
        const failing = line.add({ userId: 'c', run: async () => { throw new Error('boom'); } });
        const last = line.add({ userId: 'd', run: async () => ran.push('last') });

        expect(dropped.cancel()).toBe(true);
        expect(await dropped.done).toBeNull();
        first.open();
        await expect(failing.done).rejects.toThrow('boom');
        await last.done;
        expect(ran).toEqual(['last']);
    });
});

const serverRequire = createRequire(new URL('../../server/package.json', import.meta.url));

const makePdf = (text) => {
    const stream = `BT /F1 12 Tf 72 720 Td (${text}) Tj ET`;
    const objects = [
        '<< /Type /Catalog /Pages 2 0 R >>',
        '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>',
        `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`,
        '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'
    ];
    let body = '%PDF-1.4\n';
    const offsets = [];
    objects.forEach((object, index) => {
        offsets.push(body.length);
        body += `${index + 1} 0 obj\n${object}\nendobj\n`;
    });
    const xref = body.length;
    body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
    offsets.forEach((offset) => {
        body += `${String(offset).padStart(10, '0')} 00000 n \n`;
    });
    body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
    return Buffer.from(body, 'latin1');
};

const makeDocx = async (text) => {
    const JSZip = serverRequire('jszip');
    const zip = new JSZip();
    zip.file('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>');
    zip.file('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>');
    zip.file('word/document.xml', `<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>${text}</w:t></w:r></w:p></w:body></w:document>`);
    return zip.generateAsync({ type: 'nodebuffer' });
};

describe('Reading text from documents', () => {
    test.each([
        ['notes.txt', 'Plain notes about the harbour survey.'],
        ['notes.md', '# Survey\n\nThe **harbour** is deep.'],
        ['figures.csv', 'port,depth\nHarwich,12\nDover,9']
    ])('reads %s as it is', async (name, content) => {
        expect(await extractText({ buffer: Buffer.from(content), name })).toBe(content);
    });

    test('reads the text of a PDF', async () => {
        const text = await extractText({ buffer: makePdf('The quarterly harbour report'), name: 'report.pdf' });
        expect(text).toContain('The quarterly harbour report');
    });

    test('reads the text of a Word file', async () => {
        const text = await extractText({ buffer: await makeDocx('Minutes of the harbour board'), name: 'minutes.docx' });
        expect(text).toContain('Minutes of the harbour board');
    });

    test('removes a byte order mark and Windows line endings', async () => {
        const text = await extractText({ buffer: Buffer.from('﻿one\r\ntwo'), name: 'a.txt' });
        expect(text).toBe('one\ntwo');
    });

    test.each([
        ['photo.png', Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 13, 0, 1, 2, 3])],
        ['clip.mp4', Buffer.from([0, 0, 0, 24, 0x66, 0x74, 0x79, 0x70, 0, 0, 0, 0])],
        ['old.doc', Buffer.from('looks like text but is an old Word file')],
        ['broken.pdf', Buffer.from('this is not a pdf at all')],
        ['empty.txt', Buffer.from('   \n  ')]
    ])('refuses %s in plain words', async (name, buffer) => {
        await expect(extractText({ buffer, name })).rejects.toMatchObject({ code: 'FILE_UNREADABLE', status: 415 });
    });

    test('refuses a file over 20 MB', async () => {
        const buffer = Buffer.alloc(MAX_ATTACHMENT_BYTES + 1, 'a');
        await expect(extractText({ buffer, name: 'huge.txt' })).rejects.toMatchObject({ code: 'FILE_TOO_LARGE', status: 413 });
    });

    test('estimates tokens from length', () => {
        expect(estimateTokens('')).toBe(0);
        expect(estimateTokens('abcd')).toBe(1);
        expect(estimateTokens('a'.repeat(4001))).toBe(1001);
    });
});

describe('Chat validation', () => {
    const ok = (schema, value) => expect(schema.validate(value).error).toBeUndefined();
    const bad = (schema, value) => expect(schema.validate(value).error).toBeDefined();

    test('a message is 1 to 100,000 characters', () => {
        ok(chatSchemas.send, { content: 'hello' });
        ok(chatSchemas.send, { content: 'a'.repeat(100000) });
        bad(chatSchemas.send, { content: '' });
        bad(chatSchemas.send, { content: 'a'.repeat(100001) });
        bad(chatSchemas.send, {});
    });

    test('regenerate and edit are checked', () => {
        ok(chatSchemas.send, { regenerate: true });
        ok(chatSchemas.send, { content: 'again', editIndex: 0 });
        bad(chatSchemas.send, { regenerate: true, content: 'x' });
        bad(chatSchemas.send, { content: 'x', editIndex: -1 });
        bad(chatSchemas.send, { content: 'x', editIndex: 200 });
        bad(chatSchemas.send, { content: 'x', role: 'system' });
    });

    test('generation options stay inside their ranges', () => {
        ok(chatSchemas.create, { options: { temperature: 0, num_ctx: 512 } });
        ok(chatSchemas.create, { options: { temperature: 2, num_ctx: 262144, top_p: 1, top_k: 500, repeat_penalty: 5, num_predict: -1 } });
        for (const options of [{ temperature: 2.1 }, { temperature: -1 }, { num_ctx: 511 }, { num_ctx: 262145 }, { top_p: 1.1 }, { top_k: 501 }, { num_predict: 100001 }, { made_up: 1 }]) {
            bad(chatSchemas.create, { options });
        }
    });

    test('titles, instructions and file paths are limited', () => {
        ok(chatSchemas.update, { title: 'Plan' });
        bad(chatSchemas.update, {});
        bad(chatSchemas.update, { title: 'a'.repeat(121) });
        bad(chatSchemas.update, { system: 'a'.repeat(100001) });
        ok(chatSchemas.attachFile, { path: '/docs/a.md' });
        for (const path of ['docs/a.md', '/docs//a.md', '', `/${'a'.repeat(1000)}`]) bad(chatSchemas.attachFile, { path });
        bad(chatSchemas.fileSave, { folder: '/docs', name: 'a/b.md', content: 'x' });
        ok(chatSchemas.fileSave, { folder: '/docs', name: 'chat.md', content: 'x' });
    });
});

describe('Chat endpoints', () => {
    const post = (path, body, config) => client.post(`/api/v1${path}`, body, config);
    const get = (path) => client.get(`/api/v1${path}`);
    const patch = (path, body) => client.patch(`/api/v1${path}`, body);
    const remove = (path) => client.delete(`/api/v1${path}`);
    const events = (response) => String(response.data).trim().split('\n').map((line) => JSON.parse(line));
    const newChat = async (body = {}) => (await post('/chats', body)).data.chat;
    const send = async (chatId, body) => {
        const response = await post(`/chats/${chatId}/messages`, body, { responseType: 'text', transformResponse: [(data) => data] });
        return { response, events: response.status === 200 ? events(response) : [] };
    };
    const upload = (chatId, name, buffer) => post(`/chats/${chatId}/attachments?name=${encodeURIComponent(name)}`, buffer, {
        headers: { 'Content-Type': 'application/octet-stream' }
    });
    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    let filesCalls = [];
    const fsFiles = {
        '/docs/brief.md': { type: 'text', content: 'The client wants a harbour survey by June.' },
        '/docs/report.pdf': { type: 'binary', bytes: makePdf('Dredging costs rose in March') },
        '/docs/photo.png': { type: 'binary', bytes: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 13, 0, 1, 2, 3]) }
    };
    const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    const fakeFilesystemOne = async (url, options) => {
        const cookie = options.headers.Cookie || '';
        filesCalls.push({ url, method: options.method || 'GET', cookie, csrf: options.headers['X-CSRF-Token'], body: options.body });
        if (state.filesDown) throw new TypeError('fetch failed');
        const address = new URL(url);
        if (address.pathname.endsWith('/directory/contents')) {
            const folder = address.searchParams.get('filePath');
            if ([...address.searchParams.keys()].join() !== 'filePath') return json(400, { success: false, message: 'Validation error' });
            if (folder === '/private') return json(404, { success: false, message: 'Directory not found or access denied' });
            return json(200, { success: true, contents: [
                { fileName: 'brief.md', filePath: '/docs/brief.md', type: 'text', size: 42 },
                { fileName: 'docs', filePath: '/docs', type: 'directory', size: 0 }
            ] });
        }
        if ((options.method || 'GET') === 'POST') {
            const body = JSON.parse(options.body);
            if (body.filePath.startsWith('/locked')) return json(403, { success: false });
            return json(201, { success: true, file: { id: 'f1', filePath: body.filePath, type: 'text' } });
        }
        const [, encoded, part] = address.pathname.match(/\/api\/v1\/files\/([^/]+)\/(content|download)$/) || [];
        const file = fsFiles[decodeURIComponent(encoded || '')];
        if (!file) return json(404, { success: false, message: 'File not found or access denied' });
        if (part === 'content') return json(200, { success: true, fileType: file.type, content: file.type === 'text' ? file.content : { type: 'Buffer', data: [] } });
        return new Response(file.bytes, { status: 200, headers: { 'Content-Length': String(file.bytes.length) } });
    };

    beforeAll(async () => {
        resetState();
        setOllamaClientFactory(fakeOllama);
        setFilesFetch(fakeFilesystemOne);
    });

    afterAll(async () => {
        state.chatHold = false;
        setFilesFetch(null);
        queue.maxConcurrent = 1;
    });

    beforeEach(async () => {
        state.chatHold = false;
        resetState();
        filesCalls = [];
        await Chat.deleteMany({});
        await Prompt.deleteMany({});
        await Setting.deleteMany({});
        await testStartup.loginAsUser('user');
    });

    describe('Creating and listing', () => {
        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/chats')).status).toBe(401);
            expect((await post('/chats', {})).status).toBeGreaterThanOrEqual(401);
        });

        test('a chat can be created with nothing chosen', async () => {
            const response = await post('/chats', {});
            expect(response.status).toBe(201);
            expect(response.data.chat).toMatchObject({ title: 'New chat', model: null, system: '', messages: [], pendingAttachments: [] });
            const stored = await Chat.findById(response.data.chat.id);
            expect(stored.service).toBe('ai-base');
        });

        test('lists only the caller’s chats, newest first, and never caches', async () => {
            const mine = await newChat({ title: 'Mine' });
            await testStartup.loginAsUser('owner');
            await newChat({ title: 'Owner only' });
            await testStartup.loginAsUser('user');
            const second = await newChat({ title: 'Second' });

            const response = await get('/chats');
            expect(response.headers['cache-control']).toMatch(/no-store/);
            expect(response.data.chats.map((chat) => chat.id)).toEqual([second.id, mine.id]);
        });

        test('searches by title', async () => {
            await newChat({ title: 'Harbour survey' });
            await newChat({ title: 'Budget (2026)' });
            expect((await get('/chats?search=harb')).data.chats.map((chat) => chat.title)).toEqual(['Harbour survey']);
            expect((await get('/chats?search=(2026)')).data.chats).toHaveLength(1);
            expect((await get('/chats?search=zzz')).data.chats).toEqual([]);
        });

        test('renames, changes instructions and deletes', async () => {
            const chat = await newChat();
            const renamed = await patch(`/chats/${chat.id}`, { title: 'Renamed', system: 'Reply formally.', options: { temperature: 0.2 } });
            expect(renamed.data.chat).toMatchObject({ title: 'Renamed', system: 'Reply formally.', options: { temperature: 0.2 } });
            expect((await remove(`/chats/${chat.id}`)).status).toBe(200);
            expect((await get(`/chats/${chat.id}`)).status).toBe(404);
        });

        test('rejects a malformed id before it reaches the database', async () => {
            expect((await get('/chats/not-an-id')).status).toBe(400);
        });
    });

    describe('Ownership', () => {
        test('another user cannot read, change, send to, attach to, export or delete a chat', async () => {
            await testStartup.loginAsUser('owner');
            const chat = await newChat({ title: 'Private', model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'secret plan' });

            for (const role of ['user', 'admin']) {
                await testStartup.loginAsUser(role);
                expect((await get(`/chats/${chat.id}`)).status).toBe(404);
                expect((await patch(`/chats/${chat.id}`, { title: 'Taken' })).status).toBe(404);
                expect((await send(chat.id, { content: 'hello' })).response.status).toBe(404);
                expect((await send(chat.id, { regenerate: true })).response.status).toBe(404);
                expect((await post(`/chats/${chat.id}/summary`, {})).status).toBe(404);
                expect((await upload(chat.id, 'a.txt', Buffer.from('text'))).status).toBe(404);
                expect((await get(`/chats/${chat.id}/export?format=markdown`)).status).toBe(404);
                expect((await remove(`/chats/${chat.id}`)).status).toBe(404);
            }
            const stored = await Chat.findById(chat.id);
            expect(stored.title).toBe('Private');
            expect(stored.messages).toHaveLength(2);
        });
    });

    describe('Sending a message', () => {
        test('streams the reply, saves both messages and titles the chat', async () => {
            const chat = await newChat();
            state.chatReply = 'Hello there friend';
            const { response, events: list } = await send(chat.id, { content: 'Explain what you can help me with in three sentences.' });

            expect(response.status).toBe(200);
            expect(response.headers['content-type']).toMatch(/ndjson/);
            expect(list[0]).toMatchObject({ type: 'accepted', chat: { id: chat.id } });
            expect(list.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('Hello there friend');
            const done = list[list.length - 1];
            expect(done).toMatchObject({ type: 'done', reply: { role: 'assistant', content: 'Hello there friend', stopped: false, error: null } });
            expect(done.reply.stats.tokensPerSecond).toBeGreaterThan(0);
            expect(done.context).toMatchObject({ capacityTokens: 32768, warn: false });

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.title).toBe('Explain what you can help me with in three sentences.');
            expect(saved.messages.map((message) => message.role)).toEqual(['user', 'assistant']);
            expect(saved.model).toBe('qwen2.5-coder:7b');
        });

        test('uses the default model from settings when one is set', async () => {
            await Setting.create({ defaultModel: 'custom-thing:latest' });
            const chat = await newChat();
            await send(chat.id, { content: 'hi' });
            expect(state.chatCalls[0].model).toBe('custom-thing:latest');
        });

        test('sends instructions, options, the context size and how long to stay loaded', async () => {
            await Setting.create({ keepAliveMinutes: 12 });
            const chat = await newChat({ model: 'qwen2.5-coder:7b', system: 'Reply only in formal British English.', options: { temperature: 0.3 } });
            await send(chat.id, { content: 'hi' });
            const request = state.chatCalls[0];
            expect(request.messages[0]).toEqual({ role: 'system', content: 'Reply only in formal British English.' });
            expect(request.messages[1]).toEqual({ role: 'user', content: 'hi' });
            expect(request.options).toMatchObject({ temperature: 0.3, num_ctx: 8192 });
            expect(request.keep_alive).toBe('12m');
            expect(request.think).toBeUndefined();
        });

        test('shows thinking separately for a reasoning model', async () => {
            state.thinkingModel = true;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const { events: list } = await send(chat.id, { content: 'think hard' });
            expect(state.chatCalls[0].think).toBe(true);
            expect(list.some((event) => event.type === 'thinking')).toBe(true);
            const done = list[list.length - 1];
            expect(done.reply.thinking).toBe('Let me consider this.');
            expect(done.reply.content).not.toContain('consider');
        });

        test('a later message carries the earlier conversation', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'First answer';
            await send(chat.id, { content: 'first' });
            await send(chat.id, { content: 'second' });
            expect(state.chatCalls[1].messages.map((message) => message.content)).toEqual(['first', 'First answer', 'second']);
        });

        test('regenerate replaces the last reply', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'Old answer';
            await send(chat.id, { content: 'question' });
            state.chatReply = 'New answer';
            const { events: list } = await send(chat.id, { regenerate: true });
            expect(list[list.length - 1].reply.content).toBe('New answer');

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.messages.map((message) => message.content)).toEqual(['question', 'New answer']);
            expect(state.chatCalls[1].messages.map((message) => message.content)).toEqual(['question']);
        });

        test('editing an earlier message continues from the edited message', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'one' });
            await send(chat.id, { content: 'two' });
            state.chatReply = 'Edited answer';
            await send(chat.id, { content: 'one, corrected', editIndex: 0 });

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.messages.map((message) => message.content)).toEqual(['one, corrected', 'Edited answer']);
        });

        test('only a user message can be edited', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'one' });
            expect((await send(chat.id, { content: 'x', editIndex: 1 })).response.status).toBe(400);
            expect((await send(chat.id, { content: 'x', editIndex: 9 })).response.status).toBe(400);
        });

        test('refuses a message that is too long or empty', async () => {
            const chat = await newChat();
            expect((await send(chat.id, { content: 'a'.repeat(100001) })).response.status).toBe(400);
            expect((await send(chat.id, { content: '' })).response.status).toBe(400);
            expect((await Chat.findById(chat.id)).messages).toHaveLength(0);
        });

        test('refuses the 201st message', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const stored = await Chat.findById(chat.id);
            for (let count = 0; count < 100; count++) stored.messages.push({ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' });
            await stored.save();
            const response = (await send(chat.id, { content: 'one more' })).response;
            expect(response.status).toBe(422);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'CHAT_FULL' });
        });
    });

    describe('When something goes wrong', () => {
        test('Ollama not running is said plainly and nothing is saved', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.up = false;
            const response = (await send(chat.id, { content: 'hello' })).response;
            expect(response.status).toBe(503);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'OLLAMA_DOWN' });
            expect((await Chat.findById(chat.id)).messages).toHaveLength(0);
        });

        test('no installed model is said plainly', async () => {
            state.models = [];
            const chat = await newChat();
            const response = (await send(chat.id, { content: 'hello' })).response;
            expect(response.status).toBe(409);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'NO_MODELS' });
        });

        test('a model that is not installed is said plainly', async () => {
            const chat = await newChat({ model: 'gone:1b' });
            const response = (await send(chat.id, { content: 'hello' })).response;
            expect(response.status).toBe(404);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'MODEL_MISSING' });
        });

        test('Ollama stopping mid-reply keeps what was written and allows regenerate', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'Partial answer that breaks';
            state.chatFailsMidway = true;
            const { events: list } = await send(chat.id, { content: 'hello' });
            const last = list[list.length - 1];
            expect(last).toMatchObject({ type: 'error', code: 'OLLAMA_DOWN' });
            expect(last.reply.content).toBe('Partial ');
            expect(last.reply.error.code).toBe('OLLAMA_DOWN');

            state.chatFailsMidway = false;
            state.chatReply = 'Whole answer';
            await send(chat.id, { regenerate: true });
            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.messages.map((message) => message.content)).toEqual(['hello', 'Whole answer']);
        });

        test('not enough memory is said plainly', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatError = 'model requires more system memory (9.1 GiB) than is available (4.0 GiB)';
            const { events: list } = await send(chat.id, { content: 'hello' });
            expect(list[list.length - 1]).toMatchObject({ type: 'error', code: 'OUT_OF_MEMORY' });
            expect(list[list.length - 1].action).toMatch(/unload/i);
        });

        test('a conversation too long for the model is refused before it is sent', async () => {
            state.contextLength = 2048;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const response = (await send(chat.id, { content: 'word '.repeat(2000) })).response;
            expect(response.status).toBe(422);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'TOO_LONG' });
            expect(state.chatCalls).toHaveLength(0);
        });

        test('warns as the conversation nears the limit', async () => {
            state.contextLength = 8192;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const { events: list } = await send(chat.id, { content: 'word '.repeat(Math.floor(((8192 - 1024 - 300 - estimateTokens(COMPONENT_GUIDE)) * 4) / 5)) });
            expect(list[list.length - 1].context).toMatchObject({ capacityTokens: 8192, warn: true });
            expect(state.chatCalls[0].options.num_ctx).toBe(8192);
        });
    });

    describe('Stopping and waiting', () => {
        const openStream = async (chatId, body) => {
            const controller = new AbortController();
            const cookie = client.cookieJar.getCookieStringSync(client.baseURL);
            const response = await fetch(`${client.baseURL}/api/v1/chats/${chatId}/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Cookie: cookie, 'X-CSRF-Token': client.getCsrfTokenFromCookies() },
                body: JSON.stringify(body),
                signal: controller.signal
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const seen = [];
            const pump = (async () => {
                let buffer = '';
                try {
                    for (;;) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        lines.filter(Boolean).forEach((line) => seen.push(JSON.parse(line)));
                    }
                } catch {
                    return seen;
                }
                return seen;
            })();
            const until = async (check) => {
                for (let attempt = 0; attempt < 200; attempt++) {
                    if (seen.some(check)) return;
                    await wait(15);
                }
                throw new Error(`stream never showed the expected event: ${JSON.stringify(seen)}`);
            };
            return { seen, until, stop: () => controller.abort(), finished: pump };
        };

        test('leaving the page does not stop the reply, and coming back picks it up from the start', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'One two three four';
            state.chatHold = true;
            const first = await openStream(chat.id, { content: 'long request' });
            await first.until((event) => event.type === 'token');
            first.stop();
            await first.finished;
            await wait(60);

            expect(state.chatAborted).toBe(0);
            expect((await get(`/chats/${chat.id}`)).data.chat).toMatchObject({ replying: true });
            expect((await post(`/chats/${chat.id}/messages`, { regenerate: true })).status).toBe(409);

            const cookie = client.cookieJar.getCookieStringSync(client.baseURL);
            const back = await fetch(`${client.baseURL}/api/v1/chats/${chat.id}/reply`, { headers: { Cookie: cookie } });
            expect(back.status).toBe(200);
            state.chatHold = false;
            const events = (await back.text()).split('\n').filter(Boolean).map((line) => JSON.parse(line));
            expect(events.slice(0, 2).map((event) => event.type)).toEqual(['accepted', 'start']);
            expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('One two three four');
            expect(events[events.length - 1]).toMatchObject({ type: 'done', reply: { content: 'One two three four', stopped: false } });

            const after = (await get(`/chats/${chat.id}`)).data.chat;
            expect(after.replying).toBe(false);
            expect(after.messages[1]).toMatchObject({ role: 'assistant', content: 'One two three four' });
            expect((await fetch(`${client.baseURL}/api/v1/chats/${chat.id}/reply`, { headers: { Cookie: cookie } })).status).toBe(204);
        });

        test('a reply sent live answers at once and goes out on the live connection, numbered and in order', async () => {
            const service = getNotificationService();
            const sent = [];
            const real = service.sendToUser.bind(service);
            service.sendToUser = (userId, notification) => {
                if (notification.type === 'chat:reply') sent.push({ userId: String(userId), ...notification.data });
                return real(userId, notification);
            };
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                expect((await get(`/chats/${chat.id}/live`)).data).toMatchObject({ replying: false, seq: 0, events: [] });

                const started = await post(`/chats/${chat.id}/messages`, { content: 'say something', live: true });
                expect(started.status).toBe(202);
                expect(started.data).toEqual({ success: true, started: true });

                for (let attempt = 0; attempt < 200 && !sent.some((piece) => piece.event.type === 'end'); attempt++) await wait(25);
                const mine = sent.filter((piece) => piece.chatId === chat.id);
                expect(mine.map((piece) => piece.seq)).toEqual(mine.map((_, index) => index + 1));
                expect(mine[0].event.type).toBe('accepted');
                expect(mine.some((piece) => piece.event.type === 'token')).toBe(true);
                expect(mine[mine.length - 1].event).toEqual({ type: 'end' });
                expect(new Set(mine.map((piece) => piece.userId)).size).toBe(1);
                expect(mine[0].userId).toBe(String((await Chat.findById(chat.id)).owner));

                for (let attempt = 0; attempt < 100 && (await get(`/chats/${chat.id}/live`)).data.replying; attempt++) await wait(25);
                const saved = (await get(`/chats/${chat.id}`)).data.chat;
                expect(saved.messages[saved.messages.length - 1].role).toBe('assistant');
                expect(saved.messages[saved.messages.length - 1].content.length).toBeGreaterThan(0);

                const refused = await post(`/chats/${chat.id}/messages`, { live: true });
                expect(refused.status).toBe(400);
            } finally {
                service.sendToUser = real;
            }
        });

        test('a change to a chat is announced to the person\'s open apps, and a read is not', async () => {
            const service = getNotificationService();
            const sent = [];
            const real = service.sendToUser.bind(service);
            service.sendToUser = (userId, notification) => {
                if (['chat:changed', 'models:changed', 'device:changed'].includes(notification.type)) sent.push({ type: notification.type, ...notification.data });
                return real(userId, notification);
            };
            const settled = async (count) => {
                for (let attempt = 0; attempt < 80 && sent.length < count; attempt++) await wait(10);
            };
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await settled(1);
                expect(sent).toEqual([{ type: 'chat:changed', id: null, removed: false }]);

                sent.length = 0;
                await get(`/chats/${chat.id}`);
                await get('/chats');
                await wait(60);
                expect(sent).toEqual([]);

                await client.patch(`/api/v1/chats/${chat.id}`, { title: 'Renamed elsewhere' });
                await settled(1);
                expect(sent).toEqual([{ type: 'chat:changed', id: chat.id, removed: false }]);

                sent.length = 0;
                expect((await client.patch(`/api/v1/chats/${chat.id}`, { title: '' })).status).toBe(400);
                await wait(60);
                expect(sent).toEqual([]);

                await send(chat.id, { content: 'hello' });
                await settled(2);
                expect(sent.filter((note) => note.type === 'chat:changed' && note.id === chat.id).length).toBeGreaterThanOrEqual(2);

                sent.length = 0;
                await client.delete(`/api/v1/chats/${chat.id}`);
                await settled(1);
                expect(sent).toEqual([{ type: 'chat:changed', id: chat.id, removed: true }]);

                sent.length = 0;
                await post('/models/load', { name: 'qwen2.5-coder:7b' });
                await wait(60);
                expect(sent.some((note) => note.type === 'models:changed')).toBe(false);
                await testStartup.loginAsUser('admin');
                await post('/models/unload', { name: 'qwen2.5-coder:7b' });
                await settled(1);
                expect(sent.some((note) => note.type === 'models:changed')).toBe(true);
            } finally {
                service.sendToUser = real;
            }
        });

        test('an app that joins a reply part-way gets what was sent so far and the number to carry on from', async () => {
            const service = getNotificationService();
            const sent = [];
            const real = service.sendToUser.bind(service);
            service.sendToUser = (userId, notification) => {
                if (notification.type === 'chat:reply') sent.push(notification.data);
                return real(userId, notification);
            };
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                state.chatReply = 'One two three four';
                state.chatHold = true;
                expect((await post(`/chats/${chat.id}/messages`, { content: 'count for me', live: true })).status).toBe(202);
                for (let attempt = 0; attempt < 200 && !sent.some((piece) => piece.chatId === chat.id && piece.event.type === 'token'); attempt++) await wait(15);

                const joined = (await get(`/chats/${chat.id}/live`)).data;
                expect(joined.replying).toBe(true);
                const before = sent.filter((piece) => piece.chatId === chat.id && piece.seq <= joined.seq);
                expect(joined.seq).toBe(before.length);
                const text = (list) => list.filter((event) => event.type === 'token').map((event) => event.text).join('');
                expect(text(joined.events)).toBe(text(before.map((piece) => piece.event)));

                state.chatHold = false;
                for (let attempt = 0; attempt < 300 && !sent.some((piece) => piece.chatId === chat.id && piece.event.type === 'end'); attempt++) await wait(15);
                const later = sent.filter((piece) => piece.chatId === chat.id && piece.seq > joined.seq);
                expect(later.map((piece) => piece.seq)).toEqual(later.map((_, index) => joined.seq + 1 + index));
                expect(text(joined.events) + text(later.map((piece) => piece.event))).toBe('One two three four');
                expect(later[later.length - 1].event).toEqual({ type: 'end' });

                const other = client.authenticatedAs;
                await testStartup.loginAsUser(other === 'admin' ? 'user' : 'admin');
                expect((await get(`/chats/${chat.id}/live`)).status).toBe(404);
                await testStartup.loginAsUser(other);
            } finally {
                state.chatHold = false;
                service.sendToUser = real;
            }
        });

        test('activity lists your own replies in progress and the downloads, and nobody else\'s replies', async () => {
            expect((await get('/activity')).data).toMatchObject({ replies: [], downloads: expect.any(Array) });
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'One two three four';
            state.chatHold = true;
            const stream = await openStream(chat.id, { content: 'a long request about ducks' });
            await stream.until((event) => event.type === 'token');
            await post(`/chats/${chat.id}/messages`, { content: 'and then geese' });

            const mine = (await get('/activity')).data;
            expect(mine.replies).toEqual([{ chatId: chat.id, title: 'a long request about ducks', model: 'qwen2.5-coder:7b', waiting: 1, phase: expect.stringMatching(/^(starting|waiting|thinking|searching|reading|writing)$/), words: expect.any(Number) }]);

            const owner = client.authenticatedAs;
            await testStartup.loginAsUser(owner === 'admin' ? 'user' : 'admin');
            expect((await get('/activity')).data.replies).toEqual([]);
            await testStartup.loginAsUser(owner || 'user');

            await post(`/chats/${chat.id}/stop`, {});
            state.chatHold = false;
            await stream.finished;
            for (let attempt = 0; attempt < 100; attempt++) {
                if ((await get('/activity')).data.replies.length === 0) break;
                await wait(15);
            }
            expect((await get('/activity')).data.replies).toEqual([]);

            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/activity')).status).toBe(401);
        });

        test('Stop ends generation and keeps what was written', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'One two three four';
            state.chatHold = true;
            const stream = await openStream(chat.id, { content: 'long request' });
            await stream.until((event) => event.type === 'token');
            expect((await post(`/chats/${chat.id}/stop`, {})).data).toMatchObject({ stopped: true });
            await stream.finished;
            expect((await post(`/chats/${chat.id}/stop`, {})).data).toMatchObject({ stopped: false });

            for (let attempt = 0; attempt < 100 && state.chatAborted === 0; attempt++) await wait(15);
            expect(state.chatAborted).toBe(1);
            for (let attempt = 0; attempt < 100; attempt++) {
                if ((await Chat.findById(chat.id)).messages.length === 2) break;
                await wait(15);
            }
            const stored = await Chat.findById(chat.id);
            expect(stored.messages[1]).toMatchObject({ role: 'assistant', content: 'One ', stopped: true });

            state.chatHold = false;
            state.chatReply = 'Fresh answer';
            const { events: list } = await send(chat.id, { regenerate: true });
            expect(list[list.length - 1].reply.content).toBe('Fresh answer');
        });

        test('a second request waits and is told its place, then answers', async () => {
            const first = await newChat({ model: 'qwen2.5-coder:7b' });
            const second = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(first.id, { content: 'first long request' });
            await one.until((event) => event.type === 'token');
            const two = await openStream(second.id, { content: 'second long request' });
            await two.until((event) => event.type === 'queued');
            expect(two.seen.find((event) => event.type === 'queued')).toEqual({ type: 'queued', ahead: 1 });
            expect(two.seen.some((event) => event.type === 'start')).toBe(false);

            state.chatHold = false;
            const [firstEvents, secondEvents] = await Promise.all([one.finished, two.finished]);
            expect(firstEvents[firstEvents.length - 1].type).toBe('done');
            expect(secondEvents[secondEvents.length - 1].type).toBe('done');
            expect(secondEvents.findIndex((event) => event.type === 'start')).toBeGreaterThan(secondEvents.findIndex((event) => event.type === 'queued'));
        });

        test('a waiting request that is stopped never reaches the model', async () => {
            const first = await newChat({ model: 'qwen2.5-coder:7b' });
            const second = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(first.id, { content: 'first' });
            await one.until((event) => event.type === 'token');
            const two = await openStream(second.id, { content: 'second' });
            await two.until((event) => event.type === 'queued');
            await post(`/chats/${second.id}/stop`, {});
            await two.finished;
            await wait(50);

            state.chatHold = false;
            await one.finished;
            await wait(50);
            expect(state.chatCalls).toHaveLength(1);
            expect((await Chat.findById(second.id)).messages.map((message) => message.role)).toEqual(['user']);
        });

        test('a message sent while the chat is replying waits and is answered next', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'Answer';
            state.chatHold = true;
            const one = await openStream(chat.id, { content: 'first' });
            await one.until((event) => event.type === 'token');
            const queued = await post(`/chats/${chat.id}/messages`, { content: 'second' });
            expect(queued.status).toBe(202);
            expect(queued.data.queued).toEqual(['second']);
            expect((await get(`/chats/${chat.id}`)).data.chat.queued).toEqual(['second']);
            expect((await post(`/chats/${chat.id}/messages`, { regenerate: true })).status).toBe(409);

            state.chatHold = false;
            await one.finished;
            for (let attempt = 0; attempt < 200; attempt++) {
                if ((await Chat.findById(chat.id)).messages.length === 4) break;
                await wait(15);
            }
            const stored = await Chat.findById(chat.id);
            expect(stored.messages.map((message) => message.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
            expect(stored.messages[2].content).toBe('second');
            expect(stored.messages[3].content).toBe('Answer');
            for (let attempt = 0; attempt < 100; attempt++) {
                if (!(await get(`/chats/${chat.id}`)).data.chat.replying) break;
                await wait(15);
            }
            expect((await get(`/chats/${chat.id}`)).data.chat).toMatchObject({ replying: false, queued: [] });
        });

        test('only five messages can wait in one chat, and Stop drops the ones waiting', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(chat.id, { content: 'first' });
            await one.until((event) => event.type === 'token');
            for (let index = 0; index < 5; index++) {
                expect((await post(`/chats/${chat.id}/messages`, { content: `waiting ${index}` })).status).toBe(202);
            }
            const full = await post(`/chats/${chat.id}/messages`, { content: 'one too many' });
            expect(full.status).toBe(429);
            expect(full.data).toMatchObject({ code: 'QUEUE_FULL' });

            await post(`/chats/${chat.id}/stop`, {});
            await one.finished;
            state.chatHold = false;
            await wait(150);
            expect(state.chatCalls).toHaveLength(1);
            expect((await Chat.findById(chat.id)).messages.map((message) => message.role)).toEqual(['user', 'assistant']);
            expect((await get(`/chats/${chat.id}`)).data.chat).toMatchObject({ replying: false, queued: [] });
        });

        test('the number of generations at once follows the setting', async () => {
            await Setting.create({ maxConcurrentGenerations: 2 });
            const first = await newChat({ model: 'qwen2.5-coder:7b' });
            const second = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(first.id, { content: 'first' });
            const two = await openStream(second.id, { content: 'second' });
            await one.until((event) => event.type === 'token');
            await two.until((event) => event.type === 'token');
            expect(two.seen.some((event) => event.type === 'queued')).toBe(false);
            state.chatHold = false;
            await Promise.all([one.finished, two.finished]);
        });
    });

    describe('Attachments', () => {
        test.each([
            ['notes.txt', () => Buffer.from('The harbour is twelve metres deep.'), 'twelve metres'],
            ['figures.csv', () => Buffer.from('port,depth\nHarwich,12'), 'Harwich,12'],
            ['report.pdf', () => makePdf('Dredging costs rose in March'), 'Dredging costs rose'],
            ['minutes.docx', () => makeDocx('The board approved the survey'), 'board approved']
        ])('%s is read and sent with the next message only', async (name, build, expected) => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const attached = await upload(chat.id, name, await build());
            expect(attached.status).toBe(201);
            expect(attached.data.chat.pendingAttachments).toHaveLength(1);
            expect(attached.data.chat.pendingAttachments[0]).toMatchObject({ name, source: 'ai-base' });
            expect(attached.data.chat.pendingAttachments[0].text).toBeUndefined();
            expect(attached.data.chat.context.usedTokens).toBeGreaterThan(0);

            await send(chat.id, { content: 'Summarise this.' });
            expect(state.chatCalls[0].messages[0].content).toContain(expected);
            expect(state.chatCalls[0].messages[0].content).toContain(`Document "${name}"`);

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.pendingAttachments).toEqual([]);
            expect(saved.messages[0].attachments[0].name).toBe(name);
            expect(saved.messages[0].content).toBe('Summarise this.');

            await send(chat.id, { content: 'And again?' });
            expect(state.chatCalls[1].messages[0].content).toContain(expected);
            expect(state.chatCalls[1].messages[2].content).toBe('And again?');
        });

        test('a file that is not text is refused in plain words', async () => {
            const chat = await newChat();
            const response = await upload(chat.id, 'photo.png', Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 13]));
            expect(response.status).toBe(415);
            expect(response.data).toMatchObject({ success: false, code: 'FILE_UNREADABLE' });
        });

        test('a file over 20 MB is refused', async () => {
            const chat = await newChat();
            const response = await upload(chat.id, 'huge.txt', Buffer.alloc(MAX_ATTACHMENT_BYTES + 1024, 'a'));
            expect(response.status).toBe(413);
            expect(response.data).toMatchObject({ code: 'FILE_TOO_LARGE' });
            expect((await Chat.findById(chat.id)).attachments).toHaveLength(0);
        }, 30000);

        test('a document longer than the model can read is refused with the numbers', async () => {
            state.contextLength = 4096;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const response = await upload(chat.id, 'long.txt', Buffer.from('word '.repeat(8000)));
            expect(response.status).toBe(422);
            expect(response.data).toMatchObject({ code: 'DOCUMENT_TOO_LONG', neededTokens: 10000 });
            expect(response.data.message).toMatch(/10,000 tokens/);
            expect(response.data.action).toMatch(/shorter document|read more/);
            expect((await Chat.findById(chat.id)).attachments).toHaveLength(0);
        });

        test('needs a file name', async () => {
            const chat = await newChat();
            const response = await post(`/chats/${chat.id}/attachments`, Buffer.from('text'), { headers: { 'Content-Type': 'application/octet-stream' } });
            expect(response.status).toBe(400);
        });

        test('an attachment can be removed before it is sent, not after', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const attached = (await upload(chat.id, 'a.txt', Buffer.from('alpha'))).data.chat.pendingAttachments[0];
            expect((await remove(`/chats/${chat.id}/attachments/${attached.id}`)).data.chat.pendingAttachments).toEqual([]);

            const kept = (await upload(chat.id, 'b.txt', Buffer.from('beta'))).data.chat.pendingAttachments[0];
            await send(chat.id, { content: 'use it' });
            expect((await remove(`/chats/${chat.id}/attachments/${kept.id}`)).status).toBe(404);
        });

        test('attachments go when the chat is deleted', async () => {
            const chat = await newChat();
            await upload(chat.id, 'a.txt', Buffer.from('alpha'));
            await remove(`/chats/${chat.id}`);
            expect(await Chat.countDocuments({ 'attachments.name': 'a.txt' })).toBe(0);
        });
    });

    describe('Export and summary', () => {
        test('exports as markdown and as JSONL', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', title: 'Harbour: plan/2026', system: 'Be brief.' });
            state.chatReply = 'It is deep.';
            await send(chat.id, { content: 'How deep is it?' });

            const markdown = (await get(`/chats/${chat.id}/export?format=markdown`)).data;
            expect(markdown.fileName).toBe('harbour-plan-2026.md');
            expect(markdown.content).toContain('# Harbour: plan/2026');
            expect(markdown.content).toContain('## You\n\nHow deep is it?');
            expect(markdown.content).toContain('It is deep.');

            const jsonl = (await get(`/chats/${chat.id}/export?format=jsonl`)).data;
            expect(jsonl.fileName).toBe('harbour-plan-2026.jsonl');
            const lines = jsonl.content.trim().split('\n').map((line) => JSON.parse(line));
            expect(lines.map((line) => line.role)).toEqual(['system', 'user', 'assistant']);
            expect(lines[2]).toMatchObject({ content: 'It is deep.', model: 'qwen2.5-coder:7b' });

            expect((await get(`/chats/${chat.id}/export?format=pdf`)).status).toBe(400);
        });

        test('starts a new chat from a summary', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', system: 'Be brief.' });
            await send(chat.id, { content: 'We plan a harbour survey.' });
            state.chatReply = '- Goal: harbour survey';
            const response = await post(`/chats/${chat.id}/summary`, {});

            expect(response.status).toBe(201);
            expect(response.data.chat).toMatchObject({
                summary: '- Goal: harbour survey', continuedFrom: chat.id, model: 'qwen2.5-coder:7b', system: 'Be brief.', messages: []
            });
            expect(response.data.chat.title).toMatch(/\(continued\)$/);
            expect(state.chatCalls[1].stream).toBe(false);

            await send(response.data.chat.id, { content: 'What was the goal?' });
            expect(state.chatCalls[2].messages[0].content).toContain('Summary of the earlier conversation:\n- Goal: harbour survey');
        });

        test('an empty chat cannot be summarised', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect((await post(`/chats/${chat.id}/summary`, {})).status).toBe(400);
        });
    });

    describe('Built-in component guide', () => {
        test('explains the tag format, every component and its attributes', () => {
            expect(COMPONENT_GUIDE.split('\n')[0]).toMatch(/Answer the user.s message directly in normal Markdown/);
            expect(COMPONENT_GUIDE).toMatch(/Most replies need none/);
            expect(COMPONENT_GUIDE).toMatch(/Never talk about components/);
            expect(COMPONENT_GUIDE).toMatch(/Do not wrap ordinary text in a Card/);
            expect(COMPONENT_GUIDE).toMatch(/Do not make up rows/);
            expect(COMPONENT_GUIDE).toMatch(/A code fence is only for code/);
            expect(COMPONENT_GUIDE).not.toContain(TOUCH_GUIDE);
            expect(TOUCH_GUIDE).toMatch(/For this one reply only/);
            expect(COMPONENT_GUIDE).toMatch(/never repeat, summarise or explain it to the user/);
            expect(CLOSING_GUIDE).toMatch(/^REMEMBER\n/);
            expect(CLOSING_GUIDE).toMatch(/Never list, describe or explain components/);
            expect(COMPONENT_GUIDE.startsWith(BASE_GUIDE)).toBe(true);
            expect(PLAN_GUIDE).toMatch(/Start with a verb ending in -ing/);
            expect(RESULTS_GUIDE).toMatch(/the source.s own words or your account of it/);
            expect(RESULTS_GUIDE).toMatch(/never a bare list of links/);
            expect(RESULTS_GUIDE).toMatch(/match the length to the question/);
            expect(RESULTS_GUIDE).toMatch(/Do not add a list of sources or links at the end, unless the user asked/);
            expect(RESULTS_GUIDE).toMatch(/never describe the search or your own reasoning/);
            expect(PLAN_GUIDE).toMatch(/Never search for today's date, the time, how long until or since a date, a sum or a unit conversion/);
            expect(RESULTS_GUIDE).toMatch(/no more than the content needs/);
            expect(RESULTS_GUIDE).toMatch(/Then describe the work properly in your own words/);
            expect(RESULTS_GUIDE).toMatch(/Never write a quotation, a wording or a figure from memory/);
            expect(RESULTS_GUIDE).toMatch(/give the pages as links, say what each one is/);
            expect(RESULTS_GUIDE).toMatch(/never say that nothing was found/);
            expect(RESULTS_GUIDE).toMatch(/\[title\]\(address\)/);
            expect(RESULTS_GUIDE).toMatch(/A whole published work[^.]*is not written out/);
            expect(RESULTS_GUIDE).toMatch(/that it is copyrighted[^.]*give the direct source/);
            expect(COMPONENT_GUIDE).toMatch(/Do not write out a whole copyrighted work/);
            for (const heading of ['VALUES', 'COMPONENTS', 'SHARED ATTRIBUTES', 'LAYOUT ATTRIBUTES (Container and Card)', 'TAGS THAT GO INSIDE OTHERS', 'FILES BY TYPE']) {
                expect(COMPONENT_GUIDE).toContain(`\n${heading}\n|`);
            }
            expect(COMPONENT_GUIDE).toMatch(/Never put code inside a component block/);
            expect(COMPONENT_GUIDE).toContain('```component\n<Card ');
            for (const name of ['Container', 'Card', 'Typography', 'Button', 'ButtonGroup', 'FloatingActionButton', 'Badge', 'Divider', 'Icon', 'ProgressBar', 'CircularProgress', 'Data', 'Flow', 'TreeView', 'Image', 'Video', 'Audio', 'PdfViewer', 'Model3D', 'Editor', 'Input', 'Select', 'Switch']) {
                expect(COMPONENT_GUIDE).toContain(`<${name}`);
            }
            for (const part of ['<Row ', '<Column ', '<Node ', '<Edge ', '<Item ', '<Option ']) expect(COMPONENT_GUIDE).toContain(part);
            for (const word of ['justify', 'align', 'flex-column', 'grid', 'between', 'source', 'target', 'showPercentage', 'borderColor']) {
                expect(COMPONENT_GUIDE).toContain(word);
            }
            expect(COMPONENT_GUIDE).not.toMatch(/\{"component"/);
            expect(estimateTokens(COMPONENT_GUIDE)).toBeLessThan(3400);
            expect(COMPONENT_GUIDE).not.toMatch(/web_search|web_fetch|search the web|do not guess|Never invent/i);
            for (const tag of ['<Model3D src=', '<PdfViewer src=', '<Video src=', '<Audio src=', '<Editor src=', '<Data src=', '<Image src=']) expect(COMPONENT_GUIDE).toContain(tag);

            const examples = [...COMPONENT_GUIDE.matchAll(/```component\n([\s\S]*?)\n```/g)].map((match) => match[1]);
            expect(examples).toHaveLength(5);
            for (const example of examples) {
                const opened = [...example.matchAll(/<([A-Z]\w*)(?:\s[^<>]*?)?(\/?)>/g)].filter((match) => match[2] !== '/').map((match) => match[1]);
                const closed = [...example.matchAll(/<\/([A-Z]\w*)>/g)].map((match) => match[1]);
                expect(opened.sort()).toEqual(closed.sort());
            }
        });

        test('is sent with every chat and every model, with or without instructions', async () => {
            const plain = await newChat({ model: 'qwen2.5-coder:7b' });
            const guided = await newChat({ model: 'custom-thing:latest', instructions: [{ name: 'Tone', text: 'Be formal.' }] });
            await send(plain.id, { content: 'one' });
            await send(guided.id, { content: 'two' });
            await send(plain.id, { regenerate: true });

            expect(state.guidesSeen).toBe(3);
            expect(state.chatCalls[0].messages).toEqual([{ role: 'user', content: 'one' }]);
            expect(state.chatCalls[1].messages[0]).toEqual({ role: 'system', content: 'Be formal.' });
        });

        test('is never shown in the chat, the list or an export', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [{ name: 'Tone', text: 'Be formal.' }] });
            await send(chat.id, { content: 'hello' });
            const marker = 'draw interface components';
            for (const response of [
                await get(`/chats/${chat.id}`),
                await get('/chats'),
                await get(`/chats/${chat.id}/export?format=markdown`),
                await get(`/chats/${chat.id}/export?format=jsonl`)
            ]) {
                expect(JSON.stringify(response.data)).not.toContain(marker);
            }
            expect((await Chat.findById(chat.id)).system).toBe('');
        });

        test('counts toward what the model can read', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect(chat.context.usedTokens).toBeGreaterThanOrEqual(estimateTokens(COMPONENT_GUIDE));
        });
    });

    describe('Instructions made of several articles', () => {
        const tone = { name: 'Tone', text: 'Reply only in formal British English.' };
        const brief = { name: 'brief.md', text: 'The client wants a harbour survey by June.', source: 'fs-one', path: '/docs/brief.md' };

        test('a chat keeps typed text and files as separate articles and sends them all', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [tone, brief] });
            expect(chat.instructions).toEqual([
                { ...tone, source: 'ai-base', path: null },
                brief
            ]);

            await send(chat.id, { content: 'hi' });
            expect(state.chatCalls[0].messages[0]).toEqual({
                role: 'system',
                content: '# Tone\n\nReply only in formal British English.\n\n# brief.md\n\nThe client wants a harbour survey by June.'
            });
        });

        test('a single article is sent as it is, without a heading', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [tone] });
            await send(chat.id, { content: 'hi' });
            expect(state.chatCalls[0].messages[0].content).toBe('Reply only in formal British English.');
        });

        test('articles can be replaced and cleared', async () => {
            const chat = await newChat({ instructions: [tone, brief] });
            const fewer = await patch(`/chats/${chat.id}`, { instructions: [brief] });
            expect(fewer.data.chat.instructions).toEqual([brief]);
            const none = await patch(`/chats/${chat.id}`, { instructions: [] });
            expect(none.data.chat.instructions).toEqual([]);
            expect(none.data.chat.system).toBe('');
        });

        test('a chat made with the older single instruction shows it as one article', async () => {
            const chat = await newChat({ system: 'Be brief.' });
            expect(chat.instructions).toEqual([{ name: 'Instructions', text: 'Be brief.', source: 'ai-base', path: null }]);
            const changed = await patch(`/chats/${chat.id}`, { instructions: [tone] });
            expect(changed.data.chat.system).toBe(tone.text);
            expect((await Chat.findById(chat.id)).system).toBe('');
        });

        test('articles count toward what the model can read', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [{ name: 'Long', text: 'word '.repeat(4000) }] });
            expect(chat.context.usedTokens).toBeGreaterThanOrEqual(5000);
        });

        test('limits: ten articles, each named and not empty', async () => {
            const eleven = Array.from({ length: 11 }, (_, index) => ({ name: `A${index}`, text: 'x' }));
            expect((await post('/chats', { instructions: eleven })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: '', text: 'x' }] })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: 'A', text: '' }] })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: 'A', text: 'a'.repeat(100001) }] })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: 'A', text: 'x', source: 'elsewhere' }] })).status).toBe(400);
        });

        test('a new chat from a summary and an export both carry the articles', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [tone, brief] });
            await send(chat.id, { content: 'hello' });
            const exported = (await get(`/chats/${chat.id}/export?format=markdown`)).data.content;
            expect(exported).toContain('# Tone');
            expect(exported).toContain('harbour survey by June');
            const next = (await post(`/chats/${chat.id}/summary`, {})).data.chat;
            expect(next.instructions.map((item) => item.name)).toEqual(['Tone', 'brief.md']);
        });
    });

    describe('Setup guide', () => {
        const put = (body) => client.put('/api/v1/guide', body);

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/guide')).status).toBe(401);
        });

        test('is on for a first visit, remembers used buttons, and turns itself off when all are used', async () => {
            await Guide.deleteMany({});
            await testStartup.loginAsUser('admin');
            const first = (await get('/guide')).data.guide;
            expect(first).toMatchObject({ on: true, seen: [] });
            expect(first.steps).toEqual(GUIDE_STEPS);
            expect((await Guide.findOne({})).service).toBe('ai-base');

            expect((await post('/guide/seen', { key: 'not.a.button' })).status).toBe(400);
            const once = (await post('/guide/seen', { key: 'chat.send' })).data.guide;
            expect(once).toMatchObject({ on: true, seen: ['chat.send'] });
            expect((await post('/guide/seen', { key: 'chat.send' })).data.guide.seen).toEqual(['chat.send']);
            expect((await get('/guide')).data.guide.seen).toEqual(['chat.send']);

            await testStartup.loginAsUser('user');
            await testStartup.loginAsUser('admin');
            expect((await get('/guide')).data.guide).toMatchObject({ on: true, seen: ['chat.send'] });

            let last = null;
            for (const key of GUIDE_STEPS) last = (await post('/guide/seen', { key })).data.guide;
            expect(last.on).toBe(false);
            expect((await post('/guide/seen', { key: 'chat.new' })).data.guide.on).toBe(false);

            const again = (await put({ on: true })).data.guide;
            expect(again).toMatchObject({ on: true, seen: [] });
            expect((await put({ on: false })).data.guide.on).toBe(false);
            expect((await put({})).status).toBe(400);
            expect(await Guide.countDocuments({})).toBe(1);
        });

        test('each person has their own guide, and a regular user is not asked for manager-only buttons', async () => {
            await Guide.deleteMany({});
            await testStartup.loginAsUser('admin');
            await post('/guide/seen', { key: 'chat.send' });
            await testStartup.loginAsUser('user');
            const mine = (await get('/guide')).data.guide;
            expect(mine).toMatchObject({ on: true, seen: [] });
            expect(mine.steps).toEqual(GUIDE_STEPS.filter((step) => !MANAGER_GUIDE_STEPS.includes(step)));
            let last = null;
            for (const key of mine.steps) last = (await post('/guide/seen', { key })).data.guide;
            expect(last.on).toBe(false);
            expect(await Guide.countDocuments({})).toBe(2);
        });
    });

    describe('Saved instructions', () => {
        const tone = { name: 'Formal tone', text: 'Reply only in formal British English.' };

        test('validation', () => {
            expect(promptSchemas.create.validate(tone).error).toBeUndefined();
            expect(promptSchemas.create.validate({ name: 'x' }).error).toBeDefined();
            expect(promptSchemas.create.validate({ ...tone, name: 'a'.repeat(121) }).error).toBeDefined();
            expect(promptSchemas.update.validate({}).error).toBeDefined();
        });

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/prompts')).status).toBe(401);
        });

        test('an instruction can be saved, listed, renamed and deleted', async () => {
            const saved = await post('/prompts', tone);
            expect(saved.status).toBe(201);
            expect(saved.data.prompt).toMatchObject({ ...tone, source: 'ai-base', path: null });
            expect((await Prompt.findById(saved.data.prompt.id)).service).toBe('ai-base');

            await post('/prompts', { name: 'A brief', text: 'Survey by June.', source: 'fs-one', path: '/docs/brief.md' });
            const listed = (await get('/prompts')).data.prompts;
            expect(listed.map((prompt) => prompt.name)).toEqual(['A brief', 'Formal tone']);
            expect(listed[0]).toMatchObject({ source: 'fs-one', path: '/docs/brief.md' });

            const renamed = await patch(`/prompts/${saved.data.prompt.id}`, { name: 'Formal' });
            expect(renamed.data.prompt.name).toBe('Formal');
            expect((await remove(`/prompts/${saved.data.prompt.id}`)).status).toBe(200);
            expect((await get('/prompts')).data.prompts).toHaveLength(1);
        });

        test('a saved instruction carries into any number of chats', async () => {
            const saved = (await post('/prompts', tone)).data.prompt;
            const article = { name: saved.name, text: saved.text };
            const first = await newChat({ model: 'qwen2.5-coder:7b', instructions: [article] });
            const second = await newChat({ model: 'qwen2.5-coder:7b', instructions: [article] });
            await send(first.id, { content: 'one' });
            await send(second.id, { content: 'two' });
            expect(state.chatCalls.map((call) => call.messages[0].content)).toEqual([tone.text, tone.text]);

            await remove(`/prompts/${saved.id}`);
            expect((await get(`/chats/${first.id}`)).data.chat.instructions).toHaveLength(1);
        });

        test('saved instructions belong to their owner', async () => {
            const saved = (await post('/prompts', tone)).data.prompt;
            for (const role of ['owner', 'admin']) {
                await testStartup.loginAsUser(role);
                expect((await get('/prompts')).data.prompts).toEqual([]);
                expect((await patch(`/prompts/${saved.id}`, { name: 'Taken' })).status).toBe(404);
                expect((await remove(`/prompts/${saved.id}`)).status).toBe(404);
            }
            expect((await Prompt.findById(saved.id)).name).toBe('Formal tone');
            expect((await get('/prompts/not-an-id')).status).toBeGreaterThanOrEqual(400);
        });

        test('stops at 100 saved instructions', async () => {
            const owner = (await post('/prompts', tone)).data.prompt;
            const stored = await Prompt.findById(owner.id);
            await Prompt.insertMany(Array.from({ length: 99 }, (_, index) => ({ owner: stored.owner, name: `P${index}`, text: 'x' })));
            const response = await post('/prompts', tone);
            expect(response.status).toBe(422);
            expect(response.data.message).toMatch(/100 saved instructions/);
        });
    });

    describe('Loading a model ahead of the first message', () => {
        test('loads the model the next chat will use and keeps it for the set time', async () => {
            await Setting.create({ keepAliveMinutes: 20 });
            const response = await post('/models/load', {});
            expect(response.status).toBe(202);
            expect(response.data).toMatchObject({ model: 'qwen2.5-coder:7b', ready: false, loading: true });
            await wait(30);
            expect(state.unloaded).toEqual([{ model: 'qwen2.5-coder:7b', keep_alive: '20m', num_ctx: 8192 }]);
        });

        test('loads with the same context size a chat will use, so the first message does not load it again', async () => {
            process.env.TAVILY_API_KEY = 'tvly-test';
            try {
                await post('/models/load', {});
                await wait(30);
                expect(state.unloaded[0].num_ctx).toBe(16384);
                const chat = (await post('/chats', { model: 'qwen2.5-coder:7b' })).data.chat;
                await client.post(`/api/v1/chats/${chat.id}/messages`, { content: 'hi' });
                expect(state.chatCalls[state.chatCalls.length - 1].options.num_ctx).toBe(16384);
            } finally {
                process.env.TAVILY_API_KEY = '';
            }
        });

        test('loads a named model', async () => {
            const response = await post('/models/load', { name: 'custom-thing:latest' });
            expect(response.data).toMatchObject({ model: 'custom-thing:latest', loading: true });
        });

        test('does nothing when the model is already in memory', async () => {
            state.running = [{ name: 'qwen2.5-coder:7b', size: 100, size_vram: 100 }];
            const response = await post('/models/load', { name: 'qwen2.5-coder:7b' });
            expect(response.data).toMatchObject({ ready: true, loading: false });
            await wait(30);
            expect(state.unloaded).toEqual([]);
        });

        test('says plainly when Ollama is not running or nothing is installed', async () => {
            state.models = [];
            expect((await post('/models/load', {})).data).toMatchObject({ code: 'NO_MODELS' });
            state.up = false;
            expect((await post('/models/load', {})).status).toBe(503);
        });

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await post('/models/load', {})).status).toBeGreaterThanOrEqual(401);
        });
    });

    describe('Searching the web', () => {
        const withKey = async (work) => {
            process.env.TAVILY_API_KEY = 'tvly-test';
            process.env.WEB_SEARCH_MODE = 'tools';
            try {
                return await work();
            } finally {
                process.env.TAVILY_API_KEY = '';
                process.env.WEB_SEARCH_MODE = '';
            }
        };

        beforeAll(() => {
            setSearchFetch(async (url, options = {}) => {
                state.searchUrls.push(String(url));
                const sent = options.body ? JSON.parse(options.body) : null;
                const asked = sent ? sent.query : new URL(url).searchParams.get('q');
                state.searchRequests.push({ method: options.method || 'GET', key: options.headers?.Authorization || null, body: sent });
                if (state.searchFails && !state.searchStatus) throw new Error('ECONNREFUSED');
                if (state.searchFails) return { ok: false, status: state.searchStatus };
                if (state.enginesLimited) return { ok: false, status: 432 };
                state.searches.push(asked);
                const first = state.freshResults ? `https://fresh.example/${state.searches.length}/files/duck.glb` : 'https://models.example/duck';
                const results = Array.from({ length: 8 }, (_, index) => ({ title: index ? `Other ${index}` : 'Duck model', url: index ? `https://other.example/${index}` : first, content: `A duck in glTF format. ${asked}`, engine: 'x' }));
                if (state.junkResults) results.push({ title: 'Free online games', url: 'https://games.example/', content: 'Play free games now.', engine: 'x' }, { title: 'Duck pond model railway', url: 'https://railway.example/duck', content: 'A model of a duck pond.', engine: 'x' });
                if (state.junkResults) results.splice(0, 8);
                return { ok: true, status: 200, json: async () => ({ results }) };
            });
            setRemoteTransport({
                lookup: async (host) => [{ address: host === 'intranet.example' ? '10.0.0.5' : '93.184.216.34', family: 4 }],
                request: async (url) => {
                    state.fetches.push(url.href);
                    if (url.href.endsWith('/files/duck.glb')) return { status: 200, headers: { 'content-type': 'model/gltf-binary' }, stream: Readable.from([Buffer.from('glb')]) };
                    if (url.href.includes('missing')) return { status: 404, headers: {}, stream: Readable.from([]) };
                    if (url.href === 'https://raw.githubusercontent.com/owner/repo/main/data.csv') return { status: 200, headers: { 'content-type': 'text/plain' }, stream: Readable.from([Buffer.from('a,b')]) };
                    if (url.href.endsWith('/icon.png')) return { status: 200, headers: { 'content-type': 'image/png', 'content-length': '900' }, stream: Readable.from([Buffer.from('png')]) };
                    if (url.host === 'other.example') return { status: 200, headers: { 'content-type': 'text/html' }, stream: Readable.from([Buffer.from('<html><body><a href="https://models.example/files/missing.glb">a</a><a href="https://models.example/files/page.glb">b</a><img src="/icon.png"><a href="https://github.com/owner/repo/blob/main/data.csv">c</a><script>window.data = {"file":"https:\\/\\/models.example\\/hidden\\/files\\/duck.glb"}</script></body></html>')]) };
                    return { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' }, stream: Readable.from([Buffer.from("<html><head><title>Duck &amp; friends</title><style>p{color:red}</style></head><body><p>Download the duck below.</p><a href=\"/files/duck.glb\">file</a> <a href=\"https://models.example/about\">about</a><script>evil()</script></body></html>")]) };
                }
            });
        });

        afterAll(() => {
            setSearchFetch(null);
            setRemoteTransport(null);
        });
        const search = ['web_search', { query: 'duck 3d model' }];
        const fetchPage = ['web_fetch', { url: 'https://models.example/duck' }];

        test('is off without a search address: no tools are offered, the page is told so, and the model gets no extra instruction', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'find a duck model' });
            expect(state.chatCalls[0].tools).toBeUndefined();
            expect(state.searchGuides).toBe(0);
            expect(COMPONENT_GUIDE).not.toMatch(/web_search/);
            expect(state.chatCalls[0].options.num_ctx).toBe(8192);
            expect((await get('/models')).data.webSearch).toBe(false);
        });

        test('with a search address, both tools are offered with more room to read and the instruction to search with every request', async () => {
            await withKey(async () => {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                const request = state.chatCalls[0];
                expect(request.tools.map((tool) => tool.function.name)).toEqual(['web_search', 'web_fetch']);
                expect(state.searchGuides).toBe(1);
                expect(SEARCH_GUIDE).toMatch(/web_search/);
                expect(SEARCH_GUIDE).toMatch(/with every request/);
                expect(request.options.num_ctx).toBe(16384);
                expect((await get('/models')).data.webSearch).toBe(true);
            });
        });

        test('a model that cannot use tools is not offered them', async () => {
            await withKey(async () => {
                state.noTools = true;
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                expect(state.chatCalls[0].tools).toBeUndefined();
                expect(state.searchGuides).toBe(0);
            });
        });

        test('searches, reads a page, then answers from what it found', async () => {
            await withKey(async () => {
                state.toolPlan = [[search], [fetchPage]];
                state.chatReply = 'Here is the duck.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });

                expect(list.filter((event) => event.type === 'tool')).toEqual([
                    { type: 'tool', tool: 'web_search', input: 'duck 3d model' },
                    { type: 'tool', tool: 'web_fetch', input: 'https://models.example/duck' }
                ]);
                expect(state.searches).toEqual(['duck 3d model']);
                expect(state.searchUrls).toEqual(['https://api.tavily.com/search']);
                expect(state.fetches).toEqual(['https://models.example/duck']);
                expect(state.chatCalls).toHaveLength(3);

                const last = state.chatCalls[2].messages;
                const results = last.filter((message) => message.role === 'tool');
                expect(results.map((message) => message.tool_name)).toEqual(['web_search', 'web_fetch']);
                expect(results[0].content).toContain('https://models.example/duck');
                expect(JSON.parse(results[0].content)).toHaveLength(5);
                expect(JSON.parse(results[1].content)).toEqual({ title: 'Duck & friends', content: 'Download the duck below.\nfile about', links: ['https://models.example/files/duck.glb', 'https://models.example/about'] });
                expect(last.filter((message) => message.role === 'assistant' && message.tool_calls)).toHaveLength(2);

                const done = list[list.length - 1];
                expect(done).toMatchObject({ type: 'done', reply: { content: 'Here is the duck.' } });
                expect(done.reply.steps).toEqual([
                    { tool: 'web_search', input: 'duck 3d model', failed: null },
                    { tool: 'web_fetch', input: 'https://models.example/duck', failed: null }
                ]);
                expect((await get(`/chats/${chat.id}`)).data.chat.messages[1].steps).toHaveLength(2);
            });
        });

        test('accepts tool arguments written as text, and refuses to read something that is not a web address', async () => {
            await withKey(async () => {
                state.toolPlan = [[['web_search', '{"query":"duck"}'], ['web_fetch', { url: 'file:///etc/passwd' }]]];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                expect(state.searches).toEqual(['duck']);
                expect(state.fetches).toEqual([]);
                expect(state.chatCalls[1].messages.filter((message) => message.role === 'tool')[1].content).toMatch(/https/);
            });
        });

        test('a failed search is told to the model, and the reply still finishes', async () => {
            await withKey(async () => {
                state.searchFails = true;
                state.toolPlan = [[search]];
                state.chatReply = 'I could not look that up.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).toBe('Error: The search service could not be reached. (ECONNREFUSED)');
                expect(list[list.length - 1]).toMatchObject({ type: 'done', reply: { content: 'I could not look that up.', error: null } });
                expect(list.find((event) => event.type === 'toolFailed').failed).toBe('unreachable');
            });
        });

        test.each([[403, 'refused', 'The search service refused the request.'], [429, 'limited', 'The search service is limiting requests right now.']])('a search answered with %i is told to the model plainly and kept on the reply', async (status, failed, said) => {
            await withKey(async () => {
                state.searchFails = true;
                state.searchStatus = status;
                state.toolPlan = [[search]];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).toBe(`Error: ${said} (status ${status})`);
                expect(list.find((event) => event.type === 'toolFailed').failed).toBe(failed);
                expect(list[list.length - 1].reply.steps[0].failed).toBe(failed);
                expect((await get(`/chats/${chat.id}`)).data.chat.messages[1].steps[0].failed).toBe(failed);
            });
        });

        test('tool calls a model writes as text are run as calls and kept out of the reply', async () => {
            await withKey(async () => {
                state.toolsAsText = true;
                state.toolPlan = [[search, ['web_search', { query: 'second' }], ['web_search', { query: 'third' }], ['web_search', { query: 'fourth' }], ['web_search', { query: 'fifth' }]], [fetchPage]];
                state.chatReply = 'Here is the duck.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.searches).toEqual(['duck 3d model', 'second', 'third', 'fourth']);
                expect(state.fetches).toEqual(['https://models.example/duck']);
                expect(list.filter((event) => event.type === 'replace')).toEqual([{ type: 'replace', text: '' }, { type: 'replace', text: '' }]);
                const done = list[list.length - 1];
                expect(done.reply.content).toBe('Here is the duck.');
                expect(done.reply.steps).toHaveLength(5);
                const second = state.chatCalls[1].messages;
                expect(second.find((message) => message.tool_calls).tool_calls).toHaveLength(4);
                expect(second.filter((message) => message.role === 'tool')).toHaveLength(4);
            });
        });

        test('tool calls written at the end of a reply are found, and anything else is left alone', () => {
            const call = '{"name": "web_search", "arguments": {"query": "a {b} \\"c\\""}}';
            const found = { function: { name: 'web_search', arguments: { query: 'a {b} "c"' } } };
            expect(writtenToolCalls(call)).toEqual({ calls: [found], text: '' });
            expect(writtenToolCalls(`<tool_call>\n${call}\n</tool_call>`)).toEqual({ calls: [found], text: '' });
            expect(writtenToolCalls(`\`\`\`json\n[${call},\n{"name":"web_fetch","parameters":{"url":"https://a.example"}}]\n\`\`\``).calls).toHaveLength(2);
            expect(writtenToolCalls(`I will search first.\n\n${call}\n${call}`)).toEqual({ calls: [found, found], text: 'I will search first.' });
            expect(writtenToolCalls(`${call}\nThen I will answer.`).calls).toEqual([]);
            expect(writtenToolCalls('{"name": "delete_everything", "arguments": {}}').calls).toEqual([]);
            expect(writtenToolCalls('{"name": "web_search", "arguments": {"query": "cut off').calls).toEqual([]);
            expect(writtenToolCalls('Here is the duck.')).toEqual({ calls: [], text: 'Here is the duck.' });
        });

        test('will not read a page on a private network', async () => {
            await withKey(async () => {
                state.toolPlan = [[['web_fetch', { url: 'https://intranet.example/admin' }]]];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'read the intranet' });
                expect(state.fetches).toEqual([]);
                expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).toBe('Error: This address cannot be loaded.');
                expect(list.find((event) => event.type === 'toolFailed').failed).toBe('blocked');
            });
        });

        test('a decorative touch is offered only after several replies without a component', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            for (let turn = 0; turn < 5; turn++) await send(chat.id, { content: `question ${turn}` });
            const leads = state.chatCalls.map((call) => call.messages[0]);
            expect(leads.slice(0, 4).every((lead) => lead.role === 'user')).toBe(true);
            expect(leads[4]).toEqual({ role: 'system', content: TOUCH_GUIDE });
        });

        test('the skill library is whole: every skill has a name, a description, a known project and a licence that allows it', async () => {
            const { skills, sources, maxSkills } = (await get('/skills')).data;
            expect(skills.length).toBeGreaterThan(250);
            expect(new Set(skills.map((skill) => skill.id)).size).toBe(skills.length);
            for (const skill of SKILLS) {
                expect(skill.id).toMatch(/^[a-z0-9-]{1,60}$/);
                expect(skill.name.length).toBeGreaterThan(1);
                expect(SKILL_SOURCES[skill.source]).toBeTruthy();
                if (skill.source === 'mollama') continue;
                expect(skill.description.length).toBeGreaterThan(10);
                expect(skill.credit).toMatch(/\((MIT|Apache-2\.0)\)$/);
                expect(skill.text.startsWith(`SKILL: ${skill.name}`)).toBe(true);
            }
            // The five skills Mollama wrote itself are gone; the one that draws its viewers stays.
            expect(skills.filter((skill) => skill.source === 'mollama').map((skill) => skill.id)).toEqual(['components']);
            expect(sources.map((source) => source.id)).toEqual(['mollama', 'anthropic', 'engineering', 'marketing', 'medical', 'dex', 'context7', 'last30days', 'anatomy', 'hairline']);
            expect(sources.reduce((sum, source) => sum + source.count, 0)).toBe(skills.length);
            expect(skills.every((skill) => skill.tokens > 0)).toBe(true);

            // A skill written for an agent with a shell tells the model what it cannot do here; a plain one does not.
            const agent = SKILLS.find((skill) => skill.needs.includes('commands'));
            const plain = SKILLS.find((skill) => skill.source !== 'mollama' && skill.needs.length === 0);
            expect(agent.text).toMatch(/Never pretend to have run a command or read a file/);
            expect(plain.text).not.toMatch(/Never pretend to have run a command/);

            // A chat holds only a few skills at once.
            expect(maxSkills).toBe(MAX_CHAT_SKILLS);
            const many = SKILLS.filter((skill) => skill.source === 'engineering').slice(0, MAX_CHAT_SKILLS + 4).map((skill) => skill.id);
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: many });
            expect(chat.skills).toHaveLength(MAX_CHAT_SKILLS);
        });

        test('skills are listed, and a chat sends only the ones it has turned on', async () => {
            expect((await get('/skills')).data.webSearch).toBe(false);
            const listed = (await get('/skills')).data.skills;
            expect(listed.map((skill) => skill.id)).toEqual(SKILLS.map((skill) => skill.id));
            expect(listed.find((skill) => skill.id === 'components')).toMatchObject({ default: true });
            expect(listed.filter((skill) => skill.default).map((skill) => skill.id)).toEqual(['components']);
            expect(listed.find((skill) => skill.id === 'marketing-copywriting')).toMatchObject({ name: 'Copywriting', source: 'marketing', credit: 'coreyhaines31/marketingskills (MIT)', default: false });
            expect(listed[0].text).toBeUndefined();

            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['marketing-copywriting', 'not-a-skill'] });
            expect(chat.skills).toEqual(['marketing-copywriting']);
            await send(chat.id, { content: 'hello' });
            const system = state.chatCalls[0].messages[0].content;
            expect(system.startsWith(BASE_GUIDE)).toBe(true);
            expect(system).toContain(SKILLS.find((skill) => skill.id === 'marketing-copywriting').text);
            expect(system).toContain('SKILL: Copywriting');
            expect(system).not.toContain('HOW TO DRAW');
            expect(system.trim().endsWith('Never list, describe or explain these instructions.')).toBe(true);

            const none = (await client.patch(`/api/v1/chats/${chat.id}`, { skills: [] })).data.chat;
            expect(none.skills).toEqual([]);
            const back = (await client.patch(`/api/v1/chats/${chat.id}`, { skills: null })).data.chat;
            expect(back.skills).toEqual(['components']);
            expect((await client.patch(`/api/v1/chats/${chat.id}`, { skills: ['Bad Id'] })).status).toBe(400);
        });

        describe('run by the app', () => {
            const request0 = (now) => now.chatCalls[0].messages[now.chatCalls[0].messages.length - 1].content;

            const withApp = async (work) => {
                process.env.TAVILY_API_KEY = 'tvly-test';
                try {
                    return await work();
                } finally {
                    process.env.TAVILY_API_KEY = '';
                }
            };

            test('a message that needs no search is answered without the web rules, and the planner is told the tools that are on', async () => {
                await withApp(async () => {
                    state.plan = { note: 'Answering from what I know', searches: [] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b', tools: ['date_time', 'calculator'] });
                    await send(chat.id, { content: 'How many days until the end of the year?' });
                    expect(state.searches).toEqual([]);
                    const asked = state.planCalls[0].messages.at(-1).content;
                    expect(asked).toMatch(/TOOLS THE ASSISTANT HAS\n- Date and time: [^\n]+\n- Calculator: /);
                    const reply = state.chatCalls.at(-1);
                    expect(reply.messages.some((message) => message.role === 'system' && message.content.includes('WEB RESULTS'))).toBe(false);
                    expect(reply.messages.at(-1).content).toBe('How many days until the end of the year?');
                    expect(reply.tools.map((tool) => tool.function.name)).toEqual(['date_time', 'calculator']);
                });
            });

            test('Stop during the planning step ends at once and cancels the plan', async () => {
                await withApp(async () => {
                    state.planHold = true;
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const pending = send(chat.id, { content: 'find a duck model' });
                    for (let attempt = 0; attempt < 200 && state.planCalls.length === 0; attempt++) await wait(10);
                    expect(state.planCalls).toHaveLength(1);
                    const started = Date.now();
                    await post(`/chats/${chat.id}/stop`, {});
                    await pending;
                    expect(Date.now() - started).toBeLessThan(1500);
                    expect(state.planAborted).toBe(1);
                    expect(state.chatCalls).toHaveLength(0);
                    state.planHold = false;
                });
            });

            test('the model plans, the app searches and reads, and only files that load are offered', async () => {
                await withApp(async () => {
                    state.plan = { note: '  Looking for a duck   model file.  ', searches: [{ query: 'duck 3d model', fileTypes: ['.GLB'] }] };
                    state.chatReply = 'Here it is.\n\n```component\n<Model3D src="https://models.example/files/duck.glb" />\n<Image src="https://invented.example/duck.png" />\n<PdfViewer src="https://models.example/duck" />\n```';
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const { events: list } = await send(chat.id, { content: 'find a duck model, like https://given.example/a.glb' });

                    expect(state.planCalls).toHaveLength(1);
                    expect(state.planCalls[0]).toMatchObject({ stream: true, format: { type: 'object' } });
                    // The planner is told today's date, and the rules for web results are not in its prompt yet.
                    expect(state.planCalls[0].messages.at(-1).role).toBe('user');
                    expect(state.planCalls[0].messages.at(-1).content).toMatch(/fileTypes[\s\S]*TODAY\nToday is [A-Z][a-z]+,? \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)\.\n\nTHE USER'S MESSAGE\n/);
                    expect(state.planCalls[0].messages.some((message) => message.role === 'system' && message.content.includes('WEB RESULTS'))).toBe(false);
                    expect(state.searches).toEqual(['duck 3d model']);
                    expect(state.planCalls[0].format.required).toEqual(['note', 'searches']);
                    expect(list.find((event) => event.type === 'note')).toEqual({ type: 'note', text: 'Looking for a duck model file.' });
                    expect(list.find((event) => event.type === 'start')).toMatchObject({ loaded: expect.any(Boolean) });
                    expect(list.filter((event) => event.type === 'tool').map((event) => event.tool)).toEqual(['web_search', 'web_fetch', 'web_fetch', 'web_fetch']);
                    expect(request0(state).indexOf('Files found')).toBeLessThan(request0(state).indexOf('Pages (web pages'));

                    expect(state.chatCalls).toHaveLength(1);
                    const request = state.chatCalls[0];
                    expect(request.tools).toBeUndefined();
                    expect(request.options.num_ctx).toBe(16384);
                    expect(request.options.num_predict).toBe(16384);
                    expect(state.planCalls[0].options.num_ctx).toBe(16384);
                    expect(request.messages[0].content).toMatch(/WEB RESULTS/);
                    const asked = request.messages[request.messages.length - 1].content;
                    expect(asked).toMatch(/^WEB RESULTS/);
                    expect(asked).toContain('- https://models.example/files/duck.glb (glb)');
                    expect(asked).not.toContain('missing.glb');
                    expect(asked).toContain('- https://models.example/hidden/files/duck.glb (glb)');
                    expect(asked).not.toContain('page.glb');
                    expect(asked.endsWith('find a duck model, like https://given.example/a.glb')).toBe(true);

                    const done = list[list.length - 1];
                    expect(state.reviewCalls).toHaveLength(0);
                    expect(done.reply.unverified).toEqual(['https://invented.example/duck.png', 'https://models.example/duck']);
                    const saved = (await get(`/chats/${chat.id}`)).data.chat.messages;
                    expect(saved[0].content).toBe('find a duck model, like https://given.example/a.glb');
                    expect(saved[1].unverified).toEqual(['https://invented.example/duck.png', 'https://models.example/duck']);
                    expect(saved[1].steps).toHaveLength(4);
                });
            });

            test('an address the user gave is never marked, and nothing is searched when the plan is empty', async () => {
                await withApp(async () => {
                    state.chatReply = '```component\n<Model3D src="https://given.example/a.glb" />\n```';
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const { events: list } = await send(chat.id, { content: 'show https://given.example/a.glb' });
                    expect(state.searches).toEqual([]);
                    expect(state.chatCalls[0].messages[state.chatCalls[0].messages.length - 1].content).toBe('show https://given.example/a.glb');
                    expect(list[list.length - 1].reply.unverified).toEqual([]);
                    expect(list[list.length - 1].reply.steps).toEqual([]);
                });
            });

            test('when the first search finds no file, the app searches once more by file type, and small images are not offered', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'tiny icons', fileTypes: ['png'] }] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const { events: list } = await send(chat.id, { content: 'find icons' });
                    expect(state.searches).toEqual(['tiny icons', 'tiny icons filetype:png']);
                    expect(request0(state)).toContain('- none');
                    expect(list[list.length - 1].reply.steps.filter((step) => step.tool === 'web_search')).toHaveLength(2);
                });
            });

            test('a GitHub page for a file is offered as its raw file', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'github table', fileTypes: ['csv'] }] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    await send(chat.id, { content: 'find a table' });
                    expect(request0(state)).toContain('- https://raw.githubusercontent.com/owner/repo/main/data.csv (csv)');
                });
            });

            test('when the search service is limiting requests the app says so and stops searching', async () => {
                await withApp(async () => {
                    state.enginesLimited = true;
                    state.plan = { searches: [{ query: 'duck', fileTypes: ['glb'] }, { query: 'goose', fileTypes: ['glb'] }] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const { events: list } = await send(chat.id, { content: 'find birds' });
                    expect(list.filter((event) => event.type === 'toolFailed').map((event) => event.failed)).toEqual(['limited']);
                    expect(state.searchUrls).toHaveLength(1);
                    expect(request0(state)).toContain('SEARCH: duck\nThe search service is limiting requests right now, so this search could not be run.');
                    expect(list[list.length - 1]).toMatchObject({ type: 'done', reply: { error: null } });
                });
            });

            test('results that share fewer than two words with the search are dropped as unrelated', async () => {
                await withApp(async () => {
                    state.junkResults = true;
                    state.plan = { searches: [{ query: 'free duck 3d model', fileTypes: [] }] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    await send(chat.id, { content: 'find a duck' });
                    expect(request0(state)).toContain('https://railway.example/duck');
                    expect(request0(state)).not.toContain('games.example');
                    expect(state.fetches).toEqual(['https://railway.example/duck']);
                });
            });

            test('with a search key the app asks the search service with that key, and a spent allowance is reported as limited', async () => {
                process.env.TAVILY_API_KEY = 'tvly-test';
                try {
                    expect((await get('/models')).data.webSearch).toBe(true);
                    state.plan = { searches: [{ query: 'duck 3d model', fileTypes: ['glb'] }] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    await send(chat.id, { content: 'find a duck model' });
                    expect(state.searchUrls).toEqual(['https://api.tavily.com/search']);
                    expect(state.searchRequests[0]).toMatchObject({ method: 'POST', key: 'Bearer tvly-test', body: { query: 'duck 3d model', max_results: 8, search_depth: 'basic' } });
                    expect(request0(state)).toContain('- https://models.example/files/duck.glb (glb)');

                    state.searchFails = true;
                    state.searchStatus = 432;
                    const second = await send(chat.id, { content: 'again' });
                    expect(second.events.find((event) => event.type === 'toolFailed').failed).toBe('limited');
                } finally {
                    process.env.TAVILY_API_KEY = '';
                }
            });

            test('when a search finds nothing, the model is asked for deeper searches and pages, and the app stops once they find something', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'duck 3d model', fileTypes: ['pdf'] }] };
                    state.reviews = [{
                        enough: false,
                        searches: [{ query: 'mallard glb github', fileTypes: ['glb'] }, { query: 'Duck 3D Model', fileTypes: ['glb'] }],
                        pages: [
                            { url: 'https://other.example/4', fileTypes: ['csv'] },
                            { url: 'https://other.example/1', fileTypes: ['csv'] },
                            { url: 'https://never-found.example/x', fileTypes: ['glb'] }
                        ]
                    }, { enough: true, searches: [], pages: [] }];
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const { events: list } = await send(chat.id, { content: 'find a duck model and a table' });

                    expect(state.reviewCalls).toHaveLength(1);
                    expect(state.reviewCalls[0].format.properties.enough).toEqual({ type: 'boolean' });
                    expect(state.reviewCalls[0].messages.at(-1).content).toMatch(/\n\nTHE USER'S MESSAGE\nfind a duck model and a table\n\nFOUND SO FAR\n\nSEARCH: duck 3d model/);
                    expect(state.reviewCalls[0].messages.at(-1).content).toContain('SEARCHES ALREADY RUN\n- duck 3d model\n\nWORDS ALREADY SEARCHED\nduck, 3d, model');
                    expect(state.searches).toEqual(['duck 3d model', 'duck 3d model filetype:pdf', 'mallard glb github']);
                    expect(state.fetches).toContain('https://other.example/4');
                    expect(state.fetches.filter((address) => address === 'https://other.example/1')).toHaveLength(1);
                    expect(state.fetches.some((address) => address.includes('never-found'))).toBe(false);
                    expect(request0(state)).toContain('PAGE: https://other.example/4\nFiles found on this page (csv; each one was loaded successfully; use these in src):\n- https://raw.githubusercontent.com/owner/repo/main/data.csv (csv)');
                    expect(list[list.length - 1]).toMatchObject({ type: 'done', reply: { error: null } });
                });
            });

            test('the search stops when two rounds in a row find nothing new, and a review that cannot be read ends it', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'duck', fileTypes: ['pdf'] }] };
                    state.reviews = [1, 2, 3, 4].map((round) => ({ enough: false, searches: [{ query: `alpha${round} beta${round}`, fileTypes: ['pdf'] }], pages: [] }));
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    await send(chat.id, { content: 'find a duck model' });
                    expect(state.reviewCalls).toHaveLength(2);
                    expect(state.searches).toEqual(['duck', 'duck filetype:pdf', 'alpha1 beta1', 'alpha1 beta1 filetype:pdf', 'alpha2 beta2', 'alpha2 beta2 filetype:pdf']);

                    state.reviews = ['not json'];
                    await send(chat.id, { content: 'again' });
                    expect(state.reviewCalls).toHaveLength(3);
                    expect(state.searches.slice(6)).toEqual(['duck', 'duck filetype:pdf']);
                });
            });

            test('a deeper search must use new words: rewordings of earlier searches are not run', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'NASA Curiosity rover 3D model glb', fileTypes: ['pdf'] }] };
                    state.reviews = [{
                        enough: false,
                        searches: [
                            { query: 'Curiosity rover 3D model NASA glb download', fileTypes: ['glb'] },
                            { query: "NASA's official 3D model of the Curiosity rover", fileTypes: ['glb'] },
                            { query: 'Curiosity rover 3D model Sketchfab', fileTypes: ['glb'] },
                            { query: 'Mars Science Laboratory MSL spacecraft mesh', fileTypes: ['pdf'] }
                        ],
                        pages: []
                    }];
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    await send(chat.id, { content: 'find the rover' });
                    expect(state.searches.filter((query) => !query.includes('filetype:'))).toEqual(['NASA Curiosity rover 3D model glb', 'Mars Science Laboratory MSL spacecraft mesh']);
                    expect(state.reviewCalls[1].messages.at(-1).content).toContain('- Mars Science Laboratory MSL spacecraft mesh');
                });
            });

            test('a chat with web search switched off is answered without searching', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'duck facts', fileTypes: [] }] };
                    expect((await get('/skills')).data.webSearch).toBe(true);
                    state.up = false;
                    expect((await get('/models')).data).toMatchObject({ ollamaAvailable: false, webSearch: true });
                    state.up = true;
                    const chat = await newChat({ model: 'qwen2.5-coder:7b', webSearch: false });
                    expect(chat.webSearch).toBe(false);
                    await send(chat.id, { content: 'search the web for ducks' });
                    expect(state.searches).toEqual([]);
                    expect(state.planCalls).toHaveLength(0);
                    expect(state.chatCalls).toHaveLength(1);

                    expect((await client.patch(`/api/v1/chats/${chat.id}`, { webSearch: true })).data.chat.webSearch).toBe(true);
                    await send(chat.id, { content: 'search the web for ducks' });
                    expect(state.searches).toEqual(['duck facts']);
                });
            });

            test('a question that needs facts gets the text of the best page', async () => {
                await withApp(async () => {
                    state.plan = { searches: [{ query: 'duck facts', fileTypes: [] }, { query: '', fileTypes: [] }] };
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    await send(chat.id, { content: 'tell me about ducks' });
                    const asked = state.chatCalls[0].messages[state.chatCalls[0].messages.length - 1].content;
                    expect(state.fetches[0]).toBe('https://models.example/duck');
                    expect(state.fetches.length).toBe(3);
                    expect(asked).toContain('Text of https://models.example/duck:\nDownload the duck below.');
                    const saved = (await get(`/chats/${chat.id}`)).data.chat.messages.pop().content;
                    expect(saved).toMatch(/\n\nSources:\n- \[Duck model\]\(https:\/\/models\.example\/duck\)\n- \[Other 1\]\(https:\/\/other\.example\/1\)\n- \[Other 2\]\(https:\/\/other\.example\/2\)$/);

                    state.chatReply = 'It is on [this page](https://models.example/duck).';
                    await send(chat.id, { content: 'and where is that?' });
                    expect((await get(`/chats/${chat.id}`)).data.chat.messages.pop().content).toBe('It is on [this page](https://models.example/duck).');
                    expect(asked).not.toContain('Files found');
                });
            });

            test('a plan that cannot be read or a search that fails still ends in a reply', async () => {
                await withApp(async () => {
                    state.plan = 'not json';
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const first = await send(chat.id, { content: 'find a duck model' });
                    expect(first.events[first.events.length - 1]).toMatchObject({ type: 'done', reply: { content: 'A short answer', error: null } });

                    state.plan = { searches: [{ query: 'duck', fileTypes: ['glb'] }] };
                    state.searchFails = true;
                    state.searchStatus = 403;
                    const second = await send(chat.id, { content: 'try again' });
                    expect(second.events.find((event) => event.type === 'toolFailed').failed).toBe('refused');
                    expect(second.events[second.events.length - 1]).toMatchObject({ type: 'done', reply: { error: null } });
                });
            });
        });

        test('the plan and results instructions are short and say what the app does', () => {
            expect(PLAN_GUIDE).toMatch(/at most 8/);
            expect(RESULTS_GUIDE).toMatch(/Never put a page address in src/);
            expect(RESULTS_GUIDE).toMatch(/Never tell the user that you cannot search the web/);
            expect(PLAN_GUIDE).toMatch(/always plan at least one search/);
            expect(REVIEW_GUIDE).toMatch(/deeper searches/);
            expect(estimateTokens(PLAN_GUIDE) + estimateTokens(RESULTS_GUIDE) + estimateTokens(REVIEW_GUIDE)).toBeLessThan(1950);
        });

        test('stops offering tools after six rounds so a reply always ends', async () => {
            await withKey(async () => {
                state.toolPlan = Array.from({ length: 10 }, () => [search]);
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.searches).toHaveLength(6);
                expect(state.chatCalls).toHaveLength(7);
                expect(state.chatCalls[6].tools).toBeUndefined();
                expect(list[list.length - 1].type).toBe('done');
            });
        });
    });

    describe('Model picker data', () => {
        test('installed models carry a fit badge and whether they are ready now', async () => {
            state.running = [{ name: 'qwen2.5-coder:7b', size: 100, size_vram: 100 }];
            const models = (await get('/models')).data.models;
            const loaded = models.find((model) => model.name === 'qwen2.5-coder:7b');
            const idle = models.find((model) => model.name === 'custom-thing:latest');
            expect(loaded).toMatchObject({ ready: true, canChat: true });
            expect(loaded.fit).toMatchObject({ fits: expect.any(Boolean), label: expect.any(String) });
            expect(idle.ready).toBe(false);
        });
    });

    describe('Files from FilesystemOne', () => {
        test('browse passes the caller’s own sign-in through', async () => {
            const response = await get('/files/browse?path=/');
            expect(response.status).toBe(200);
            expect(response.data).toMatchObject({ path: '/', parent: null });
            expect(response.data.entries).toEqual([
                { name: 'docs', path: '/docs', type: 'folder', size: 0, updatedAt: null },
                { name: 'brief.md', path: '/docs/brief.md', type: 'file', size: 42, updatedAt: null }
            ]);

            const mine = client.cookieJar.getCookieStringSync(client.baseURL).match(/accessToken=([^;]+)/)[1];
            expect(filesCalls[0].url).toMatch(/^http:\/\/fs-one\.test\/api\/v1\/files\/directory\/contents\?filePath=%2F/);
            expect(filesCalls[0].cookie).toContain(`accessToken=${mine}`);

            await testStartup.loginAsUser('owner');
            await get('/files/browse?path=/docs');
            const theirs = client.cookieJar.getCookieStringSync(client.baseURL).match(/accessToken=([^;]+)/)[1];
            expect(theirs).not.toBe(mine);
            expect(filesCalls[1].cookie).toContain(`accessToken=${theirs}`);
        });

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/files/browse?path=/')).status).toBe(401);
            expect(filesCalls).toHaveLength(0);
        });

        test('a folder FilesystemOne refuses stays refused', async () => {
            const response = await get('/files/browse?path=/private');
            expect(response.status).toBe(404);
            expect(response.data).toMatchObject({ code: 'FILE_NOT_FOUND' });
        });

        test('a file FilesystemOne refuses stays refused', async () => {
            const response = await get('/files/text?path=/private/secret.md');
            expect(response.status).toBe(404);
            expect(response.data).toMatchObject({ code: 'FILE_NOT_FOUND' });
            const chat = await newChat();
            expect((await post(`/chats/${chat.id}/attachments`, { path: '/private/secret.md' })).status).toBe(404);
        });

        test('reads a text file with its size against the model', async () => {
            const response = await get('/files/text?path=/docs/brief.md&model=qwen2.5-coder:7b');
            expect(response.data).toMatchObject({
                name: 'brief.md', path: '/docs/brief.md', text: 'The client wants a harbour survey by June.', capacityTokens: 32768, fits: true
            });
        });

        test('attaches a text file and a PDF from FilesystemOne as copies', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect((await post(`/chats/${chat.id}/attachments`, { path: '/docs/brief.md' })).status).toBe(201);
            const second = await post(`/chats/${chat.id}/attachments`, { path: '/docs/report.pdf' });
            expect(second.data.chat.pendingAttachments.map((file) => [file.name, file.source, file.path])).toEqual([
                ['brief.md', 'fs-one', '/docs/brief.md'],
                ['report.pdf', 'fs-one', '/docs/report.pdf']
            ]);

            fsFiles['/docs/brief.md'].content = 'CHANGED AFTERWARDS';
            await send(chat.id, { content: 'Compare them.' });
            const sent = state.chatCalls[0].messages[0].content;
            expect(sent).toContain('harbour survey by June');
            expect(sent).toContain('Dredging costs rose in March');
            expect(sent).not.toContain('CHANGED AFTERWARDS');
            fsFiles['/docs/brief.md'].content = 'The client wants a harbour survey by June.';
        });

        test('an image is refused in plain words', async () => {
            const response = await get('/files/text?path=/docs/photo.png');
            expect(response.status).toBe(415);
            expect(response.data).toMatchObject({ code: 'FILE_UNREADABLE' });
        });

        test('an oversized file is refused', async () => {
            fsFiles['/docs/huge.txt'] = { type: 'binary', bytes: Buffer.alloc(MAX_ATTACHMENT_BYTES + 10, 'a') };
            const response = await get('/files/text?path=/docs/huge.txt');
            delete fsFiles['/docs/huge.txt'];
            expect(response.status).toBe(413);
            expect(response.data).toMatchObject({ code: 'FILE_TOO_LARGE' });
        }, 30000);

        test('FilesystemOne being down gives a plain message and chat still works', async () => {
            state.filesDown = true;
            for (const response of [await get('/files/browse?path=/'), await get('/files/text?path=/docs/brief.md')]) {
                expect(response.status).toBe(503);
                expect(response.data).toMatchObject({ code: 'FILES_DOWN' });
                expect(response.data.action).toMatch(/FilesystemOne/);
            }
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect((await send(chat.id, { content: 'still here?' })).response.status).toBe(200);
        });

        test('saves text into a folder with the caller’s sign-in', async () => {
            const response = await post('/files/save', { folder: '/docs', name: 'chat.md', content: '# Chat' });
            expect(response.status).toBe(201);
            expect(response.data.path).toBe('/docs/chat.md');
            const call = filesCalls[0];
            expect(call.method).toBe('POST');
            expect(JSON.parse(call.body)).toEqual({ filePath: '/docs/chat.md', content: '# Chat', onConflict: 'rename' });
            expect(call.cookie).toContain(`csrfToken=${call.csrf}`);
        });

        test('a folder that cannot be written to is said plainly', async () => {
            const response = await post('/files/save', { folder: '/locked', name: 'chat.md', content: '# Chat' });
            expect(response.status).toBe(403);
            expect(response.data).toMatchObject({ code: 'FILE_NOT_SAVED' });
        });

        test('rejects a path that is not absolute', async () => {
            expect((await get('/files/browse?path=docs')).status).toBe(400);
            expect((await get('/files/text?path=/docs//brief.md')).status).toBe(400);
            expect(filesCalls).toHaveLength(0);
        });
    });
});

describe('Files loaded from the web', () => {
    const get = (path) => client.get(`/api/v1${path}`);
    const address = (url) => `/files/remote?url=${encodeURIComponent(url)}`;
    const body = (text, headers = {}) => ({ status: 200, headers, stream: Readable.from([Buffer.from(text)]) });
    let requests = [];
    let pages = {};

    beforeAll(() => {
        setRemoteTransport({
            lookup: async (host) => {
                if (host === 'intranet.example') return [{ address: '10.0.0.5', family: 4 }];
                if (host === 'mixed.example') return [{ address: '93.184.216.34', family: 4 }, { address: '127.0.0.1', family: 4 }];
                if (host === 'gone.example') throw new Error('ENOTFOUND');
                return [{ address: '93.184.216.34', family: 4 }];
            },
            request: async (url, pinned) => {
                requests.push({ url: url.href, pinned: pinned.address });
                const page = pages[url.href];
                if (!page) return { status: 404, headers: {}, stream: Readable.from([]) };
                return typeof page === 'function' ? page() : page;
            }
        });
    });

    afterAll(() => setRemoteTransport(null));

    beforeEach(async () => {
        requests = [];
        pages = {};
        await testStartup.loginAsUser('user');
    });

    test.each([
        ['8.8.8.8', true], ['93.184.216.34', true], ['2606:4700:4700::1111', true], ['192.0.66.161', true], ['198.51.99.1', true],
        ['192.0.0.8', false], ['192.0.2.1', false], ['198.51.100.7', false], ['203.0.113.9', false],
        ['127.0.0.1', false], ['10.1.2.3', false], ['172.16.0.1', false], ['172.31.255.255', false], ['192.168.1.10', false],
        ['169.254.169.254', false], ['100.64.0.1', false], ['0.0.0.0', false], ['224.0.0.1', false],
        ['::1', false], ['fd00::1', false], ['fe80::1', false], ['::ffff:127.0.0.1', false], ['not-an-ip', false]
    ])('%s is public: %s', (ip, expected) => {
        expect(isPublicAddress(ip)).toBe(expected);
    });

    test('needs sign-in', async () => {
        client.clearCookies();
        client.authenticatedAs = null;
        expect((await get(address('https://files.example/a.obj'))).status).toBe(401);
        expect(requests).toHaveLength(0);
    });

    test('passes a public https file through', async () => {
        pages['https://files.example/house.obj'] = body('v 0 0 0\nv 1 0 0\n', { 'content-type': 'text/html' });
        const response = await get(address('https://files.example/house.obj'));
        expect(response.status).toBe(200);
        expect(String(response.data)).toContain('v 1 0 0');
        expect(response.headers['content-type']).toMatch(/octet-stream/);
        expect(response.headers['content-disposition']).toMatch(/attachment/);
        expect(response.headers['x-content-type-options']).toBe('nosniff');
        expect(requests).toEqual([{ url: 'https://files.example/house.obj', pinned: '93.184.216.34' }]);
    });

    test.each([
        'http://files.example/a.obj',
        'https://user:pass@files.example/a.obj',
        'https://127.0.0.1/a.obj',
        'https://[::1]/a.obj',
        'https://169.254.169.254/latest/meta-data',
        'https://intranet.example/a.obj',
        'https://mixed.example/a.obj'
    ])('refuses %s without contacting it', async (url) => {
        const response = await get(address(url));
        expect([400, 403]).toContain(response.status);
        expect(requests).toHaveLength(0);
    });

    test('says plainly when an address is blocked', async () => {
        const response = await get(address('https://intranet.example/a.obj'));
        expect(response.data).toMatchObject({ success: false, code: 'REMOTE_BLOCKED' });
        expect(response.data.action).toMatch(/public https/);
    });

    test('follows a redirect to a public address but not to a private one', async () => {
        pages['https://files.example/old.pdf'] = { status: 302, headers: { location: 'https://files.example/new.pdf' }, stream: Readable.from([]) };
        pages['https://files.example/new.pdf'] = body('%PDF-1.4');
        expect(String((await get(address('https://files.example/old.pdf'))).data)).toContain('%PDF-1.4');

        pages['https://files.example/trap.pdf'] = { status: 302, headers: { location: 'https://intranet.example/secret.pdf' }, stream: Readable.from([]) };
        const trapped = await get(address('https://files.example/trap.pdf'));
        expect(trapped.data).toMatchObject({ code: 'REMOTE_BLOCKED' });
        expect(requests.some((entry) => entry.url.includes('intranet'))).toBe(false);
    });

    test('stops after too many redirects', async () => {
        pages['https://files.example/loop'] = () => ({ status: 302, headers: { location: 'https://files.example/loop' }, stream: Readable.from([]) });
        const response = await get(address('https://files.example/loop'));
        expect(response.status).toBe(502);
        expect(response.data).toMatchObject({ code: 'REMOTE_FAILED' });
    });

    test('a site that is limiting requests is tried once more, then reported as busy', async () => {
        let calls = 0;
        pages['https://files.example/slow.png'] = () => (++calls === 1 ? { status: 429, headers: {}, stream: Readable.from([]) } : body('picture'));
        const first = await get(address('https://files.example/slow.png'));
        expect(first.status).toBe(200);
        expect(calls).toBe(2);

        pages['https://files.example/busy.png'] = () => ({ status: 429, headers: {}, stream: Readable.from([]) });
        const second = await get(address('https://files.example/busy.png'));
        expect(second.status).toBe(503);
        expect(second.data).toMatchObject({ code: 'REMOTE_BUSY' });
    });

    test('a missing file, an unknown host and an oversized file each get a plain message', async () => {
        const missing = await get(address('https://files.example/missing.glb'));
        expect(missing.status).toBe(404);
        expect(missing.data).toMatchObject({ code: 'REMOTE_NOT_FOUND', message: 'Nothing exists at this address.' });
        expect((await get(address('https://gone.example/a.glb'))).data).toMatchObject({ code: 'REMOTE_FAILED' });
        pages['https://files.example/huge.mp4'] = body('x', { 'content-length': String(MAX_REMOTE_BYTES + 1) });
        const huge = await get(address('https://files.example/huge.mp4'));
        expect(huge.status).toBe(413);
        expect(huge.data).toMatchObject({ code: 'REMOTE_TOO_LARGE' });
    });

    test('needs a url', async () => {
        expect((await get('/files/remote')).status).toBe(400);
        expect((await get(address('not a url'))).status).toBe(400);
    });
});

describe('Linked devices', () => {
    const TAGS = { models: [{ name: 'tiny:1b', model: 'tiny:1b', size: 1000, digest: 'abc', details: { parameter_size: '1B', quantization_level: 'Q4' }, modified_at: new Date().toISOString() }] };
    const through = (deviceId) => ({ headers: { 'X-Mollama-Device': deviceId }, validateStatus: () => true });
    const hardware = { hostname: 'laptop', platform: 'darwin', cpu: 'Test CPU', cpuThreads: 8, memoryMb: 16384, gpus: [], gpuMemoryMb: 0, diskFreeMb: 50000, diskTotalMb: 100000 };
    const open = [];

    // Stands in for the desktop app: connects, says hello, and answers what the server asks of it.
    const fakeApp = (deviceId, key, { holdPull = false } = {}) => new Promise((resolve) => {
        const ws = new WebSocket(`ws://localhost:${testStartup.port}/devices`);
        const held = [];
        const app = { ws, welcomed: false, requests: [], closed: new Promise((done) => ws.addEventListener('close', done)), release: () => held.splice(0).forEach((answer) => answer()) };
        open.push(ws);
        const send = (message) => ws.send(JSON.stringify(message));
        ws.addEventListener('open', () => send({ type: 'hello', deviceId, key }));
        ws.addEventListener('close', () => resolve(app));
        ws.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'welcome') {
                app.welcomed = true;
                return resolve(app);
            }
            if (message.type === 'ask') {
                return send({ id: message.id, type: 'answer', data: message.what === 'disk' ? { diskFreeMb: 4242, diskTotalMb: 9000 } : { ...hardware, memoryMb: 32768 } });
            }
            if (message.type !== 'request') return undefined;
            app.requests.push(`${message.method} ${message.path}`);
            const body = message.path === '/api/version' ? { version: '9.9.9' } : message.path === '/api/tags' ? TAGS : message.path === '/api/ps' ? { models: [] } : null;
            if (!body) {
                const notFound = () => {
                    send({ id: message.id, type: 'head', status: 404, headers: { 'content-type': 'application/json' } });
                    send({ id: message.id, type: 'chunk', data: Buffer.from('{"error":"not found"}').toString('base64') });
                    send({ id: message.id, type: 'end' });
                };
                if (holdPull && message.path === '/api/pull') return held.push(notFound);
                return notFound();
            }
            const text = JSON.stringify(body);
            send({ id: message.id, type: 'head', status: 200, headers: { 'content-type': 'application/json' } });
            send({ id: message.id, type: 'chunk', data: Buffer.from(text.slice(0, 5)).toString('base64') });
            send({ id: message.id, type: 'chunk', data: Buffer.from(text.slice(5)).toString('base64') });
            return send({ id: message.id, type: 'end' });
        });
    });

    const link = async (hostname = 'laptop') => {
        const response = await client.post('/api/v1/devices/link', { hostname, hardware });
        expect(response.status).toBe(201);
        return { id: response.data.device.id, key: response.data.key };
    };
    const listed = async () => (await client.get('/api/v1/devices')).data.devices;
    test('a hello that cannot be checked just now asks the app to try again instead of forgetting its link', async () => {
        const gateway = new DeviceGateway();
        const closed = [];
        const answer = async (verify) => {
            let handler = null;
            const ws = { once: (event, fn) => { handler = fn; }, close: (code) => closed.push(code), on: () => {}, send: () => {} };
            gateway.handleConnection(ws, verify);
            await handler(JSON.stringify({ type: 'hello', deviceId: 'x', key: 'y' }));
        };
        await answer(async () => { throw new Error('database did not answer'); });
        await answer(async () => null);
        expect(closed).toEqual([1011, 1008]);
    });

    test('online model names are made readable when the service gives none', () => {
        expect(cloudTitle('claude-sonnet-4-5-20250929')).toBe('Claude Sonnet 4.5');
        expect(cloudTitle('models/gemini-2.5-pro')).toBe('Gemini 2.5 Pro');
        expect(cloudTitle('gpt-oss:120b-cloud')).toBe('GPT OSS 120B');
        expect(cloudTitle('grok-4-latest')).toBe('Grok 4');
    });

    test('a computer is shown by a readable name until the person names it', async () => {
        await testStartup.loginAsUser('admin');
        const named = async (id) => (await listed()).find((device) => device.id === id).name;
        const first = await client.post('/api/v1/devices/link', { hostname: 'Davids-MacBook-Pro.local', hardware });
        expect(first.data.device.name).toBe("David's MacBook Pro");
        expect(await named(first.data.device.id)).toBe("David's MacBook Pro");

        // The desktop app sends the computer's own name, which takes the place of the network name.
        const again = await client.post('/api/v1/devices/link', { hostname: 'Davids-MacBook-Pro.local', name: 'Studio Mac', hardware });
        expect(again.data.device.id).toBe(first.data.device.id);
        expect(await named(first.data.device.id)).toBe('Studio Mac');

        // A name the person chose is never replaced.
        await client.put(`/api/v1/devices/${first.data.device.id}`, { name: 'My desk' });
        await client.post('/api/v1/devices/link', { hostname: 'Davids-MacBook-Pro.local', name: 'Studio Mac', hardware });
        expect(await named(first.data.device.id)).toBe('My desk');

        const plain = await client.post('/api/v1/devices/link', { hostname: 'zero-pc', hardware });
        expect(plain.data.device.name).toBe('zero-pc');
    });

    test('a computer that reconnects while still online is not announced as a change', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('reconnecting-computer');
        const service = getNotificationService();
        const sent = [];
        const real = service.sendToUser.bind(service);
        service.sendToUser = (userId, notification) => {
            if (notification.type === 'device:changed' && notification.data?.deviceId === id) sent.push(notification.data.online);
            return real(userId, notification);
        };
        const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        try {
            await fakeApp(id, key);
            for (let attempt = 0; attempt < 50 && sent.length < 1; attempt++) await pause(10);
            expect(sent).toEqual([true]);
            await fakeApp(id, key);
            await pause(150);
            expect(sent).toEqual([true]);
        } finally {
            service.sendToUser = real;
        }
    });

    test('the models on a linked computer are listed for the chat picker', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('picker-computer');
        await fakeApp(id, key);
        const response = await client.client.get('/api/v1/models', through(id));
        expect(response.status).toBe(200);
        expect(response.data.ollamaAvailable).toBe(true);
        expect(response.data.models.map((model) => model.name)).toEqual(['tiny:1b']);
        expect(response.data.models[0].canChat).toBe(true);
        expect(response.data.device).toBe(id);
    });
    const until = async (check) => {
        for (let attempt = 0; attempt < 60; attempt += 1) {
            if (await check()) return true;
            await new Promise((resolve) => setTimeout(resolve, 50));
        }
        return false;
    };

    beforeAll(async () => {
        await Device.deleteMany({ kind: 'companion' });
    });

    beforeEach(() => {
        setOllamaClientFactory(simpleOllama());
    });

    afterAll(async () => {
        delete process.env.LOCAL_DEVICE;
        process.env.LOCAL_DEVICE = 'true';
        for (const ws of open) ws.close();
        await Device.deleteMany({ kind: 'companion' });
    });

    test('linking needs sign-in and a computer name', async () => {
        client.clearCookies();
        client.authenticatedAs = null;
        expect((await client.post('/api/v1/devices/link', { hostname: 'laptop' })).status).toBe(401);
        await testStartup.loginAsUser('user');
        expect((await client.post('/api/v1/devices/link', {})).status).toBe(400);
        expect((await client.post('/api/v1/devices/link', { hostname: 'laptop', hardware: { memoryMb: 'lots' } })).status).toBe(400);
    });

    test('a linked computer is listed for its owner as offline, and its key is never stored or shown', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link();
        expect(key).toMatch(/^[0-9a-f]{64}$/);

        const mine = (await listed()).find((device) => device.id === id);
        expect(mine).toMatchObject({ kind: 'companion', name: 'laptop', online: false, service: 'ai-base' });
        expect(mine.hardware.memoryMb).toBe(16384);
        expect(mine.ollama.available).toBe(false);
        expect(JSON.stringify(mine)).not.toContain(key);

        const stored = await Device.findById(id).select('+keyHash');
        expect(stored.keyHash).toHaveLength(64);
        expect(stored.keyHash).not.toBe(key);
        expect(String(stored.registeredBy)).toBe(String(testStartup.user.id || testStartup.user._id || stored.registeredBy));
    });

    test('another person cannot see, read, change or remove it', async () => {
        await testStartup.loginAsUser('user');
        const { id } = await link();

        await testStartup.loginAsUser('creator');
        expect((await listed()).some((device) => device.id === id)).toBe(false);
        expect((await client.get(`/api/v1/devices/${id}`)).status).toBe(404);
        expect((await client.delete(`/api/v1/devices/${id}`)).status).toBe(404);
        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.data.device?.id).not.toBe(id);

        await testStartup.loginAsUser('admin');
        expect((await client.put(`/api/v1/devices/${id}`, { name: 'Taken' })).status).toBe(404);
        expect(await Device.exists({ _id: id })).toBeTruthy();
    });

    test('the same computer name under two accounts makes two separate devices', async () => {
        await testStartup.loginAsUser('user');
        const first = await link('shared-name');
        await testStartup.loginAsUser('creator');
        const second = await link('shared-name');
        expect(second.id).not.toBe(first.id);
    });

    test('a wrong key, an unknown device and a silent connection are all turned away', async () => {
        await testStartup.loginAsUser('user');
        const { id } = await link();
        expect((await fakeApp(id, 'f'.repeat(64))).welcomed).toBe(false);
        expect((await fakeApp('0'.repeat(24), 'f'.repeat(64))).welcomed).toBe(false);
        expect((await fakeApp('not-an-id', 'x')).welcomed).toBe(false);
        expect((await listed()).find((device) => device.id === id).online).toBe(false);
    });

    test('with the right key the computer comes online and requests travel through it', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link();
        const app = await fakeApp(id, key);
        expect(app.welcomed).toBe(true);

        const mine = (await listed()).find((device) => device.id === id);
        expect(mine.online).toBe(true);
        expect(mine.ollama).toMatchObject({ available: true, version: '9.9.9', modelCount: 1 });

        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.status).toBe(200);
        expect(status.data.device).toMatchObject({ id, kind: 'companion' });
        expect(status.data.ollama).toMatchObject({ available: true, version: '9.9.9', modelCount: 1 });
        expect(status.data.steps).toEqual({ ollama: true, device: true, model: true });

        const running = await client.client.get('/api/v1/models/running', through(id));
        expect(running.status).toBe(200);
        expect(app.requests).toEqual(expect.arrayContaining(['GET /api/version', 'GET /api/tags', 'GET /api/ps']));

        const withoutHeader = await client.get('/api/v1/setup/status');
        expect(withoutHeader.data.device.kind).toBe('local');
    });

    test('one computer without Ollama is not a problem while another can run models', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('works');
        const app = await fakeApp(id, key);

        setOllamaClientFactory(simpleOllama({ up: false }));
        const status = await client.get('/api/v1/setup/status');
        expect(status.data.ollama).toMatchObject({ available: false, code: 'OLLAMA_DOWN' });
        expect(status.data.account).toEqual({ available: true });
        expect(status.data.ready).toBe(id);

        for (const ws of open) ws.close();
        await app.closed;
        expect(await until(async () => (await listed()).every((device) => device.kind === 'local' || !device.online))).toBe(true);
        const none = await client.get('/api/v1/setup/status');
        expect(none.data.ready).toBeNull();
        expect(none.data.account).toMatchObject({ available: false, code: 'NOTHING_READY' });
        expect(none.data.account.message).not.toMatch(/this device/i);
    });

    test('reading the hardware again asks the computer itself', async () => {
        await testStartup.loginAsUser('admin');
        const { id, key } = await link('workstation');
        await fakeApp(id, key);
        const response = await client.post(`/api/v1/devices/${id}/detect`, {});
        expect(response.status).toBe(200);
        expect(response.data.device.hardware.memoryMb).toBe(32768);
    });

    test('when the app disconnects the computer goes offline and requests to it fail plainly', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link();
        const app = await fakeApp(id, key);
        app.ws.close();
        await app.closed;
        expect(await until(async () => (await listed()).find((device) => device.id === id).online === false)).toBe(true);

        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.data.ollama).toMatchObject({ available: false, code: 'DEVICE_OFFLINE' });
        const models = await client.client.get('/api/v1/models/running', through(id));
        expect(models.status).toBe(503);
        expect(models.data.code).toBe('DEVICE_OFFLINE');
    });

    test('linking again replaces the key, so the old one stops working', async () => {
        await testStartup.loginAsUser('user');
        const first = await link();
        const app = await fakeApp(first.id, first.key);
        const second = await link();
        expect(second.id).toBe(first.id);
        expect(second.key).not.toBe(first.key);
        await app.closed;
        expect((await fakeApp(first.id, first.key)).welcomed).toBe(false);
        expect((await fakeApp(second.id, second.key)).welcomed).toBe(true);
    });

    test('signing out of the desktop app takes the computer off the account, but a dropped connection does not', async () => {
        await testStartup.loginAsUser('user');
        const kept = await link('just-offline');
        const dropped = await fakeApp(kept.id, kept.key);
        dropped.ws.close();
        await dropped.closed;
        expect(await until(async () => (await listed()).find((device) => device.id === kept.id)?.online === false)).toBe(true);

        const leaving = await link('signing-out');
        const app = await fakeApp(leaving.id, leaving.key);
        app.ws.send(JSON.stringify({ type: 'goodbye' }));
        app.ws.close();
        await app.closed;
        expect(await until(async () => !(await listed()).some((device) => device.id === leaving.id))).toBe(true);
        expect(await Device.exists({ _id: leaving.id })).toBeNull();
        expect((await fakeApp(leaving.id, leaving.key)).welcomed).toBe(false);
        expect(await Device.exists({ _id: kept.id })).toBeTruthy();
    });

    test('the same model can be downloading on two computers, each with its own list', async () => {
        await testStartup.loginAsUser('admin');
        const model = 'shared-model:1b';
        await Download.deleteMany({ model });
        const { id, key } = await link('second-computer');
        const app = await fakeApp(id, key, { holdPull: true });

        const onServer = await client.post('/api/v1/models/pull', { name: model });
        const onLinked = await client.client.post('/api/v1/models/pull', { name: model }, through(id));
        expect(onServer.status).toBe(202);
        expect(onLinked.status).toBe(202);
        expect(onLinked.data.download.id).not.toBe(onServer.data.download.id);
        expect(onServer.data.download.device).toBeNull();
        expect(onLinked.data.download.device).toBe(id);

        const again = await client.client.post('/api/v1/models/pull', { name: model }, through(id));
        expect(again.data.download.id).toBe(onLinked.data.download.id);

        const serverList = (await client.get('/api/v1/models/downloads')).data.downloads.filter((job) => job.model === model);
        const linkedList = (await client.client.get('/api/v1/models/downloads', through(id))).data.downloads.filter((job) => job.model === model);
        expect(serverList.map((job) => job.id)).toEqual([onServer.data.download.id]);
        expect(linkedList.map((job) => job.id)).toEqual([onLinked.data.download.id]);

        await testStartup.loginAsUser('creator');
        const others = (await client.client.get('/api/v1/models/downloads', through(id))).data.downloads;
        expect(others.some((job) => job.id === onLinked.data.download.id)).toBe(false);
        app.release();
        await downloads.settled(onServer.data.download.id).catch(() => {});
        await downloads.settled(onLinked.data.download.id).catch(() => {});
        await Download.deleteMany({ model });
    });

    test('a chat remembers its computer and uses it without being told again', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('chat-computer');
        const app = await fakeApp(id, key);

        const created = await client.post('/api/v1/chats', { device: id, model: 'tiny:1b' });
        expect(created.status).toBe(201);
        expect(created.data.chat.device).toBe(id);

        app.requests.length = 0;
        const opened = await client.get(`/api/v1/chats/${created.data.chat.id}`);
        expect(opened.data.chat.device).toBe(id);
        expect(app.requests.some((request) => request.startsWith('POST /api/show'))).toBe(true);

        const plain = await client.post('/api/v1/chats', { model: 'tiny:1b' });
        expect(plain.data.chat.device).toBeNull();
        app.requests.length = 0;
        await client.get(`/api/v1/chats/${plain.data.chat.id}`);
        expect(app.requests).toEqual([]);

        const cleared = await client.patch(`/api/v1/chats/${created.data.chat.id}`, { device: null });
        expect(cleared.data.chat.device).toBeNull();
        expect((await client.patch(`/api/v1/chats/${created.data.chat.id}`, { device: id })).data.chat.device).toBe(id);
    });

    test('a chat cannot be put on another person\'s computer', async () => {
        await testStartup.loginAsUser('user');
        const { id } = await link('not-yours');
        await testStartup.loginAsUser('creator');
        expect((await client.post('/api/v1/chats', { device: id })).status).toBe(404);
        const mine = await client.post('/api/v1/chats', {});
        expect((await client.patch(`/api/v1/chats/${mine.data.chat.id}`, { device: id })).status).toBe(404);
        expect((await client.patch(`/api/v1/chats/${mine.data.chat.id}`, { device: 'not-an-id' })).status).toBe(400);
    });

    test('a paused computer stays connected, is given no work, and works again when resumed', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('pause-me');
        const app = await fakeApp(id, key);

        app.requests.length = 0;
        const paused = await client.post(`/api/v1/devices/${id}/pause`, {});
        expect(paused.status).toBe(200);
        expect(paused.data.device).toMatchObject({ id, paused: true, online: true });
        expect(paused.data.device.ollama).toMatchObject({ available: false, paused: true });
        expect(app.requests).toContain('GET /api/ps');
        expect((await Device.findById(id)).paused).toBe(true);

        app.requests.length = 0;
        const mine = (await listed()).find((device) => device.id === id);
        expect(mine).toMatchObject({ paused: true, online: true });
        const refused = await client.client.get('/api/v1/models/running', through(id));
        expect(refused.status).toBe(503);
        expect(refused.data).toMatchObject({ code: 'DEVICE_PAUSED' });
        expect(refused.data.message).toMatch(/paused/i);
        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.data.ollama).toMatchObject({ available: false, code: 'DEVICE_PAUSED' });
        expect(app.requests).toEqual([]);

        const resumed = await client.post(`/api/v1/devices/${id}/resume`, {});
        expect(resumed.data.device).toMatchObject({ paused: false });
        expect(resumed.data.device.ollama).toMatchObject({ available: true });
        expect((await client.client.get('/api/v1/models/running', through(id))).status).toBe(200);

        await testStartup.loginAsUser('creator');
        expect((await client.post(`/api/v1/devices/${id}/pause`, {})).status).toBe(404);
        expect((await Device.findById(id)).paused).toBe(false);
    });

    // Stands in for a desktop app whose computer can chat and download, and whose connection can be cut part-way.
    const workingApp = (deviceId, key, { cutPull = false, cutChat = false } = {}) => new Promise((resolve) => {
        const ws = new WebSocket(`ws://localhost:${testStartup.port}/devices`);
        const app = { ws, requests: [], closed: new Promise((done) => ws.addEventListener('close', done)) };
        open.push(ws);
        const say = (message) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(message));
        const head = (id, status = 200, type = 'application/json') => say({ id, type: 'head', status, headers: { 'content-type': type } });
        const piece = (id, value) => say({ id, type: 'chunk', data: Buffer.from(typeof value === 'string' ? value : `${JSON.stringify(value)}\n`).toString('base64') });
        const whole = (id, body) => {
            head(id);
            piece(id, JSON.stringify(body));
            say({ id, type: 'end' });
        };
        ws.addEventListener('open', () => say({ type: 'hello', deviceId, key }));
        ws.addEventListener('close', () => resolve(app));
        ws.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'welcome') return resolve(app);
            if (message.type === 'ask') return say({ id: message.id, type: 'answer', data: message.what === 'disk' ? { diskFreeMb: 900000, diskTotalMb: 1000000 } : { ...hardware, memoryMb: 32768 } });
            if (message.type !== 'request') return undefined;
            app.requests.push(`${message.method} ${message.path}`);
            if (message.path === '/api/version') return whole(message.id, { version: '9.9.9' });
            if (message.path === '/api/tags') return whole(message.id, TAGS);
            if (message.path === '/api/ps') return whole(message.id, { models: [] });
            if (message.path === '/api/show') return whole(message.id, { capabilities: ['completion'], details: { parameter_size: '1B' }, model_info: { 'tiny.context_length': 8192 } });
            if (message.path === '/api/pull') {
                head(message.id, 200, 'application/x-ndjson');
                piece(message.id, { status: 'pulling manifest' });
                piece(message.id, { status: 'downloading', digest: 'sha256:abc', total: 1000, completed: 400 });
                if (cutPull) return setTimeout(() => ws.close(), 60);
                piece(message.id, { status: 'downloading', digest: 'sha256:abc', total: 1000, completed: 1000 });
                piece(message.id, { status: 'success' });
                return say({ id: message.id, type: 'end' });
            }
            if (message.path === '/api/chat') {
                head(message.id, 200, 'application/x-ndjson');
                piece(message.id, { model: 'tiny:1b', message: { role: 'assistant', content: 'Half an ' }, done: false });
                if (cutChat) return setTimeout(() => ws.close(), 60);
                piece(message.id, { model: 'tiny:1b', message: { role: 'assistant', content: 'answer.' }, done: false });
                piece(message.id, { model: 'tiny:1b', message: { role: 'assistant', content: '' }, done: true, eval_count: 3, eval_duration: 1000000 });
                return say({ id: message.id, type: 'end' });
            }
            head(message.id, 404);
            piece(message.id, '{"error":"not found"}');
            return say({ id: message.id, type: 'end' });
        });
    });

    test('a download whose computer drops off fails plainly, then carries on when the computer returns', async () => {
        await Download.deleteMany({});
        await testStartup.loginAsUser('admin');
        const { id, key } = await link('drops-mid-download');
        const first = await workingApp(id, key, { cutPull: true });

        const started = await client.client.post('/api/v1/models/pull', { name: 'someone/unlisted:q4' }, through(id));
        expect(started.status).toBe(202);
        const jobId = started.data.download.id;
        await first.closed;
        expect(await until(async () => (await Download.findById(jobId)).status === 'failed')).toBe(true);
        const failed = await Download.findById(jobId);
        expect(failed.errorCode).toBe('DEVICE_OFFLINE');
        expect(failed.completedBytes).toBe(400);
        expect(new OllamaError(failed.errorCode).message).toMatch(/not connected/i);

        const second = await workingApp(id, key);
        expect(await until(async () => (await Download.findById(jobId)).status === 'completed')).toBe(true);
        expect(second.requests).toContain('POST /api/pull');
        const done = await Download.findById(jobId);
        expect(done.errorCode).toBeNull();
        expect(done.completedBytes).toBe(1000);
        expect(String(done.device)).toBe(id);
    });

    test('a download that failed for another reason is not started again when the computer returns', async () => {
        await Download.deleteMany({});
        await testStartup.loginAsUser('admin');
        const { id, key } = await link('no-auto-resume');
        const owner = (await Device.findById(id)).registeredBy;
        const old = await Download.create({ model: 'someone/old:q4', device: id, status: 'failed', errorCode: 'DEVICE_OFFLINE', requestedBy: owner, finishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000) });
        const other = await Download.create({ model: 'someone/full:q4', device: id, status: 'failed', errorCode: 'DISK_FULL', requestedBy: owner, finishedAt: new Date() });
        const app = await workingApp(id, key);
        await new Promise((resolve) => setTimeout(resolve, 400));
        expect(app.requests).not.toContain('POST /api/pull');
        expect((await Download.findById(old.id)).status).toBe('failed');
        expect((await Download.findById(other.id)).status).toBe('failed');
    });

    test('a reply whose computer drops off keeps what was written and ends with a plain reason', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('drops-mid-reply');
        const app = await workingApp(id, key, { cutChat: true });
        const chat = (await client.post('/api/v1/chats', { device: id, model: 'tiny:1b' })).data.chat;

        const response = await client.post(`/api/v1/chats/${chat.id}/messages`, { content: 'say something long' }, { responseType: 'text', transformResponse: [(data) => data] });
        expect(response.status).toBe(200);
        const pieces = String(response.data).split('\n').filter(Boolean).map((line) => JSON.parse(line));
        expect(pieces.filter((piece) => piece.type === 'token').map((piece) => piece.text).join('')).toBe('Half an ');
        await app.closed;

        const saved = (await client.get(`/api/v1/chats/${chat.id}`)).data.chat;
        const last = saved.messages[saved.messages.length - 1];
        expect(last).toMatchObject({ role: 'assistant', content: 'Half an ' });
        expect(last.error).toMatchObject({ code: 'DEVICE_OFFLINE' });
        expect(last.error.message).toMatch(/not connected/i);
        expect(saved.replying).toBe(false);

        await workingApp(id, key);
        const again = await client.post(`/api/v1/chats/${chat.id}/messages`, { regenerate: true }, { responseType: 'text', transformResponse: [(data) => data] });
        expect(again.status).toBe(200);
        const after = (await client.get(`/api/v1/chats/${chat.id}`)).data.chat;
        expect(after.messages[after.messages.length - 1]).toMatchObject({ role: 'assistant', content: 'Half an answer.' });
        expect(after.messages[after.messages.length - 1].error).toBeNull();
    });

    test('removing a computer disconnects it and cancels its key', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('to-remove');
        const app = await fakeApp(id, key);
        expect((await client.delete(`/api/v1/devices/${id}`)).status).toBe(200);
        await app.closed;
        expect(await Device.exists({ _id: id })).toBeNull();
        expect((await fakeApp(id, key)).welcomed).toBe(false);
    });

    test('the server\'s own computer cannot be removed', async () => {
        await testStartup.loginAsUser('admin');
        const local = (await listed()).find((device) => device.kind === 'local');
        expect((await client.delete(`/api/v1/devices/${local.id}`)).status).toBe(404);
    });

    test('an Apple-chip Mac shows its built-in graphics even when its app reported none', async () => {
        await testStartup.loginAsUser('user');
        const reported = { ...hardware, platform: 'darwin', gpus: [], gpuMemoryMb: 0 };
        const mac = await client.post('/api/v1/devices/link', { hostname: 'apple-mac', hardware: { ...reported, cpu: 'Apple M3 Pro', memoryMb: 18432 } });
        const intel = await client.post('/api/v1/devices/link', { hostname: 'intel-mac', hardware: { ...reported, cpu: 'Intel(R) Core(TM) i7-9750H', memoryMb: 16384 } });
        const devices = await listed();
        const shown = (id) => devices.find((device) => device.id === id).hardware;

        expect(shown(mac.data.device.id)).toMatchObject({ gpuName: 'Apple M3 Pro built-in graphics', gpuMemoryMb: 12288, hasGpu: true });
        expect(shown(intel.data.device.id)).toMatchObject({ gpuName: null, gpuMemoryMb: 0, hasGpu: false });
    });

    test('a browser on another address is allowed to send the chosen device with its requests', async () => {
        const origin = process.env.ALLOWED_ORIGINS.split(',')[0].trim();
        const preflight = await fetch(`http://localhost:${testStartup.port}/api/v1/devices`, {
            method: 'OPTIONS',
            headers: { Origin: origin, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-mollama-device' }
        });
        expect(preflight.status).toBe(200);
        expect(preflight.headers.get('access-control-allow-origin')).toBe(origin);
        expect(preflight.headers.get('access-control-allow-headers').toLowerCase()).toContain('x-mollama-device');
    });

    test('no running server counts as a computer, in development, staging or production, whatever its settings say', async () => {
        await testStartup.loginAsUser('admin');
        const kinds = async () => (await listed()).map((device) => device.kind);
        const before = { local: process.env.LOCAL_DEVICE, mode: process.env.NODE_ENV };
        try {
            expect(await kinds()).toContain('local');
            for (const mode of ['development', 'staging', 'production']) {
                process.env.NODE_ENV = mode;
                for (const setting of [undefined, '', 'false', 'true']) {
                    if (setting === undefined) delete process.env.LOCAL_DEVICE;
                    else process.env.LOCAL_DEVICE = setting;
                    expect(await kinds()).not.toContain('local');
                    const status = await client.get('/api/v1/setup/status');
                    expect(status.data.device?.kind).not.toBe('local');
                }
            }
        } finally {
            process.env.NODE_ENV = before.mode;
            process.env.LOCAL_DEVICE = before.local;
        }
    });

    test('with the server\'s own computer switched off, only linked computers count', async () => {
        await Device.deleteMany({ kind: 'companion' });
        process.env.LOCAL_DEVICE = 'false';
        try {
            await testStartup.loginAsUser('creator');
            expect(await listed()).toEqual([]);
            const none = await client.get('/api/v1/setup/status');
            expect(none.data.device).toBeNull();
            expect(none.data.ollama).toMatchObject({ available: false, code: 'NO_DEVICE' });
            expect(none.data.account).toMatchObject({ available: false, code: 'NO_DEVICE' });
            const refused = await client.get('/api/v1/models/running');
            expect(refused.status).toBe(503);
            expect(refused.data.code).toBe('NO_DEVICE');

            const { id, key } = await link('only-one');
            const offline = await client.get('/api/v1/setup/status');
            expect(offline.data.ollama.code).toBe('DEVICE_OFFLINE');

            await fakeApp(id, key);
            const devices = await listed();
            expect(devices.map((device) => device.kind)).toEqual(['companion']);
            const ready = await client.get('/api/v1/setup/status');
            expect(ready.data.device).toMatchObject({ id, kind: 'companion' });
            expect(ready.data.ollama.available).toBe(true);
            expect(ready.data.ready).toBe(id);
        } finally {
            process.env.LOCAL_DEVICE = 'true';
        }
    });
});

describe('Tool servers', () => {
    const any = { validateStatus: () => true };
    const api = '/api/v1/tools/servers';
    const TOOLS = [
        { name: 'find_page', title: 'Find a page', description: 'Finds a page by words in it', annotations: { readOnlyHint: true }, inputSchema: { type: 'object', properties: { words: { type: 'string' } } } },
        { name: 'write_page', description: 'Writes a page', inputSchema: { type: 'object' } }
    ];
    const hardware = { hostname: 'toolbox', platform: 'darwin', cpu: 'Test CPU', cpuThreads: 8, memoryMb: 16384, gpus: [], gpuMemoryMb: 0, diskFreeMb: 50000, diskTotalMb: 100000 };
    const calls = [];
    const open = [];
    let finished = [];

    // Stands in for the tool-server library: what an address does is read from its host name.
    const connector = async (address, options) => {
        const host = new URL(address).hostname;
        calls.push({ host, headers: options.requestInit?.headers || null, keeper: Boolean(options.authProvider) });
        if (host === 'down.example') throw new Error('connect failed');
        if (host === 'crowded.example') throw Object.assign(new Error('slow down'), { status: 429 });
        if (host === 'inside.example') throw new OllamaError('REMOTE_BLOCKED');
        if (host === 'keyed.example' && options.requestInit?.headers?.Authorization !== 'Bearer good-key') throw Object.assign(new Error('no'), { status: 401 });
        if (host === 'account.example' && !(await options.authProvider?.tokens())) {
            await options.authProvider?.redirectToAuthorization(new URL(`https://account.example/authorize?state=${await options.authProvider.state?.()}`));
            throw new ToolSignInError('sign in');
        }
        return { listTools: async () => ({ tools: host === 'bare.example' ? [] : TOOLS }), close: async () => {} };
    };
    const finisher = async (address, keeper, params) => {
        finished.push(Object.fromEntries(params));
        if (params.get('code') !== 'good-code') throw new Error('bad code');
        await keeper.saveTokens({ access_token: 'secret-access-token', token_type: 'Bearer' });
    };
    const add = (body) => client.post(api, { kind: 'remote', ...body }, any);
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };

    beforeAll(async () => {
        setToolConnector(connector, finisher);
    });
    afterAll(async () => {
        setToolConnector();
        for (const ws of open) ws.close();
        await ToolServer.deleteMany({});
    });
    beforeEach(async () => {
        calls.length = 0;
        finished = [];
        await ToolServer.deleteMany({});
        signOut();
        await testStartup.loginAsUser('user');
    });

    test('sealing keeps a value unreadable and gives it back whole, and a damaged seal gives nothing', () => {
        const sealed = seal({ key: 'good-key' });
        expect(sealed).not.toContain('good-key');
        expect(unseal(sealed)).toEqual({ key: 'good-key' });
        expect(unseal(`${sealed.slice(0, -4)}AAAA`)).toBeNull();
        expect(seal({ key: 'good-key' })).not.toBe(sealed);
    });

    test('the list of online apps is served, and every entry is a named https address that can be added as it is', async () => {
        const response = await client.get('/api/v1/tools/catalogue', any);
        expect(response.status).toBe(200);
        expect(response.data.apps).toEqual(TOOL_CATALOGUE.map((entry) => ({ ...entry, keyNeeded: entry.signIn === 'key' })));
        expect(TOOL_CATALOGUE.length).toBeGreaterThanOrEqual(30);
        expect(new Set(TOOL_CATALOGUE.map((entry) => entry.name)).size).toBe(TOOL_CATALOGUE.length);
        expect(new Set(TOOL_CATALOGUE.map((entry) => entry.address)).size).toBe(TOOL_CATALOGUE.length);
        for (const entry of TOOL_CATALOGUE) {
            expect(toolSchemas.add.validate({ kind: 'remote', name: entry.name, address: entry.address }).error).toBeUndefined();
            expect(['account', 'key', 'none']).toContain(entry.signIn);
            if (entry.signIn === 'key') expect(Boolean(entry.key)).toBe(true);
            if (entry.signIn === 'account') expect(entry.key).toBeNull();
            if (entry.key) {
                expect(entry.key.page).toMatch(/^https:\/\//);
                expect(entry.key.steps.length).toBeGreaterThanOrEqual(3);
                expect(entry.key.cost.length).toBeGreaterThan(3);
            }
            expect(entry.does.length).toBeGreaterThan(10);
            expect(entry.group.length).toBeGreaterThan(3);
        }
        signOut();
        expect((await client.get('/api/v1/tools/catalogue', any)).status).toBe(401);
    });

    test('the suggested local apps are served, each with what starts it and what it needs first', async () => {
        const response = await client.get('/api/v1/tools/catalogue', any);
        expect(response.data.localApps).toEqual(TOOL_LOCAL_CATALOGUE);
        expect(TOOL_LOCAL_CATALOGUE.length).toBeGreaterThanOrEqual(12);
        expect(new Set(TOOL_LOCAL_CATALOGUE.map((entry) => entry.name)).size).toBe(TOOL_LOCAL_CATALOGUE.length);
        for (const entry of TOOL_LOCAL_CATALOGUE) {
            expect(entry.command).toMatch(/^[a-z]+$/);
            expect(entry.args).not.toMatch(/[;&|`$]/);
            expect(['maker', 'standard', 'community']).toContain(entry.by);
            expect(entry.needs.length).toBeGreaterThan(0);
            expect(entry.does.length).toBeGreaterThan(10);
            if (entry.by === 'community') expect(entry.caution).toMatch(/community/);
            if (entry.args.includes('<')) expect(entry.needs.join(' ')).toMatch(/Replace </);
        }
    });

    test('every health state has a label, and every state but healthy says what is needed', () => {
        expect(Object.keys(TOOL_HEALTH).sort()).toEqual([...TOOL_HEALTH_STATES].sort());
        for (const [state, entry] of Object.entries(TOOL_HEALTH)) {
            expect(entry.label.length).toBeGreaterThan(0);
            expect(entry.needs.length > 0).toBe(state !== 'healthy');
        }
    });

    test('a remote server that answers is healthy and lists its tools, with read-only ones marked', async () => {
        const response = await add({ name: 'Notes', address: 'https://open.example/mcp' });
        expect(response.status).toBe(201);
        const { server } = response.data;
        expect(server.health.state).toBe('healthy');
        expect(server.health.needs).toEqual([]);
        expect(server.tools).toEqual([
            { name: 'find_page', title: 'Find a page', description: 'Finds a page by words in it', readOnly: true },
            { name: 'write_page', title: null, description: 'Writes a page', readOnly: false }
        ]);
        expect(server.signIn).toBe('none');
        const listed = await client.get(api, any);
        expect(listed.data.servers.map((one) => one.name)).toEqual(['Notes']);
    });

    test('a key is sent to the server, stored sealed, and never returned', async () => {
        const response = await add({ name: 'Keyed', address: 'https://keyed.example/mcp', key: 'good-key' });
        expect(response.data.server.health.state).toBe('healthy');
        expect(response.data.server.signIn).toBe('key');
        expect(calls[0].headers).toEqual({ Authorization: 'Bearer good-key' });
        const stored = await ToolServer.findOne({ name: 'Keyed' }).select('+sealed');
        expect(stored.sealed).not.toContain('good-key');
        expect(JSON.stringify((await client.get(api, any)).data)).not.toContain('good-key');
        expect(JSON.stringify(response.data)).not.toContain('good-key');
    });

    test('a wrong key is reported as refused, and replacing it makes the server healthy', async () => {
        const response = await add({ name: 'Keyed', address: 'https://keyed.example/mcp', key: 'wrong' });
        expect(response.data.server.health.state).toBe('refused');
        const fixed = await client.patch(`${api}/${response.data.server.id}`, { key: 'good-key' }, any);
        expect(fixed.data.server.health.state).toBe('healthy');
    });

    test('unreachable, blocked and empty servers are saved with a state that says what is needed', async () => {
        const down = (await add({ name: 'Down', address: 'https://down.example/mcp' })).data.server;
        const inside = (await add({ name: 'Inside', address: 'https://inside.example/mcp' })).data.server;
        const bare = (await add({ name: 'Bare', address: 'https://bare.example/mcp' })).data.server;
        expect([down.health.state, inside.health.state, bare.health.state]).toEqual(['unreachable', 'blocked', 'empty']);
        for (const server of [down, inside, bare]) expect(server.health.needs.length).toBeGreaterThan(0);
        expect(down.tools).toEqual([]);
    });

    test('on a hosted site an app that offers free keys needs one, and elsewhere it can be added without', async () => {
        const exa = TOOL_CATALOGUE.find((entry) => entry.name === 'Exa');
        const open = TOOL_CATALOGUE.find((entry) => entry.name === 'DeepWiki');
        expect((await add({ name: exa.name, address: exa.address })).status).toBe(201);
        await ToolServer.deleteMany({});

        const usual = process.env.NODE_ENV;
        process.env.NODE_ENV = 'production';
        try {
            const listed = (await client.get('/api/v1/tools/catalogue', any)).data.apps;
            expect(listed.filter((entry) => entry.keyNeeded).map((entry) => entry.name).sort()).toEqual(['Context7', 'Exa', 'Firecrawl', 'GitHub', 'Hugging Face']);
            const refused = await add({ name: exa.name, address: exa.address });
            expect(refused.status).toBe(422);
            expect(refused.data.action).toMatch(/free key/);
            expect(await ToolServer.countDocuments({})).toBe(0);
            expect((await add({ name: open.name, address: open.address })).status).toBe(201);
            calls.length = 0;
            expect((await add({ name: exa.name, address: exa.address, key: 'exa-key' })).status).toBe(201);
            expect(calls[0].headers).toEqual({ 'x-api-key': 'exa-key' });
        } finally {
            process.env.NODE_ENV = usual;
        }
    });

    test('a server that says it is taking too many requests is busy, not unreachable, and the way out is a key', async () => {
        const crowded = (await add({ name: 'Crowded', address: 'https://crowded.example/mcp' })).data.server;
        expect(crowded.health.state).toBe('busy');
        expect(crowded.health.label).toBe('Busy right now');
        expect(crowded.health.needs.join(' ')).toMatch(/your own key/);
    });

    test('an address that is not https, or has no name, is refused before anything is saved', async () => {
        expect((await add({ name: 'Plain', address: 'http://open.example/mcp' })).status).toBe(400);
        expect((await add({ address: 'https://open.example/mcp' })).status).toBe(400);
        expect((await add({ name: 'Mixed', address: 'https://open.example/mcp', localId: 'abcdefgh' })).status).toBe(400);
        expect(await ToolServer.countDocuments({})).toBe(0);
    });

    test('a server that wants an account says sign-in is needed, and signing in makes it healthy', async () => {
        const added = (await add({ name: 'Account', address: 'https://account.example/mcp' })).data.server;
        expect(added.health.state).toBe('signin');
        expect(added.signIn).toBe('account');

        const started = await client.post(`${api}/${added.id}/sign-in`, {}, any);
        expect(started.status).toBe(200);
        const state = new URL(started.data.url).searchParams.get('state');
        expect(state.length).toBeGreaterThan(20);

        signOut();
        const back = await client.client.get(`/api/v1/tools/callback?code=good-code&state=${state}`, { maxRedirects: 0, validateStatus: () => true });
        expect(back.status).toBe(302);
        expect(back.headers.location).toBe(`${process.env.APP_URL}/tools?signin=done`);
        expect(finished).toEqual([{ code: 'good-code', state }]);

        await testStartup.loginAsUser('user');
        const after = (await client.get(api, any)).data.servers[0];
        expect(after.health.state).toBe('healthy');
        expect(after.tools.length).toBe(2);
        const stored = await ToolServer.findById(added.id).select('+sealed +signInState');
        expect(stored.sealed).not.toContain('secret-access-token');
        expect(unseal(stored.sealed).tokens.access_token).toBe('secret-access-token');
        expect(stored.signInState).toBeNull();
        expect(JSON.stringify(after)).not.toContain('secret-access-token');
    });

    test('a sign-in that comes back with the wrong state or a bad code changes nothing', async () => {
        const added = (await add({ name: 'Account', address: 'https://account.example/mcp' })).data.server;
        const started = await client.post(`${api}/${added.id}/sign-in`, {}, any);
        const state = new URL(started.data.url).searchParams.get('state');
        const options = { maxRedirects: 0, validateStatus: () => true };
        const wrong = await client.client.get('/api/v1/tools/callback?code=good-code&state=someone-elses', options);
        expect(wrong.headers.location).toBe(`${process.env.APP_URL}/tools?signin=failed`);
        const bad = await client.client.get(`/api/v1/tools/callback?code=bad&state=${state}`, options);
        expect(bad.headers.location).toBe(`${process.env.APP_URL}/tools?signin=failed`);
        const again = await client.client.get(`/api/v1/tools/callback?code=good-code&state=${state}`, options);
        expect(again.headers.location).toBe(`${process.env.APP_URL}/tools?signin=failed`);
        expect((await client.get(api, any)).data.servers[0].health.state).toBe('signin');
    });

    test('a server with no sign-in to offer says so plainly', async () => {
        const added = (await add({ name: 'Down', address: 'https://down.example/mcp' })).data.server;
        const started = await client.post(`${api}/${added.id}/sign-in`, {}, any);
        expect(started.status).toBe(422);
        expect(started.data.action).toMatch(/key/);
    });

    test('checking again picks up a change and tells the person over the notification channel', async () => {
        const added = (await add({ name: 'Notes', address: 'https://open.example/mcp' })).data.server;
        await ToolServer.updateOne({ _id: added.id }, { address: 'https://down.example/mcp' });
        const sent = [];
        const service = getNotificationService();
        const original = service.sendToUser;
        service.sendToUser = (userId, message) => { sent.push(message); };
        try {
            const checked = await client.post(`${api}/${added.id}/check`, {}, any);
            expect(checked.data.server.health.state).toBe('unreachable');
            expect(checked.data.server.tools.length).toBe(2);
            await client.post(`${api}/${added.id}/check`, {}, any);
        } finally {
            service.sendToUser = original;
        }
        expect(sent.filter((message) => message.type === 'tools:changed')).toEqual([{ type: 'tools:changed', data: { id: added.id, health: 'unreachable' } }]);
    });

    test('one person cannot see, check, change or remove the tool servers of another', async () => {
        const added = (await add({ name: 'Mine', address: 'https://open.example/mcp' })).data.server;
        signOut();
        await testStartup.loginAsUser('creator');
        expect((await client.get(api, any)).data.servers).toEqual([]);
        expect((await client.post(`${api}/${added.id}/check`, {}, any)).status).toBe(404);
        expect((await client.patch(`${api}/${added.id}`, { name: 'Theirs' }, any)).status).toBe(404);
        expect((await client.post(`${api}/${added.id}/sign-in`, {}, any)).status).toBe(404);
        expect((await client.delete(`${api}/${added.id}`, any)).status).toBe(404);
        expect((await ToolServer.findById(added.id)).name).toBe('Mine');
    });

    test('renaming and removing work, the limit is twenty, and a signed-out visitor is turned away', async () => {
        const added = (await add({ name: 'Notes', address: 'https://open.example/mcp' })).data.server;
        const owner = (await ToolServer.findById(added.id)).owner;
        expect((await client.patch(`${api}/${added.id}`, { name: 'Notebook' }, any)).data.server.name).toBe('Notebook');
        expect((await client.delete(`${api}/${added.id}`, any)).status).toBe(200);
        expect(await ToolServer.countDocuments({})).toBe(0);

        await ToolServer.insertMany(Array.from({ length: 20 }, (unused, index) => ({ owner, name: `s${index}`, kind: 'remote', address: 'https://open.example/mcp' })));
        expect((await add({ name: 'One more', address: 'https://open.example/mcp' })).status).toBe(422);

        signOut();
        expect((await client.get(api, any)).status).toBe(401);
    });

    describe('On a linked computer', () => {
        const fakeApp = (deviceId, key, answer) => new Promise((resolve) => {
            const ws = new WebSocket(`ws://localhost:${testStartup.port}/devices`);
            const app = { ws, asked: [] };
            open.push(ws);
            ws.addEventListener('open', () => ws.send(JSON.stringify({ type: 'hello', deviceId, key })));
            ws.addEventListener('message', (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'welcome') return resolve(app);
                if (message.type !== 'tools') return undefined;
                app.asked.push({ action: message.action, localId: message.localId });
                const data = answer(message);
                return data ? ws.send(JSON.stringify({ id: message.id, type: 'answer', data })) : undefined;
            });
        });
        const link = async () => {
            const response = await client.post('/api/v1/devices/link', { hostname: `toolbox-${Date.now()}`, hardware }, any);
            expect(response.status).toBe(201);
            return { id: response.data.device.id, key: response.data.key };
        };
        const addLocal = (device, localId = 'blender-1234') => client.post(api, { kind: 'local', name: 'Blender', device, localId }, any);

        test('a local server on a computer that is not connected is saved as offline', async () => {
            const { id } = await link();
            const response = await addLocal(id);
            expect(response.status).toBe(201);
            expect(response.data.server.health.state).toBe('offline');
            expect(response.data.server.kind).toBe('local');
            expect(response.data.server.address).toBeNull();
        });

        test('a connected computer is asked, and what its desktop app reports is recorded', async () => {
            const { id, key } = await link();
            let reply = { state: 'healthy', tools: [{ name: 'make_cube', description: 'Adds a cube', readOnly: false }] };
            const app = await fakeApp(id, key, () => reply);
            const added = (await addLocal(id)).data.server;
            expect(app.asked).toEqual([{ action: 'check', localId: 'blender-1234' }]);
            expect(added.health.state).toBe('healthy');
            expect(added.tools.map((tool) => tool.name)).toEqual(['make_cube']);

            reply = { state: 'stopped', detail: 'The program uvx was not found.' };
            const checked = (await client.post(`${api}/${added.id}/check`, {}, any)).data.server;
            expect(checked.health.state).toBe('stopped');
            expect(checked.health.detail).toBe('The program uvx was not found.');
            expect(checked.tools.map((tool) => tool.name)).toEqual(['make_cube']);
            expect((await addLocal(id)).status).toBe(409);
        });

        test('a desktop app too old to answer is reported as needing an update', async () => {
            const { id, key } = await link();
            await fakeApp(id, key, () => null);
            const usual = deviceGateway.askMs;
            deviceGateway.askMs = 200;
            try {
                expect((await addLocal(id)).data.server.health.state).toBe('outdated');
            } finally {
                deviceGateway.askMs = usual;
            }
        });

        test('a computer that belongs to someone else cannot be used', async () => {
            const { id } = await link();
            signOut();
            await testStartup.loginAsUser('creator');
            expect((await addLocal(id)).status).toBe(404);
        });
    });
});

describe('Online model providers', () => {
    const any = { validateStatus: () => true };
    const api = '/api/v1/providers';
    const asked = [];
    const through = (id) => ({ headers: { 'X-Mollama-Device': id }, validateStatus: () => true });
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };
    const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    const stream = (events) => new Response(new ReadableStream({
        start(controller) {
            for (const event of events) controller.enqueue(new TextEncoder().encode(`data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`));
            controller.close();
        }
    }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });

    // Stands in for every online service: what a key does is read from the key itself.
    const service = async (address, init = {}) => {
        const url = new URL(String(address));
        const key = new Headers(init.headers || {}).get('authorization')?.replace('Bearer ', '') || '';
        const body = init.body ? JSON.parse(init.body) : null;
        asked.push({ host: url.host, path: url.pathname, key, body });
        if (key === 'bad-key-0000') return json({ error: { message: 'invalid key' } }, 401);
        if (key === 'odd-key-0000') return json({ error: { code: 400, message: 'Please pass a valid API key' } }, 400);
        if (key === 'poor-key-0000') return json({ error: { message: 'insufficient balance' } }, 402);
        if (key === 'slow-key-0000') return json({ error: { message: 'slow down' } }, 429);
        if (url.host === 'api.anthropic.com') {
            asked.at(-1).version = new Headers(init.headers || {}).get('anthropic-version');
            if (key !== 'good-key-0000') return json({ type: 'error', error: { type: 'authentication_error', message: 'invalid x-api-key' } }, 401);
            if (url.pathname === '/v1/models') return json({ data: [{ type: 'model', id: 'claude-large', display_name: 'Claude Large', max_input_tokens: 1000000, max_tokens: 128000, capabilities: { image_input: { supported: true } } }], has_more: false, first_id: 'claude-large', last_id: 'claude-large' });
            if (url.pathname === '/v1/messages') {
                const events = [
                    ['message_start', { type: 'message_start', message: { id: 'msg_1', type: 'message', role: 'assistant', model: body.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 30, output_tokens: 1 } } }],
                    ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } }],
                    ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Considering. ' } }],
                    ['content_block_stop', { type: 'content_block_stop', index: 0 }],
                    ['content_block_start', { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } }],
                    ['content_block_delta', { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Hello from ' } }],
                    ['content_block_delta', { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Claude.' } }],
                    ['content_block_stop', { type: 'content_block_stop', index: 1 }],
                    ['message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 12 } }],
                    ['message_stop', { type: 'message_stop' }]
                ];
                return new Response(events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join(''), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
            }
        }
        if (url.pathname.endsWith('/api/tags')) return json({ models: [{ name: 'kimi-k3', capabilities: ['completion', 'vision'] }, { name: 'embed-one', capabilities: ['embedding'] }] });
        if (url.pathname.endsWith('/models')) {
            return json({ data: key === 'none-key-0000' ? [] : [
                { id: 'big-one', context_length: 200000, supports_image_in: true },
                { id: 'models/small-one' },
                { id: 'text-embedding-3' },
                { id: 'whisper-9' }
            ] });
        }
        if (url.pathname.endsWith('/chat/completions')) {
            if (!body.stream) return json({ choices: [{ message: { content: 'A short summary.' } }], usage: { prompt_tokens: 9, completion_tokens: 4 } });
            return stream([
                { choices: [{ delta: { reasoning_content: 'Let me think. ' } }] },
                { choices: [{ delta: { content: 'Hello ' } }] },
                { choices: [{ delta: { content: [{ type: 'text', text: 'from online.' }] } }] },
                { choices: [], usage: { prompt_tokens: 21, completion_tokens: 7 } },
                '[DONE]'
            ]);
        }
        return json({ error: { message: 'no such path' } }, 404);
    };
    const add = (kind, key = 'good-key-0000') => client.post(api, { kind, key }, any);

    beforeAll(() => setCloudFetch(service));
    afterAll(async () => {
        setCloudFetch();
        await Provider.deleteMany({});
    });
    beforeEach(async () => {
        asked.length = 0;
        await Provider.deleteMany({});
        signOut();
        await testStartup.loginAsUser('user');
    });

    test('every suggested service has a secure address, a key page and steps, and every state says what is needed', () => {
        expect(CLOUD_PROVIDERS.length).toBeGreaterThanOrEqual(10);
        expect(CLOUD_PROVIDERS[0].id).toBe('ollama');
        expect(new Set(CLOUD_PROVIDERS.map((entry) => entry.id)).size).toBe(CLOUD_PROVIDERS.length);
        for (const entry of CLOUD_PROVIDERS) {
            expect(entry.base).toMatch(/^https:\/\/[^/]+/);
            expect(entry.base.endsWith('/')).toBe(false);
            expect(entry.key.page).toMatch(/^https:\/\//);
            expect(entry.key.steps.length).toBe(3);
            expect(providerSchemas.add.validate({ kind: entry.id, key: 'good-key-0000' }).error).toBeUndefined();
        }
        expect(Object.keys(PROVIDER_HEALTH).sort()).toEqual([...PROVIDER_HEALTH_STATES].sort());
        for (const [state, entry] of Object.entries(PROVIDER_HEALTH)) expect(entry.needs.length > 0).toBe(state !== 'healthy');
        for (const code of ['KEY_REFUSED', 'OUT_OF_CREDIT', 'PROVIDER_BUSY', 'PROVIDER_DOWN']) expect(new OllamaError(code).code).toBe(code);
    });

    test('adding a service with a working key lists its chat models, and the key is sealed and never returned', async () => {
        const response = await add('openai');
        expect(response.status).toBe(201);
        const { provider } = response.data;
        expect(provider.health.state).toBe('healthy');
        expect(provider.models).toEqual(['big-one', 'small-one']);
        expect(asked[0]).toMatchObject({ host: 'api.openai.com', path: '/v1/models', key: 'good-key-0000' });
        const stored = await Provider.findById(provider.id).select('+sealed');
        expect(stored.sealed).not.toContain('good-key-0000');
        expect(unseal(stored.sealed)).toEqual({ key: 'good-key-0000' });
        const listed = await client.get(api, any);
        expect(JSON.stringify(listed.data)).not.toContain('good-key-0000');
        expect(JSON.stringify(response.data)).not.toContain('good-key-0000');
        expect(listed.data.providers.map((one) => one.name)).toEqual(['OpenAI']);
        expect(listed.data.suggested.map((one) => one.id)).toEqual(CLOUD_PROVIDERS.map((one) => one.id));
        expect(listed.data.suggested[0].base).toBeUndefined();
    });

    test('Ollama online is asked the way Ollama is asked, and models that only make embeddings are left out', async () => {
        const { provider } = (await add('ollama')).data;
        expect(provider.health.state).toBe('healthy');
        expect(provider.models).toEqual(['kimi-k3']);
        expect(asked.map((one) => `${one.host}${one.path}`)).toEqual(['ollama.com/api/me', 'ollama.com/api/tags']);
        expect(asked.every((one) => one.key === 'good-key-0000')).toBe(true);
    });

    test('a service that lists its models to anyone is still asked to prove the key, and one that answers a bad key oddly is understood', async () => {
        expect((await add('ollama', 'bad-key-0000')).data.provider.health.state).toBe('refused');
        expect((await add('openrouter', 'bad-key-0000')).data.provider.health.state).toBe('refused');
        expect(asked.some((one) => one.host === 'openrouter.ai' && one.path === '/api/v1/key')).toBe(true);
        expect((await add('google', 'odd-key-0000')).data.provider.health.state).toBe('refused');
    });

    test('the models Ollama runs online are read without a key, and a library entry is matched by its exact size or its bare name', async () => {
        setCloudFetch(service);
        const names = await ollamaOnlineNames();
        expect([...names]).toEqual(['kimi-k3', 'embed-one']);
        expect(asked.at(-1)).toMatchObject({ host: 'ollama.com', path: '/api/tags', key: '' });
        asked.length = 0;
        await ollamaOnlineNames();
        expect(asked.length).toBe(0);

        const listed = new Set(['gpt-oss:120b', 'kimi-k3', 'deepseek-v4-pro:0813']);
        expect(runsOnline({ tag: 'gpt-oss:120b' }, listed)).toBe(true);
        expect(runsOnline({ tag: 'gpt-oss:20b' }, listed)).toBe(false);
        expect(runsOnline({ tag: 'kimi-k3', cloudOnly: true }, listed)).toBe(true);
        expect(runsOnline({ tag: 'deepseek-v4-pro', cloudOnly: true }, listed)).toBe(true);
        expect(runsOnline({ tag: 'deepseek-v4-pro' }, listed)).toBe(false);
        expect(runsOnline({ tag: 'llama3.2:3b' }, listed)).toBe(false);
    });

    test('a refused key, an empty account, no credit and a busy service are told apart, and a new key puts it right', async () => {
        const refused = (await add('openai', 'bad-key-0000')).data.provider;
        const credit = (await add('deepseek', 'poor-key-0000')).data.provider;
        const busy = (await add('groq', 'slow-key-0000')).data.provider;
        const empty = (await add('mistral', 'none-key-0000')).data.provider;
        expect([refused.health.state, credit.health.state, busy.health.state, empty.health.state]).toEqual(['refused', 'credit', 'busy', 'empty']);
        for (const one of [refused, credit, busy, empty]) expect(one.health.needs.length).toBeGreaterThan(0);
        const fixed = await client.patch(`${api}/${refused.id}`, { key: 'good-key-0000' }, any);
        expect(fixed.data.provider.health.state).toBe('healthy');
        expect((await client.post(`${api}/${credit.id}/check`, {}, any)).data.provider.health.state).toBe('credit');
    });

    test('an unknown service, a second copy of the same one and a key that is too short are refused', async () => {
        expect((await add('nowhere')).status).toBe(404);
        expect((await add('openai')).status).toBe(201);
        expect((await add('openai')).status).toBe(409);
        expect((await client.post(api, { kind: 'groq', key: 'x' }, any)).status).toBe(400);
        expect((await client.post(api, { kind: 'groq' }, any)).status).toBe(400);
    });

    test('a connected service appears among the computers only when asked for, with its models ready to chat', async () => {
        const { provider } = (await add('moonshot')).data;
        const plain = (await client.get('/api/v1/devices', any)).data.devices;
        expect(plain.some((device) => device.kind === 'cloud')).toBe(false);
        const withOnline = (await client.get('/api/v1/devices?with=online', any)).data.devices;
        const entry = withOnline.find((device) => device.id === provider.id);
        expect(entry).toMatchObject({ kind: 'cloud', name: 'Moonshot (online)', online: true, paused: false, ollama: { available: true, modelCount: 2 } });

        const models = (await client.client.get('/api/v1/models', through(provider.id))).data;
        expect(models.online).toBe(true);
        expect(models.device).toBe(provider.id);
        expect(models.models.map((model) => model.displayName)).toEqual(['Big One', 'Small One']);
        expect(models.models.map((model) => model.name)).toEqual(['big-one', 'small-one']);
        expect(models.models[0]).toMatchObject({ ready: true, canChat: true, contextLength: 200000, fit: { verdict: 'cloud', label: 'Online', fits: true } });
        expect(models.models[0].capabilities).toContain('vision');

        // The app's watcher must not move a person off an online service: it counts as ready.
        const status = (await client.client.get('/api/v1/setup/status', through(provider.id))).data;
        expect(status.ready).toBe(provider.id);
        expect(status.account.available).toBe(true);
    });

    test('a chat on an online model streams the reply, shows its thinking, counts its tokens and sends only the conversation', async () => {
        const { provider } = (await add('moonshot')).data;
        const made = await client.post('/api/v1/chats', { model: 'big-one', device: provider.id }, any);
        expect(made.status).toBe(201);
        expect(made.data.chat.device).toBe(provider.id);
        asked.length = 0;
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Say hello' }, { responseType: 'text', validateStatus: () => true });
        expect(sent.status).toBe(200);
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        expect(events.filter((event) => event.type === 'thinking').map((event) => event.text).join('')).toBe('Let me think. ');
        expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('Hello from online.');
        const done = events.find((event) => event.type === 'done');
        expect(done.reply.content).toBe('Hello from online.');
        expect(done.reply.stats.evalCount).toBe(7);
        expect(done.reply.stats.promptEvalCount).toBe(21);

        const call = asked.find((one) => one.path.endsWith('/chat/completions'));
        expect(call).toMatchObject({ host: 'api.moonshot.ai', key: 'good-key-0000' });
        expect(call.body.model).toBe('big-one');
        expect(call.body.stream).toBe(true);
        expect(call.body.messages.at(-1)).toEqual({ role: 'user', content: 'Say hello' });
        expect(Object.keys(call.body).sort()).toEqual(['messages', 'model', 'stream', 'stream_options']);
        expect(call.body.options).toBeUndefined();
        expect(call.body.keep_alive).toBeUndefined();
    });

    test('Claude is reached through its own library: a dated request, its own request shape, and its thinking shown', async () => {
        const { provider } = (await add('anthropic')).data;
        expect(provider.health.state).toBe('healthy');
        expect(provider.models).toEqual(['claude-large']);
        expect(asked[0]).toMatchObject({ host: 'api.anthropic.com', path: '/v1/models', key: 'good-key-0000' });
        expect(asked[0].version).toMatch(/^\d{4}-\d{2}-\d{2}$/);
        expect((await Provider.findById(provider.id)).models[0]).toMatchObject({ contextLength: 1000000, maxOutput: 128000, vision: true });

        const made = await client.post('/api/v1/chats', { model: 'claude-large', device: provider.id }, any);
        asked.length = 0;
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Say hello' }, { responseType: 'text', validateStatus: () => true });
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        expect(events.filter((event) => event.type === 'thinking').map((event) => event.text).join('')).toBe('Considering. ');
        expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('Hello from Claude.');
        const done = events.find((event) => event.type === 'done');
        expect(done.reply.stats).toMatchObject({ evalCount: 12, promptEvalCount: 30 });

        const call = asked.find((one) => one.path === '/v1/messages');
        expect(call.body.model).toBe('claude-large');
        expect(call.body.max_tokens).toBe(64000);
        expect(call.body.stream).toBe(true);
        expect(typeof call.body.system).toBe('string');
        expect(call.body.messages.every((message) => message.role !== 'system')).toBe(true);
        expect(call.body.messages.at(-1)).toEqual({ role: 'user', content: 'Say hello' });
        expect(call.body.temperature).toBeUndefined();

        expect((await add('anthropic', 'bad-key-0000')).status).toBe(409);
        await Provider.deleteMany({});
        expect((await add('anthropic', 'bad-key-0000')).data.provider.health.state).toBe('refused');
    });

    test('a key that stops working mid-use ends the reply with a plain reason', async () => {
        const { provider } = (await add('openai')).data;
        const made = await client.post('/api/v1/chats', { model: 'big-one', device: provider.id }, any);
        await Provider.updateOne({ _id: provider.id }, { sealed: seal({ key: 'bad-key-0000' }) });
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Hello?' }, { responseType: 'text', validateStatus: () => true });
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        const failed = events.find((event) => event.type === 'error');
        expect(failed.code).toBe('KEY_REFUSED');
        expect(failed.message).toMatch(/did not accept your key/);
    });

    test('one person cannot see, use, check, change or remove a service another person connected', async () => {
        const { provider } = (await add('openai')).data;
        signOut();
        await testStartup.loginAsUser('creator');
        expect((await client.get(api, any)).data.providers).toEqual([]);
        expect((await client.get('/api/v1/devices?with=online', any)).data.devices.some((device) => device.id === provider.id)).toBe(false);
        expect((await client.post(`${api}/${provider.id}/check`, {}, any)).status).toBe(404);
        expect((await client.patch(`${api}/${provider.id}`, { key: 'good-key-0000' }, any)).status).toBe(404);
        expect((await client.delete(`${api}/${provider.id}`, any)).status).toBe(404);
        const models = (await client.client.get('/api/v1/models', through(provider.id))).data;
        expect(models.online).toBeUndefined();
        expect((await Provider.findById(provider.id)).name).toBe('OpenAI');
    });

    test('removing a service works, and a signed-out visitor is turned away', async () => {
        const { provider } = (await add('groq')).data;
        expect((await client.delete(`${api}/${provider.id}`, any)).status).toBe(200);
        expect(await Provider.countDocuments({})).toBe(0);
        signOut();
        expect((await client.get(api, any)).status).toBe(401);
    });
});

describe('Tools in chats', () => {
    const any = { validateStatus: () => true };
    const post = (path, body, config) => client.post(`/api/v1${path}`, body, { validateStatus: () => true, ...config });
    const called = [];
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };
    const SERVER_TOOLS = [
        { name: 'find_page', title: 'Find a page', description: 'Finds a page', annotations: { readOnlyHint: true }, inputSchema: { type: 'object', properties: { words: { type: 'string' } } } },
        { name: 'write_page', description: 'Writes a page', inputSchema: { type: 'object', properties: { text: { type: 'string' } } } }
    ];
    const connector = async () => ({
        listTools: async () => ({ tools: SERVER_TOOLS }),
        callTool: async (params) => {
            called.push(params);
            if (params.arguments?.words === 'broken') return { isError: true, content: [{ type: 'text', text: 'No page has those words.' }] };
            return { content: [{ type: 'text', text: params.name === 'find_page' ? 'Found: the launch plan.' : 'Saved.' }] };
        },
        close: async () => {}
    });
    const newChat = async (body = {}) => (await post('/chats', { model: 'qwen2.5-coder:7b', ...body })).data.chat;
    const send = async (chatId, content = 'do it') => {
        const response = await post(`/chats/${chatId}/messages`, { content }, { responseType: 'text', transformResponse: [(data) => data] });
        return String(response.data).trim().split('\n').map((line) => JSON.parse(line));
    };
    const until = async (check) => {
        for (let attempt = 0; attempt < 100; attempt += 1) {
            if (await check()) return true;
            await new Promise((resolve) => setTimeout(resolve, 30));
        }
        return false;
    };
    const addServer = async () => (await post('/tools/servers', { kind: 'remote', name: 'Notes', address: 'https://notes.example/mcp' })).data.server;
    const toolTurns = (request) => request.messages.filter((message) => message.role === 'tool');

    beforeAll(() => {
        setOllamaClientFactory(fakeOllama);
        setToolConnector(connector);
    });
    afterAll(async () => {
        setToolConnector();
        await ToolServer.deleteMany({});
    });
    beforeEach(async () => {
        resetState();
        called.length = 0;
        await ToolServer.deleteMany({});
        signOut();
        await testStartup.loginAsUser('user');
    });

    test('the built-in tools work on their own: sums are exact, nothing but sums is run, and a bad time zone is said plainly', async () => {
        const tool = (id) => BUILTIN_TOOLS.find((one) => one.id === id);
        expect(await tool('calculator').run({ expression: '(12.5 * 4) / 5 ^ 2' })).toBe('2');
        expect(await tool('calculator').run({ expression: '1,000 + 1' })).toBe('1001');
        expect(await tool('calculator').run({ expression: 'process.exit()' })).toMatch(/not an arithmetic expression/);
        expect(await tool('calculator').run({ expression: '1/0' })).toMatch(/not a finite number/);
        expect(await tool('date_time').run({ timezone: 'Europe/London' })).toMatch(/\d{4}/);
        expect(await tool('date_time').run({ timezone: 'Nowhere/Place' })).toMatch(/not one I know/);
        expect(await tool('read_web_page').run({ url: 'http://plain.example' })).toMatch(/https/);
    });

    test('what a chat can switch on is listed, and the choice is kept with the chat', async () => {
        const server = await addServer();
        const listed = (await client.get('/api/v1/skills', any)).data.tools;
        expect(listed.filter((tool) => tool.kind === 'builtin').map((tool) => tool.id)).toEqual(['date_time', 'calculator', 'read_web_page']);
        expect(listed.find((tool) => tool.id === server.id)).toMatchObject({ name: 'Notes', kind: 'remote', ready: true, count: 2 });

        const chat = await newChat({ tools: ['calculator', server.id] });
        expect(chat.tools).toEqual(['calculator', server.id]);
        const changed = await client.patch(`/api/v1/chats/${chat.id}`, { tools: [] }, any);
        expect(changed.data.chat.tools).toEqual([]);
        expect((await post('/chats', { tools: ['Not Valid!'] })).status).toBe(400);
    });

    test('a switched-on built-in tool is offered, run when called, and its answer reaches the model', async () => {
        const chat = await newChat({ tools: ['calculator', 'date_time'] });
        state.toolPlan = [[['calculator', { expression: '6 * 7' }]]];
        const events = await send(chat.id, 'what is six times seven');
        expect(state.chatCalls[0].tools.map((tool) => tool.function.name)).toEqual(['date_time', 'calculator']);
        expect(state.chatCalls[0].messages[0].content).toContain('You have tools for this chat');
        expect(events.find((event) => event.type === 'tool')).toMatchObject({ tool: 'Calculator', input: '{"expression":"6 * 7"}' });
        expect(toolTurns(state.chatCalls[1])).toEqual([{ role: 'tool', tool_name: 'calculator', content: '42' }]);
        const done = events.find((event) => event.type === 'done');
        expect(done.reply.steps).toEqual([expect.objectContaining({ tool: 'Calculator', failed: null })]);
    });

    test('no tools are offered when none are switched on, or when the model cannot use them', async () => {
        const plain = await newChat();
        await send(plain.id);
        expect(state.chatCalls[0].tools).toBeUndefined();
        expect(state.chatCalls[0].messages[0].content).not.toContain('You have tools for this chat');

        resetState();
        state.noTools = true;
        const chat = await newChat({ tools: ['calculator'] });
        await send(chat.id);
        expect(state.chatCalls[0].tools).toBeUndefined();
    });

    test('a tool from a tool server is offered under its own name, called on that server, and a failure is passed on plainly', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        state.toolPlan = [[['notes_find_page', { words: 'launch' }]], [['notes_find_page', '{"words":"broken"}']]];
        const events = await send(chat.id, 'find the launch plan');
        expect(state.chatCalls[0].tools.map((tool) => tool.function.name)).toEqual(['notes_find_page', 'notes_write_page']);
        expect(called).toEqual([{ name: 'find_page', arguments: { words: 'launch' } }, { name: 'find_page', arguments: { words: 'broken' } }]);
        expect(events.filter((event) => event.type === 'tool').map((event) => event.tool)).toEqual(['Notes: Find a page', 'Notes: Find a page']);
        expect(toolTurns(state.chatCalls[1])[0].content).toBe('Found: the launch plan.');
        expect(toolTurns(state.chatCalls[2])[1].content).toBe('No page has those words.');
        expect(events.filter((event) => event.type === 'toolFailed').map((event) => event.failed)).toEqual(['failed']);
        expect(events.some((event) => event.type === 'permission')).toBe(false);
    });

    test('a tool that changes something waits for the person: allowed once it runs, refused it does not, and always stops the asking', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        const turn = async (answer) => {
            state.toolPlan = [[['notes_write_page', { text: 'hello' }]]];
            state.chatCalls = [];
            const pending = send(chat.id, 'save this');
            let asked = false;
            if (answer) {
                asked = await until(() => waitingPermissions().some((waiting) => waiting.chatId === chat.id));
                if (asked) expect((await post(`/chats/${chat.id}/permission`, { id: waitingPermissions().find((waiting) => waiting.chatId === chat.id).id, answer })).status).toBe(200);
            }
            return { events: await pending, asked };
        };

        const once = await turn('once');
        expect(once.asked).toBe(true);
        expect(once.events.find((event) => event.type === 'permission')).toMatchObject({ tool: 'Notes: write_page', input: '{"text":"hello"}' });
        expect(called).toEqual([{ name: 'write_page', arguments: { text: 'hello' } }]);

        called.length = 0;
        const refused = await turn('no');
        expect(called).toEqual([]);
        expect(toolTurns(state.chatCalls[1])[0].content).toMatch(/did not allow/);
        expect(refused.events.find((event) => event.type === 'toolFailed')).toMatchObject({ failed: 'refused' });

        await turn('always');
        expect(called.length).toBe(1);
        const after = await turn(null);
        expect(after.events.some((event) => event.type === 'permission')).toBe(false);
        expect(called.length).toBe(2);
        expect(waitingPermissions()).toEqual([]);
    });

    test('only the person whose chat it is can answer, and a question that is not waiting cannot be answered', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        state.toolPlan = [[['notes_write_page', { text: 'hello' }]]];
        const pending = send(chat.id, 'save this');
        expect(await until(() => waitingPermissions().length > 0)).toBe(true);
        const { id } = waitingPermissions()[0];
        expect((await post(`/chats/${chat.id}/permission`, { id: 'abcdef123456', answer: 'once' })).status).toBe(404);
        expect((await post(`/chats/${chat.id}/permission`, { id, answer: 'maybe' })).status).toBe(400);
        expect((await post(`/chats/${chat.id}/permission`, { id, answer: 'no' })).status).toBe(200);
        await pending;
        expect((await post(`/chats/${chat.id}/permission`, { id, answer: 'once' })).status).toBe(404);
        expect(called).toEqual([]);
    });

    test('a tool server that is not ready, or belongs to someone else, gives a chat no tools', async () => {
        const server = await addServer();
        await ToolServer.updateOne({ _id: server.id }, { health: 'unreachable' });
        const chat = await newChat({ tools: [server.id] });
        await send(chat.id);
        expect(state.chatCalls[0].tools).toBeUndefined();

        await ToolServer.updateOne({ _id: server.id }, { health: 'healthy' });
        signOut();
        await testStartup.loginAsUser('creator');
        resetState();
        const theirs = await newChat({ tools: [server.id, 'calculator'] });
        await send(theirs.id);
        expect(state.chatCalls[0].tools.map((tool) => tool.function.name)).toEqual(['calculator']);
    });
});
