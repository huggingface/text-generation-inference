import { check, randomSeed } from 'k6';
import http from 'k6/http';
import { Trend, Counter } from 'k6/metrics';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const seed = 0;

const host = __ENV.HOST;
const model_id = __ENV.MODEL_ID;
const timePerToken = new Trend('time_per_token', true);
const tokens = new Counter('tokens');
const new_tokens = new Counter('new_tokens');
const input_tokens = new Counter('input_tokens');
const max_new_tokens = 50;
const reference_latency_ms = 32;

randomSeed(seed);
// const shareGPT = JSON.parse(open("ShareGPT_V3_unfiltered_cleaned_split.json"))
const shareGPT = JSON.parse(open("small.json"))


export function get_options() {
    return {
        thresholds: {
            http_req_failed: ['rate==0'],
            // time_per_token: [{
            //     threshold: `p(50)<${5 * reference_latency_ms}`,
            //     abortOnFail: true,
            //     delayAbortEval: '10s'
            // }],
        },
        scenarios: {
            load_test: {
                executor: 'constant-arrival-rate',
                duration: '60s',
                preAllocatedVUs: 1000,
                rate: 10,
                timeUnit: '1s',
            },
        },
    };
}

function generate_payload(gpt, max_new_tokens) {
    const input = gpt["conversations"][0]["value"];
    return { "messages": [{ "role": "user", "content": input }], "temperature": 0.5, "ignore_eos": true, "model": `${model_id}`, "max_tokens": max_new_tokens }
}

export function run() {
    const headers = { 'Content-Type': 'application/json' };
    const query = randomItem(shareGPT);
    const payload = JSON.stringify(generate_payload(query, max_new_tokens));
    const res = http.post(`http://${host}/v1/chat/completions`, payload, {
        headers,
    });
    if (res.status >= 400 && res.status < 500) {
        return;
    }


    check(res, {
        'Post status is 200': (r) => res.status === 200,
    });
    const duration = res.timings.duration;

    if (res.status === 200) {
        const body = res.json();
        const n_tokens = body.usage.completion_tokens;
        const latency_ms_per_token = duration / n_tokens;
        timePerToken.add(latency_ms_per_token);
        const _input_tokens = body.usage.prompt_tokens;
        tokens.add(n_tokens + _input_tokens);
        input_tokens.add(_input_tokens);
        new_tokens.add(n_tokens);
        tokens.add(n_tokens + _input_tokens);
    }
}

export const options = get_options();

export default function() {
    run();
}
