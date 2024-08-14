import { check, sleep } from 'k6';
import { scenario } from 'k6/execution';
import http from 'k6/http';
import { Trend, Counter } from 'k6/metrics';

const host = __ENV.HOST;
const model_id = __ENV.MODEL_ID;
const timePerToken = new Trend('time_per_token', true);
const tokens = new Counter('tokens');
const new_tokens = new Counter('new_tokens');
const input_tokens = new Counter('input_tokens');
const max_new_tokens = 50;

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
            // single_user: {
            //     executor: 'constant-arrival-rate',
            //     duration: '60s',
            //     preAllocatedVUs: 1,
            //     rate: 20,
            //     timeUnit: '1s',
            // },
            // load_test: {
            //     executor: 'constant-arrival-rate',
            //     duration: '60s',
            //     preAllocatedVUs: 10, // not enough RAM for 100 VUs
            //     rate: 1,
            //     timeUnit: '1s',
            // },
            // breakpoint: {
            //     executor: 'ramping-arrival-rate', //Assure load increase if the system slows
            //     preAllocatedVUs: 300,
            //     stages: [
            //         { duration: '60s', target: 100 }, // just slowly ramp-up to a HUGE load
            //     ],
            // },
            throughput: {
                executor: 'shared-iterations',
                vus: 16,
                iterations: 500,
                maxDuration: '400s',
            },
        },
    };
}

function generate_payload(gpt, max_new_tokens) {
    const max_input_tokens = 10000 - max_new_tokens - 1;
    const input = gpt["conversations"][0]["value"].substring(0, max_input_tokens);
    return {
        "prompt": `<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "model": `${model_id}`,
    }
}

export const options = get_options();

export default function run() {
    const headers = { 'Content-Type': 'application/json' };
    const query = shareGPT[scenario.iterationInTest % shareGPT.length];
    const payload = JSON.stringify(generate_payload(query, max_new_tokens));
    const res = http.post(`https://${host}/v1/completions`, payload, {
        headers,
    });
    // if (res.status >= 400 && res.status < 500) {
    //     return;
    // }

    if (res.status !== 200) {
        console.error(res.body);
        console.error('status: ' + res.status);
        return;
    }

    check(res, {
        'Post status is 200': (res) => res.status === 200,
    });
    const duration = res.timings.duration;

    if (res.status === 200) {
        const body = res.json();
        if (body.usage) {
            const completion_tokens = body.usage.completion_tokens;
            const latency_ms_per_token = duration / completion_tokens;
            timePerToken.add(latency_ms_per_token);
            const prompt_tokens = body.usage.prompt_tokens;
            input_tokens.add(prompt_tokens);
            new_tokens.add(completion_tokens);
            tokens.add(completion_tokens + prompt_tokens);
        }
        if (body.tokens_predicted) {
            // llama.cpp specific
            const completion_tokens = body.tokens_predicted;
            const latency_ms_per_token = duration / completion_tokens;
            timePerToken.add(latency_ms_per_token);
            const prompt_tokens = body.tokens_evaluated;
            input_tokens.add(prompt_tokens);
            new_tokens.add(completion_tokens);
            tokens.add(completion_tokens + prompt_tokens);
        }
    }

    sleep(1);
}
