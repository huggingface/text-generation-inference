import { check, randomSeed } from 'k6';
import http from 'k6/http';
import { Trend, Counter } from 'k6/metrics';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const seed = 0;

const host = __ENV.HOST || '127.0.0.1:8000';
const timePerToken = new Trend('time_per_token', true);
const tokens = new Counter('tokens');
const new_tokens = new Counter('new_tokens');
const input_tokens = new Counter('input_tokens');

randomSeed(seed);
// const shareGPT = JSON.parse(open("ShareGPT_V3_unfiltered_cleaned_split.json"))
const shareGPT = JSON.parse(open("small.json"))


export function get_options(reference_latency_ms){
    return {
        thresholds: {
            http_req_failed: ['rate==0'],
            time_per_token: [{
                threshold: `p(50)<${5 * reference_latency_ms}`,
                abortOnFail: true,
                delayAbortEval: '10s'
            }],
        },
        scenarios: {
            load_test: {
                executor: 'constant-arrival-rate',
                duration: '60s',
                preAllocatedVUs: 10,
                rate: 10,
                timeUnit: '1s',
            },
        },
    };
}


export function run(host, generate_payload, max_new_tokens) {
    const headers = {'Content-Type': 'application/json'};
    const query = randomItem(shareGPT);
    const payload = JSON.stringify(generate_payload(query));
    const res = http.post(`http://${host}/generate`, payload, {
        headers,
    });
    if(res.status >= 400 && res.status < 500){
        return;
    }


    check(res, {
        'Post status is 200': (r) => res.status === 200,
    });
    const duration = res.timings.duration;

    if (res.status === 200) {
        const body = res.json();
        const n_tokens = body.details.tokens.length;
        const latency_ms_per_token = duration / n_tokens;
        timePerToken.add(latency_ms_per_token);
        const latency_in_s = latency_ms_per_token / 1000;
        const individual_throughput = 1 / latency_in_s;
        const _input_tokens = body.details.prefill.length;
        tokens.add(n_tokens + _input_tokens);
        input_tokens.add(_input_tokens);
        new_tokens.add(n_tokens);
    }
}
