import { check, randomSeed } from 'k6';
import http from 'k6/http';
import { Trend, Counter } from 'k6/metrics';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const seed = 0;

const host = __ENV.HOST || '127.0.0.1:8000';
const timePerToken = new Trend('time_per_token', true);
const throughput = new Counter('tokens_per_s');

randomSeed(seed);
// const shareGPT = JSON.parse(open("ShareGPT_V3_unfiltered_cleaned_split.json"))
const shareGPT = JSON.parse(open("small.json"))


export function get_options(reference_latency_ms){
    return {
        thresholds: {
            http_req_failed: ['rate==0'],
            time_per_token: [{
                threshold: `p(50)<${3 * reference_latency_ms}`,
                abortOnFail: true,
                delayAbortEval: '10s'
            }],
        },
        scenarios: {
            load_test: {
                executor: 'constant-arrival-rate',
                duration: '60s',
                preAllocatedVUs: 100,
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
    const n_tokens = max_new_tokens;
    const timings = res.timings.duration;

    if (res.status === 200) {
        const latency_ms_per_token = timings / n_tokens;
        timePerToken.add(latency_ms_per_token);
        const latency_in_s = latency_ms_per_token / 1000;
        const individual_throughput = 1 / latency_in_s;
        throughput.add(individual_throughput);
    }
}
