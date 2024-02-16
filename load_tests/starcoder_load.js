import {check} from 'k6';
import http from 'k6/http';
import {Trend} from 'k6/metrics';

const host = __ENV.HOST || '127.0.0.1:3000';

const totalTime = new Trend('total_time', true);
const validationTime = new Trend('validation_time', true);
const queueTime = new Trend('queue_time', true);
const inferenceTime = new Trend('inference_time', true);
const timePerToken = new Trend('time_per_token', true);

const example = {
    payload: JSON.stringify({
        inputs: '# This is a fibonacci function written in the Python programming language.' +
            'def fibonacci',
        parameters: {
            details: true,
            max_new_tokens: 60,
            temperature: 0.2,
            top_p: 0.95,
            seed: 0,
        },
    }),
    generated_tokens: 60
};

export const options = {
    thresholds: {
        http_req_failed: ['rate==0'],
        time_per_token: ['p(95)<90'],
        queue_time: ['p(95)<1500'],
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

export default function () {
    const headers = {'Content-Type': 'application/json'};
    const res = http.post(`http://${host}/generate`, example.payload, {
        headers,
    });

    check(res, {
        'Post status is 200': (r) => res.status === 200,
        'Post response generated tokens': (r) => res.status === 200 && res.json().details.generated_tokens === example.generated_tokens,
    });

    if (res.status === 200) {
        totalTime.add(res.headers["X-Total-Time"]);
        validationTime.add(res.headers["X-Validation-Time"]);
        queueTime.add(res.headers["X-Queue-Time"]);
        inferenceTime.add(res.headers["X-Inference-Time"]);
        timePerToken.add(res.headers["X-Time-Per-Token"]);
    }
}
