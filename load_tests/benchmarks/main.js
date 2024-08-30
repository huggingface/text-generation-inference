import {check, fail} from 'k6';
import sse from "k6/x/sse"
import {scenario} from 'k6/execution';
import http from 'k6/http';
import {Trend, Counter} from 'k6/metrics';

const host = "127.0.0.1:8080";
const model_id = "Qwen/Qwen2-72B";

const endToEndLatency = new Trend('end_to_end_latency', true);
const requestThroughput = new Counter('request_throughput');
const tokenThroughput = new Counter('tokens_throughput');

const timeToFirstToken = new Trend('time_to_first_token', true);
const interTokenLatency = new Trend('inter_token_latency', true); // is microseconds

const tokensReceived = new Trend('tokens_received');

if (__ENV.MAX_NEW_TOKENS === undefined) {
    throw new Error('MAX_NEW_TOKENS must be defined');
}
const max_new_tokens = parseInt(__ENV.MAX_NEW_TOKENS)
const input_filename = __ENV.INPUT_FILENAME;
if (input_filename === undefined) {
    throw new Error('INPUT_FILENAME must be defined');

}

const shareGPT = JSON.parse(open(input_filename))

export function handleSummary(data) {
    return {
        'summary.json': JSON.stringify(data),
    };
}

function generate_payload(gpt, max_new_tokens) {
    let input = gpt["message"];
    return {
        "messages": [{"role": "user", "content": input}],
        "temperature": 0,
        "model": `${model_id}`,
        "max_tokens": max_new_tokens,
        "stream": true
    };
}

export const options = get_options();

export default function run() {
    const headers = {'Content-Type': 'application/json'};
    const query = shareGPT[scenario.iterationInTest % shareGPT.length];
    const payload = JSON.stringify(generate_payload(query, max_new_tokens));
    const url = `http://${host}/v1/chat/completions`;
    const params = {
        method: 'POST',
        body: payload,
        headers
    }

    const startTime = Date.now();
    let firstTokenTime = null;
    let lastTokenTime = null;
    let tokensCount = 0;
    let response = ""

    const res = sse.open(url, params, function (client) {
        client.on('event', function (event) {
            if (parseInt(event.id) === 4) {
                client.close()
            }
            if (event.data.includes("[DONE]") || event.data === "") {
                return
            }
            try {
                const data = JSON.parse(event.data);
                if (!'choices' in data) {
                    fail('http_200')
                    return;
                }
                const content = data['choices'][0]['delta']['content']
                if (content !== undefined) {
                    response += data['choices'][0]['delta']['content']
                    tokensCount += 1;
                }

                // Measure time to first token
                if (!firstTokenTime) {
                    firstTokenTime = Date.now();
                    timeToFirstToken.add(firstTokenTime - startTime);
                }

                // Measure inter-token latency
                const currentTime = Date.now();
                if (lastTokenTime) {
                    interTokenLatency.add((currentTime - lastTokenTime) * 1000.);
                }
                lastTokenTime = currentTime;

                if ('finish_reason' in data['choices'][0]) {
                    if (data['choices'][0]['finish_reason'] != null) {
                        const endTime = Date.now();
                        const deltaMs = endTime - startTime;
                        endToEndLatency.add(deltaMs)
                        requestThroughput.add(1);
                        tokenThroughput.add(tokensCount);
                        tokensReceived.add(tokensCount);
                    }
                }
            } catch (e) {
                // catch any errors that occur during the event processing
                // increase the fail count of the 'http_200' check
                check(true, {
                    'http_200': (val) => false,
                })
                fail('http_200')
            }
        })

        client.on('error', function (e) {
            console.log('An unexpected error occurred: ', e.error())
        })
    })

    if (tokensCount === 0) {
        // something went wrong with generation
        fail('http_200')
    }

    if (res.status >= 400 && res.status < 500) {
        return;
    }

    check(res, {
        'http_200': (res) => res.status === 200,
    });

}

export function get_options() {
    const test_type = __ENV.TEST_TYPE;
    if (test_type === undefined) {
        throw new Error('TEST_TYPE must be defined');
    }
    switch (test_type) {
        case 'constant_arrival_rate':
            return get_constant_arrival_rate_options();
        case 'constant_vus':
            return get_constant_vus_options();
        default:
            throw new Error('Invalid test type');
    }
}

function get_constant_arrival_rate_options() {
    const duration = __ENV.DURATION;
    if (duration === undefined) {
        throw new Error('DURATION must be defined');
    }
    if (__ENV.PRE_ALLOCATED_VUS === undefined) {
        throw new Error('PRE_ALLOCATED_VUS must be defined');
    }
    const pre_allocated_vus = parseInt(__ENV.PRE_ALLOCATED_VUS);
    if (__ENV.RATE === undefined) {
        throw new Error('RATE must be defined');
    }
    const rate = parseInt(__ENV.RATE);
    return {
        scenarios: {
            load_test: {
                executor: 'constant-arrival-rate',
                gracefulStop: '0s',
                duration: duration,
                preAllocatedVUs: pre_allocated_vus,
                rate: rate,
                timeUnit: '1s',
            },
        },
    }
        ;
}

function get_constant_vus_options() {
    const duration = __ENV.DURATION;
    if (duration === undefined) {
        throw new Error('DURATION must be defined');
    }
    if (__ENV.VUS === undefined) {
        throw new Error('VUS must be defined');
    }
    const vus = parseInt(__ENV.VUS);
    return {
        scenarios: {
            load_test: {
                executor: 'constant-vus',
                gracefulStop: '0s',
                duration: duration,
                vus: vus,
            },
        },
    };
}
