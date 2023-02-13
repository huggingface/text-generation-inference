import http from 'k6/http';
import {check, sleep} from 'k6';

export const options = {
    stages: [
        {duration: '1m', target: 50},
        {duration: '2m', target: 100},
        {duration: '1m', target: 0},
    ],
    hosts: {
        'text-generation-inference.huggingface.co': '127.0.0.1:3000',
    },
};
const SLEEP_DURATION = 1;

function greedy_example(inputs, max_new_tokens, name) {
    let body = JSON.stringify({
        inputs: inputs,
        parameters: {
            max_new_tokens: max_new_tokens,
            do_sample: false,
        }
    });
    let params = {
        headers: {
            'Content-Type': 'application/json',
        },
        tags: {
            name: name
        }
    };
    return http.post('http://text-generation-inference.huggingface.co/generate', body, params);
}

function sample_example(inputs, max_new_tokens, name) {
    let body = JSON.stringify({
        inputs: inputs,
        parameters: {
            max_new_tokens: max_new_tokens,
            do_sample: true,
            top_p: 0.9,
            seed: 0
        }
    });
    let params = {
        headers: {
            'Content-Type': 'application/json',
        },
        tags: {
            name: name
        }
    };
    return http.post('http://text-generation-inference.huggingface.co/generate', body, params);
}

export default function () {
    const response_1 = sample_example('A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:', 32, 'example-1');
    check(response_1, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);

    const response_2 = sample_example("A poem about the beauty of science by Alfred Edgar Brittle\\nTitle: The Magic Craft\\nIn the old times", 50, "example-2");
    check(response_2, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);

    const response_3 = greedy_example("استخراج العدد العاملي في لغة بايثون: ", 30, "example-3");
    check(response_3, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);

    const response_4 = sample_example("Pour déguster un ortolan, il faut tout d'abord", 32, "example-4");
    check(response_4, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);

    const response_5 = sample_example("Traduce español de España a español de Argentina\nEl coche es rojo - el auto es rojo\nEl ordenador es nuevo - la computadora es nueva\nel boligrafo es negro -", 16, "example-5");
    check(response_5, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);

    const response_6 = sample_example("Question: If I put cheese into the fridge, will it melt?\nAnswer:", 32, "example-6");
    check(response_6, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);

    const response_7 = greedy_example("Question: Where does the Greek Goddess Persephone spend half of the year when she is not with her mother?\nAnswer:", 24, "example-7");
    check(response_7, {
        'is status 200': (r) => r.status === 200,
    });
    sleep(SLEEP_DURATION);
}