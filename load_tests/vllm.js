import { get_options, run } from "./common.js";

const reference_latency_ms = 22;
const host = __ENV.HOST || '127.0.0.1:8000';
const max_new_tokens = 50;


function generate_payload(gpt){
    const input = gpt["conversations"][0]["value"];
    return {"prompt": input, "temperature": 0.5, "ignore_eos": true}
}

export const options = get_options(reference_latency_ms);

export default function(){
    run(host, generate_payload, max_new_tokens);
}
