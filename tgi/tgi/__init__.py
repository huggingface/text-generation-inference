import threading
from tgi import _tgi
import asyncio
from dataclasses import dataclass, asdict
from text_generation_server.cli import app

# add the rust_launcher coroutine to the __all__ list
__doc__ = _tgi.__doc__
if hasattr(_tgi, "__all__"):
    __all__ = _tgi.__all__


def text_generation_server_cli_main():
    app()


def text_generation_router_cli_main():
    _tgi.rust_router()


def text_generation_launcher_cli_main():
    _tgi.rust_launcher_cli()


@dataclass
class Args:
    model_id = "google/gemma-2b-it"
    revision = None
    validation_workers = 2
    sharded = None
    num_shard = None
    quantize = None
    speculate = None
    dtype = None
    trust_remote_code = True
    max_concurrent_requests = 128
    max_best_of = 2
    max_stop_sequences = 4
    max_top_n_tokens = 5
    max_input_tokens = None
    max_input_length = None
    max_total_tokens = None
    waiting_served_ratio = 0.3
    max_batch_prefill_tokens = None
    max_batch_total_tokens = None
    max_waiting_tokens = 20
    max_batch_size = None
    cuda_graphs = None
    hostname = "0.0.0.0"
    port = 3000
    shard_uds_path = "/tmp/text-generation-server"
    master_addr = "localhost"
    master_port = 29500
    huggingface_hub_cache = None
    weights_cache_override = None
    disable_custom_kernels = False
    cuda_memory_fraction = 1.0
    rope_scaling = None
    rope_factor = None
    json_output = False
    otlp_endpoint = None
    cors_allow_origin = []
    watermark_gamma = None
    watermark_delta = None
    ngrok = False
    ngrok_authtoken = None
    ngrok_edge = None
    tokenizer_config_path = None
    disable_grammar_support = False
    env = False
    max_client_batch_size = 4


class TGI(object):
    # only allow a limited set of arguments for now
    def __init__(self, model_id=None):
        app_args = Args()
        if model_id:
            app_args.model_id = model_id

        print(asdict(app_args))
        self.thread = threading.Thread(target=self.run, args=(asdict(app_args),))
        self.thread.start()

    async def runit(self, args: dict):
        print(args)
        args = Args(**args)
        try:
            await _tgi.rust_launcher(
                args.model_id,
                args.revision,
                args.validation_workers,
                args.sharded,
                args.num_shard,
                args.quantize,
                args.speculate,
                args.dtype,
                args.trust_remote_code,
                args.max_concurrent_requests,
                args.max_best_of,
                args.max_stop_sequences,
                args.max_top_n_tokens,
                args.max_input_tokens,
                args.max_input_length,
                args.max_total_tokens,
                args.waiting_served_ratio,
                args.max_batch_prefill_tokens,
                args.max_batch_total_tokens,
                args.max_waiting_tokens,
                args.max_batch_size,
                args.cuda_graphs,
                args.hostname,
                args.port,
                args.shard_uds_path,
                args.master_addr,
                args.master_port,
                args.huggingface_hub_cache,
                args.weights_cache_override,
                args.disable_custom_kernels,
                args.cuda_memory_fraction,
                args.rope_scaling,
                args.rope_factor,
                args.json_output,
                args.otlp_endpoint,
                args.cors_allow_origin,
                args.watermark_gamma,
                args.watermark_delta,
                args.ngrok,
                args.ngrok_authtoken,
                args.ngrok_edge,
                args.tokenizer_config_path,
                args.disable_grammar_support,
                args.env,
                args.max_client_batch_size,
            )
        except Exception as e:
            print(e)

    def run(self, args: dict):
        asyncio.run(self.runit(args))

    def close(self):
        self.thread.join()
