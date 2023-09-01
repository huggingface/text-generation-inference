Install:

```bash
pip3 install deepsparse-nightly[transformer] fastapi uvicorn
```

Download model:

```bash
sparsezoo.download zoo:nlg/text_generation/codegen_multi-350m/pytorch/huggingface/bigquery_thepile/base_quant-none --save-dir codegen-quant
```