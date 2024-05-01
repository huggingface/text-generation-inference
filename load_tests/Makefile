
ShareGPT_V3_unfiltered_cleaned_split.json:
	wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

prepare_share: ShareGPT_V3_unfiltered_cleaned_split.json
	python filter.py

prepare_orca:
	python orca.py
