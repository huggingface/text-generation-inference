# Model safety.

[Pytorch uses pickle](https://pytorch.org/docs/master/generated/torch.load.html) by default meaning that for quite a long while
*Every* model using that format is potentially executing unintended code while purely loading the model.

There is a big red warning on Python's page for pickle [link](https://docs.python.org/3/library/pickle.html) but for quite a while
this was ignored by the community. Now that AI/ML is getting used much more ubiquitously we need to switch away from this format.

HuggingFace is leading the effort here by creating a new format which contains pure data ([safetensors](https://github.com/huggingface/safetensors))
and moving slowly but surely all the libs to make use of it by default.
The move is intentionnally slow in order to make breaking changes as little impact as possible on users throughout.


# TGI 2.0

Since the release of TGI 2.0, we take the opportunity of this major version increase to break backward compatibility for these pytorch
models (since they are a huge security risk for anyone deploying them).


From now on, TGI will not convert automatically pickle files without having `--trust-remote-code` flag or `TRUST_REMOTE_CODE=true` in the environment variables.
This flag is already used for community defined inference code, and is therefore quite representative of the level of confidence you are giving the model providers.


If you want to use a model that uses pickle, but you still do not want to trust the authors entirely we recommend making a convertion on our space made for that.

https://huggingface.co/spaces/safetensors/convert

This space will create a PR on the original model, which you are use directly regardless of merge status from the original authors. Just use
```
docker run .... --revision refs/pr/#ID # Or use REVISION=refs/pr/#ID in the environment
```
