# Developing TGI

If you're interested in contributing to TGI, you will need to have an installation that allows for quick edits and 
testing.

This part of the documentation revolves around setting up the project for you to get up and running quickly. There are
two main components of the project: the launcher and router, in rust, and the server, in Python. 

We'll first take a look at setting up a Python workspace using TGI as an *editable* installation. There are several
ways to get this done:

- Using the docker image and its content as the "interpreter" of your code, as a simple way to get started.
- Doing a manual installation of TGI, which gives more freedom in the way that you approach it

Different IDEs can be setup differently to achieve this, so we'll split this document per IDE.

## Raw - no IDE

### Using the docker image

The docker image differs according to your hardware, please refer to the following guides to see which docker image
to us ([NVIDIA](./installation_nvidia), [AMD](./installation_amd), [Gaudi](./installation_gaudi), 
[Inferentia](./installation_inferentia)). We'll refer to the docker image as `$docker_image` in the following snippets.

#### Consuming the docker image

You can consume the docker image easily with `docker run`:

```shell
docker run <docker_container_settings> $docker_image --model-id $model
```

This boots up the launcher with the docker container settings you have passed alongside the model ID. This isn't very
flexible as a debugging tool: you can switch the container settings around, but you don't have access to the code.

#### Running the docker image in interactive mode

You can be much more flexible by running the docker in [interactive mode](
https://docs.docker.com/reference/cli/docker/container/run/#interactive) using the `-i` (interactive) and `-t`
(TTY, I/O streams) flags.

You usually want to override the Dockerfile's `ENTRYPOINT` command so that you have access to the contents of the 
container straight away:

```shell
docker run -it --entrypoint=/bin/bash <docker_container_settings> $docker_image --model-id $model
```

This opens up the container for you to play around with a bash shell:

```shell
root@9103ca841d30:/usr/src#
root@47cd8a15e612:/usr/src# ls
proto  server
```

Here you have access to a few folders from the TGI library: `proto`, and `server`. You could theoretically get started
straight away by installing the contents of `server` as an editable install:
```shell
root@47cd8a15e612:/usr/src# pip install -e ./server
# [...]
# Successfully installed text-generation-server-2.0.4
```

However, it can be easier to have the code and files from TGI passed to the container as an additional volume, so
that any change outside of the container are reflected within it. This makes editing files and opening PRs much simpler
as you don't have to do all of that from within the container.

Here's an example of how it would work:

```shell
docker run -it --entrypoint=/bin/bash -v $PATH_TO_TGI/text_generation_inference:/tgi <docker_container_settings> $docker_image --model-id $model
root@47cd8a15e612:/usr/src# pip install -e /tgi/server
# [...]
# Successfully installed text-generation-server-<version>.dev0
```

This is good for quick inspection but it's recommended to setup an IDE for longer term/deeper changes.

### With a manual installation

A manual installation 

## VS Code

In order to develop on TGI on the long term, we recommend setting up an IDE like vscode.

Once again there are two ways to go about it: manual, or using docker. In order to use a manual install, we recommend
following the [section above](#With-a-manual-installation) and having VS Code point to that folder.

However, in the situation where you would like to setup VS Code to run on a local (or remote) machine using the docker 
image, you can do so by following the [Dev containers](https://code.visualstudio.com/docs/devcontainers/tutorial)
tutorial.

Here are the steps to do this right:

If using a remote machine, you should have ssh access to it:

```shell
ssh tgi-machine
```

Once you validate you have SSH access there, or if you're running locally, we recommend cloning TGI on the machine 
and launching the container with the additional volume:

```shell
$ git clone https://github.com/huggingface/text-generation-inference
$ docker run \
    -it --entrypoint=/bin/bash 
    -p 8080:80 
    --gpus all 
    -v /home/ubuntu/text-generation-inference:/tgi    
    ghcr.io/huggingface/text-generation-inference:2.0.4
```

In the container, you can install TGI through the new path as an editable install:

```shell
root@47cd8a15e612:/usr/src# pip install -e /tgi/server
# [...]
# Successfully installed text-generation-server-<version>.dev0
```

From there, an after having installed the "Dev Containers" VS Code plugin, you can attach to the running container
by doing `Cmd + Shift + P` (or `Ctrl + Shift + P` on non-MacOS devices) and running

```
>Dev Containers: Attach to Running Container
```

You should find the running container and use it as your dev container.
