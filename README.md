### Local install

You can also opt to install `text-generation-inference` locally.

First [install Rust](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install conda:

```bash
curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
sudo install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" | sudo tee -a /etc/apt/sources.list.d/conda.list
sudo apt update && sudo apt install conda -y
source /opt/conda/etc/profile.d/conda.sh
conda -V
```

Create Env:

```shell
conda create -n dscb python=3.9 
conda activate dscb
```

Install PROTOC
```shell
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

You might need to install these:
```shell
sudo apt-get install libssl-dev gcc -y
sudo apt-get install pkg-config
```

Install DeepSparse:
```shell
pip install deepsparse-nightly[transformers]
```

Install Server
```shell
make install-server
```

Launch Server
```shell
python3 server/text_generation_server/cli.py download-weights bigscience/bloom-560m
python3 server/text_generation_server/cli.py serve bigscience/bloom-560m
```

Launch Router
```shell
make router-dev
```