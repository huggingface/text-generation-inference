# Monitoring TGI server with Prometheus and Grafana dashboard

TGI server deployment can easily be monitored through a Grafana dashboard, consuming a Prometheus data collection. Example of inspectable metrics are statistics on the effective batch sizes used by TGI, prefill/decode latencies, number of generated tokens, etc.

In this tutorial, we look at how to set up a local Grafana dashboard to monitor TGI usage.

![Grafana dashboard for TGI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/grafana.png)

## Setup on the server machine

First, on your server machine, TGI needs to be launched as usual. TGI exposes [multiple](https://github.com/huggingface/text-generation-inference/discussions/1127#discussioncomment-7240527) metrics that can be collected by Prometheus monitoring server.

In the rest of this tutorial, we assume that TGI was launched through Docker with `--network host`.

On the server where TGI is hosted, a Prometheus server needs to be installed and launched. To do so, please follow [Prometheus installation instructions](https://prometheus.io/download/#prometheus). For example, at the time of writing on a Linux machine:

```
wget https://github.com/prometheus/prometheus/releases/download/v2.52.0/prometheus-2.52.0.linux-amd64.tar.gz
tar -xvzf prometheus-2.52.0.linux-amd64.tar.gz
cd prometheus
```

Prometheus needs to be configured to listen on TGI's port. To do so, in Prometheus configuration file `prometheus.yml`, one needs to edit the lines:
```
    static_configs:
      - targets: ["0.0.0.0:80"]
```
to use the correct IP address and port.

We suggest to try `curl 0.0.0.0:80/generate -X POST -d '{"inputs":"hey chatbot, how are","parameters":{"max_new_tokens":15}}' -H 'Content-Type: application/json'` on the server side to make sure to configure the correct IP and port.

Once Prometheus is configured, Prometheus server can be launched on the same machine where TGI is launched:
```
./prometheus --config.file="prometheus.yml"
```

In this guide, Prometheus monitoring data will be consumed on a local computer. Hence, we need to forward Prometheus port (by default 9090) to the local computer. To do so, we can for example:
* Use ssh [local port forwarding](https://www.ssh.com/academy/ssh/tunneling-example)
* Use ngrok port tunneling

For simplicity, we will use [Ngrok](https://ngrok.com/docs/) in this guide to tunnel Prometheus port from the TGI server to the outside word.

For that, you should follow the steps at https://dashboard.ngrok.com/get-started/setup/linux, and once Ngrok is installed, use:
```bash
ngrok http http://0.0.0.0:9090
```

As a sanity check, one can make sure that Prometheus server can be accessed at the URL given by Ngrok (in the style of https://d661-4-223-164-145.ngrok-free.app) from a local machine.

## Setup on the monitoring machine

Monitoring is typically done on an other machine than the server one. We use a Grafana dashboard to monitor TGI's server usage.

Two options are available:
* Use Grafana Cloud for an hosted dashboard solution (https://grafana.com/products/cloud/).
* Self-host a grafana dashboard.

In this tutorial, for simplicity, we will self host the dashbard. We recommend installing Grafana Open-source edition following [the official install instructions](https://grafana.com/grafana/download?platform=linux&edition=oss), using the available Linux binaries. For example:

```bash
wget https://dl.grafana.com/oss/release/grafana-11.0.0.linux-amd64.tar.gz
tar -zxvf grafana-11.0.0.linux-amd64.tar.gz
cd grafana-11.0.0
./bin/grafana-server
```

Once the Grafana server is launched, the Grafana interface is available at http://localhost:3000. One needs to log in with the `admin` username and `admin` password.

Once logged in, the Prometheus data source for Grafana needs to be configured, in the option `Add your first data source`. There, a Prometheus data source needs to be added with the Ngrok address we got earlier, that exposes Prometheus port (example: https://d661-4-223-164-145.ngrok-free.app).

Once Prometheus data source is configured, we can finally create our dashboard! From home, go to `Create your first dashboard` and then `Import dashboard`. There, we will use the recommended dashboard template [tgi_grafana.json](https://github.com/huggingface/text-generation-inference/blob/main/assets/grafana.json) for a dashboard ready to be used, but you may configure your own dashboard as you like.

Community contributed dashboard templates are also available, for example [here](https://grafana.com/grafana/dashboards/19831-text-generation-inference-dashboard/) or [here](https://grafana.com/grafana/dashboards/20246-text-generation-inference/).

Load your dashboard configuration, and your TGI dashboard should be ready to go!
