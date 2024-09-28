# Build the image and get out the docker file:
#
# docker build -t tgi-nix -f Dockerfile.nix
# docker run --rm --volume $PWD/data:/data tgi-nix cp -H result /data/tgi-docker.tar.gz
# docker load < tgi-docker.tar.gz

FROM nixos/nix:2.18.8
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf
RUN nix profile install nixpkgs#cachix
RUN cachix use text-generation-inference
WORKDIR /root
ADD . .
RUN nix build .#dockerImage
