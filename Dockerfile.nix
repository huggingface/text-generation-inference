# Build the image and get out the docker file:
#
# docker build -t tgi-nix-builder -f Dockerfile.nix
# docker run --log-driver=none tgi-nix-builder | docker load

FROM nixos/nix:2.18.8
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf
RUN nix profile install nixpkgs#cachix
RUN cachix use text-generation-inference
WORKDIR /root
ADD . .
RUN nix build .#dockerImageStreamed
ENTRYPOINT ./result
