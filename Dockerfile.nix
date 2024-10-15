# Build the image and get out the docker file:
#
# docker build -t tgi-nix-builder -f Dockerfile.nix
# docker run --log-driver=none tgi-nix-builder | docker load

FROM nixos/nix:2.18.8 AS builder
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf
RUN nix profile install nixpkgs#cachix
RUN cachix use text-generation-inference
WORKDIR /root
ADD . .
RUN nix build .
RUN mkdir /tmp/nix-store-closure
RUN cp -R $(nix-store -qR result/) /tmp/nix-store-closure

FROM ubuntu:24.04

WORKDIR /app

# Copy /nix/store
COPY --from=builder /tmp/nix-store-closure /nix/store
COPY --from=builder /root/result /app
RUN ldconfig
CMD ["ldconfig", "/app/bin/text-generation-launcher"]
