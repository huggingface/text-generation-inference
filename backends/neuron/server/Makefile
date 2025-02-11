# Initialize base variables
pkg_name := text_generation_server
BUILDDIR ?= $(CURDIR)/build
VERSION ?= 0.0.1
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
pkg_dir := $(BUILDDIR)/$(pkg_name)
pkg_dist := ${BUILDDIR}/dist/${pkg_name}-${VERSION}.tar.gz

clean:
	rm -rf $(BUILDDIR)/*

# List static sources to be deployed in the package
src_dir := $(mkfile_dir)/$(pkg_name)
sources := $(wildcard $(src_dir)/*.py)
deployed_sources := $(subst $(src_dir), $(pkg_dir), $(sources))

# Static files are just copied

define COPY
	cp -f $< $@
endef

$(BUILDDIR)/pyproject.toml: $(mkfile_dir)/pyproject.toml
	mkdir -p $(BUILDDIR)
	$(COPY)
	sed -i -e 's/version = "VERSION"/version = \"${VERSION}\"/' $@

$(pkg_dir)/%.py: $(src_dir)/%.py
	mkdir -p $(pkg_dir)
	$(COPY)

# Generated files are produced by grpcio tools

# If not provided, get local proto files
ifndef PROTODIR
PROTODIR := $(mkfile_dir)/../../../proto
endif

# Three python files are generated for each protobuf
protobufs := $(PROTODIR)/generate.proto
pkg_pb_dir := $(pkg_dir)/pb
generated_sources_base := $(foreach proto, $(protobufs), $(proto:.proto=_pb2.py))
generated_sources := $(subst $(PROTODIR), $(pkg_pb_dir), $(generated_sources_base))
generated_sources += $(subst $(PROTODIR), $(pkg_pb_dir), $(generated_sources_base:.py=.pyi))
generated_sources += $(subst $(PROTODIR), $(pkg_pb_dir), $(generated_sources_base:.py=_grpc.py))

$(pkg_pb_dir)/%_pb2.py $(pkg_pb_dir)/%_pb2.pyi $(pkg_pb_dir)/%_pb2_grpc.py: $(PROTODIR)/%.proto
	mkdir -p $(pkg_pb_dir)
	python -m grpc_tools.protoc -I$(PROTODIR) --python_out=$(pkg_pb_dir) \
		--grpc_python_out=$(pkg_pb_dir) --mypy_out=$(pkg_pb_dir) $^
	sed -i -e 's/^\(import.*pb2\)/from . \1/g' $(pkg_pb_dir)/$*_pb2_grpc.py

${pkg_dist}: $(BUILDDIR)/pyproject.toml $(deployed_sources) $(generated_sources)
	python -m build $(BUILDDIR)

package: ${pkg_dist}
