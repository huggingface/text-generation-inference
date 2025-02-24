# Initialize base variables
SHELL := /bin/bash
pkg_name := text_generation_server
BUILDDIR ?= $(CURDIR)/build
VERSION ?= 0.0.1
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
pkg_dir := $(BUILDDIR)/$(pkg_name)
py_version := $(subst -,.,${VERSION})
pkg_dist := ${BUILDDIR}/dist/${pkg_name}-$(py_version).tar.gz

clean:
	rm -rf $(BUILDDIR)/*

${BUILDDIR}:
	install -d $@

# List static sources to be deployed in the package
src_dir := $(mkfile_dir)/$(pkg_name)
sources := $(wildcard $(src_dir)/*.py)
deployed_sources := $(subst $(src_dir), $(pkg_dir), $(sources))

# Static files are just copied

define COPY
	cp -f $< $@
endef

# We use a PHONY target to represent the VERSION
.PHONY: VERSION

VERSION: ${BUILDDIR}
	# The trick is to compare the value of the variable with the content of a file in the build directory
	@if [[ `cat ${BUILDDIR}/VERSION 2>&1` != '$(VERSION)' ]]; then echo -n $(VERSION) >${BUILDDIR}/VERSION; fi

# Depending on the PHONY VERSION target makes sure the pyproject.toml is regenerated if the version changes
$(BUILDDIR)/pyproject.toml: $(mkfile_dir)/pyproject.toml VERSION
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

install: ${pkg_dist}
	python3 -m pip uninstall -y ${pkg_name}
	python3 -m pip install ${pkg_dist}
