.ONESHELL:
.PHONY: test deps download build clean docker

# mlpack version to use.
MLPACK_VERSION?=3.3.2

# Go version to use when building Docker image
GOVERSION?=1.13.1

# Temporary directory to put files into.
TMP_DIR?=/tmp/

# Package list for each well-known Linux distribution
RPMS = cmake curl git unzip boost-devel boost-test boost-program-options         \
       boost-math
DEBS = unzip build-essential cmake curl git pkg-config libboost-math-dev         \
       libboost-program-options-dev libboost-test-dev libboost-serialization-dev

# Detect Linux distribution
UNAME_S := $(shell uname -s)
distro_deps=
ifneq ($(shell which dnf 2>/dev/null),)
	distro_deps=deps_fedora
else
ifneq ($(shell which apt-get 2>/dev/null),)
	distro_deps=deps_debian
else
ifneq ($(shell which yum 2>/dev/null),)
	distro_deps=deps_rh_centos
else
ifeq ($(UNAME_S),Darwin)
	distro_deps=deps_darwin
endif
endif
endif
endif

# Install all necessary dependencies.
deps: $(distro_deps)

deps_darwin:
	brew install cmake curl git unzip openblas armadillo boost

deps_rh_centos:
	sudo yum -y install pkgconfig $(RPMS)

deps_fedora:
	sudo dnf -y install pkgconf-pkg-config $(RPMS)

deps_debian:
	sudo apt-get -y update
	sudo apt-get -y install $(DEBS)

# Download mlpack source.
download:
	rm -rf $(TMP_DIR)mlpack
	mkdir $(TMP_DIR)mlpack
	cd $(TMP_DIR)mlpack
	curl -Lo mlpack.zip https://www.mlpack.org/files/mlpack-$(MLPACK_VERSION).tar.gz
	tar -xvzpf mlpack.zip
	rm mlpack.zip
	cd -

# Build mlpack(go shared libraries).
build:
	cd $(TMP_DIR)mlpack/mlpack-$(MLPACK_VERSION)
	mkdir build
	cd build
	cmake -D BUILD_TESTS=OFF           \
	      -D BUILD_JULIA_BINDINGS=OFF  \
	      -D BUILD_PYTHON_BINDINGS=OFF \
	      -D BUILD_CLI_EXECUTABLES=OFF \
	      -D BUILD_GO_BINDINGS=OFF     \
	      -D BUILD_GO_SHLIB=ON  ..
	$(MAKE) -j $(shell nproc --all)
	$(MAKE) preinstall
	cd -

# Cleanup temporary build files.
clean:
	go clean --cache
	rm -rf $(TMP_DIR)mlpack

# Do everything.
install:
	deps download build sudo_install clean test


# Install system wide.
sudo_install:
	cd $(TMP_DIR)mlpack/mlpack-$(MLPACK_VERSION)/build
	sudo $(MAKE) install
	sudo ldconfig
	cd -
# Runs tests.
test:
	go test -v . ./tests

docker:
	docker build --build-arg GOVERSION=$(GOVERSION) --build-arg MLPACK_VERSION=$(MLPACK_VERSION) .