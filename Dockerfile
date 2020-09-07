FROM debian:stretch

LABEL maintainer="mlpack-git@lists.mlpack.org"

# Steps to reduce image size.
RUN apt-get update  && apt-get install -y aptitude && apt-get purge -y \
    $(aptitude search '~i!~M!~prequired!~pimportant!~R~prequired! \
    ~R~R~prequired!~R~pimportant!~R~R~pimportant!busybox!grub!initramfs-tools' \
    | awk '{print $2}') && apt-get purge -y aptitude && \
    apt-get autoremove -y && apt-get clean && rm -rf /usr/share/man/?? && \
    rm -rf /usr/share/man/??_*

# Installing dependencies required to run mlpack.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends sudo \
    unzip build-essential cmake git pkg-config \
    curl ca-certificates libcurl4-openssl-dev libssl-dev \
    libboost-math-dev libboost-test-dev libboost-serialization-dev libarmadillo-dev && \
    apt-get clean && rm -rf /usr/share/man/?? && rm -rf /usr/share/man/??_* && \
    rm -rf /var/lib/apt/lists/* && rm -rf /usr/share/locale/* && \
    rm -rf /var/cache/debconf/*-old && rm -rf /usr/share/doc/*

# Download and install Armadillo.
RUN curl https://ftp.fau.de/macports/distfiles/armadillo/armadillo-8.400.0.tar.xz \
    | tar -xvJ &&  cd armadillo* && \
    cmake . && make && sudo make install && cd ..  && rm -rf armadillo*
    
# Download mlpack; Build and Install mlpack and go-shared bindings.
ARG MLPACK_VERSION
ENV MLPACK_VERSION $MLPACK_VERSION
RUN curl -Lo mlpack.zip https://www.mlpack.org/files/mlpack-${MLPACK_VERSION}.tar.gz && \
    tar -xvzpf mlpack.zip && \
    cd mlpack-${MLPACK_VERSION} && \
    mkdir build && cd build && \
    cmake -D BUILD_TESTS=OFF \
          -D BUILD_JULIA_BINDINGS=OFF \
          -D BUILD_PYTHON_BINDINGS=OFF \
          -D BUILD_CLI_EXECUTABLES=OFF \
          -D BUILD_GO_BINDINGS=OFF \
          -D BUILD_GO_SHLIB=ON  .. && \
    make -j $(nproc --all) && \
    make preinstall && make install && ldconfig && \
    cd / && rm -rf mlpack*

# Install Golang 1.13.5
ARG GOVERSION="1.13.5"
ENV GOVERSION $GOVERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
            git software-properties-common && \
            curl -Lo go${GOVERSION}.linux-amd64.tar.gz https://dl.google.com/go/go${GOVERSION}.linux-amd64.tar.gz && \
            tar -C /usr/local -xzf go${GOVERSION}.linux-amd64.tar.gz && \
            rm go${GOVERSION}.linux-amd64.tar.gz && \
            rm -rf /var/lib/apt/lists/*

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"
WORKDIR $GOPATH

# Download dependencies required by go-bindings and mlpack-go-bindings. 
RUN go get -u gonum.org/v1/gonum/... && \
    go get -u -d mlpack.org/v1/mlpack/... && \
    go test -v ${GOPATH}/src/mlpack.org/v1/mlpack/tests/go_binding_test.go
