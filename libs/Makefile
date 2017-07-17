ROOT_DIR=$(CURDIR)

echo:
	echo $(ROOT_DIR)
lib:
	mkdir lib
build:
	mkdir build
downloads:
	mkdir downloads

py35:
	mkdir -p venv
	source activate py35 && PYTHONUSERBASE=$(readlink -m venv)

clean:
	rm -rf ../build/
	rm -rf ../bindings/python/build/  # nessesary for swig

pre:
	make lib/boost
	make lib/CNTKCustomMKL
	make lib/protobuf-3.1.0
	make lib/mpi
	make lib/libzip
	make lib/opencv-3.1.0
	make lib/swig
	make lib/cub-1.4.1
	make lib/cudnn-5.1

# git submodule update --init -- ../Source/1BitSGD
# git submodule update --init -- ../Source/Multiverso
# conda install libgcc  # https://github.com/tensorflow/tensorflow/issues/5017
# sudo apt-get install cmake graphviz zip unzip
# make lib/CNTKCustomMKL
# make lib/protobuf-3.1.0
# make lib/mpi  # slow
# make lib/libzip  # fast
# make lib/boost  # break, ignore it
# make lib/opencv-3.1.0  # slow
# make lib/swig
# make lib/cub-1.4.1  #fast
# make lib/cudnn-5.1  #fast
# make ../build/release
# make CNTK_wheel
# make CNTK_setup.py

# Known problems:
# swig: Error with libpcre.so.1
# Verify: ldd lib/swig/bin/swig  # libpcre.so.1 => not found
# Verify: ldconfig -p | grep libpcre.so  # Sould show that libpcre.so.3 is availiable
# Solution (Hack): sudo ln -s /lib/x86_64-linux-gnu/libpcre.so.3  /lib/x86_64-linux-gnu/libpcre.so.1

.PHONY:
lib/dependencies: lib build downloads lib/CNTKCustomMKL lib/protobuf-3.1.0 lib/mpi lib/boost-1.60.0 lib/libzip lib/opencv-3.1.0 lib/swig lib/cudnn-5.1 CNTK
	# boost bricht aktuell ab, obwohl es functioniert hat (python bindings scheinen nicht zu gehen, werden aber auch nicht gebraucht)

# work
downloads/CNTKCustomMKL-Linux-3.tgz: downloads
	# cd downloads && wget https://www.cntk.ai/mkl/CNTKCustomMKL-Linux-2.tgz
	cd downloads && wget https://www.cntk.ai/mkl/CNTKCustomMKL-Linux-3.tgz

# work
lib/CNTKCustomMKL: lib downloads/CNTKCustomMKL-Linux-3.tgz
	mkdir $@
	tar -xzf downloads/CNTKCustomMKL-Linux-3.tgz -C $@
	# rm CNTKCustomMKL-Linux-2.tgz


# work
downloads/v3.1.0.tar.gz: downloads
	cd downloads && wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz

# work
build/protobuf-3.1.0: build downloads/v3.1.0.tar.gz
	tar -xzf downloads/v3.1.0.tar.gz -C build/
	cd $@ && ./autogen.sh
	cd $@ && ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=$(ROOT_DIR)/lib/protobuf-3.1.0
	cd $@ && make -j $$(nproc)

# work
lib/protobuf-3.1.0: lib build/protobuf-3.1.0
	# tar -xzf v3.1.0.tar.gz -C ..
	cd build/protobuf-3.1.0 && make install


# work
downloads/openmpi-1.10.3.tar.gz: downloads
	cd downloads && wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz

# work
build/openmpi-1.10.3: downloads downloads/openmpi-1.10.3.tar.gz
	cd build && tar -xzvf ../downloads/openmpi-1.10.3.tar.gz
	cd $@ && ./configure --prefix=$(ROOT_DIR)/lib/mpi
	cd $@ && make -j $$(nproc) all

# work
lib/mpi: lib build/openmpi-1.10.3
	cd build/openmpi-1.10.3 && make install

#work
downloads/libzip-1.1.2.tar.gz: downloads
	cd downloads && wget http://nih.at/libzip/libzip-1.1.2.tar.gz

#work
build/libzip-1.1.2: build downloads/libzip-1.1.2.tar.gz
	cd build && tar -xzvf ../downloads/libzip-1.1.2.tar.gz
	cd $@ && ./configure --prefix=$(ROOT_DIR)/lib/libzip
	cd $@ && make -j $$(nproc) all

#work
lib/libzip: lib build/libzip-1.1.2
	cd build/libzip-1.1.2 && make install

# breaks, but builds enough to work
downloads/boost_1_60_0: downloads lib/libzip
	cd downloads && wget -q -O - https://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz/download | tar -xzf -

# breaks, but builds enough to work
build/boost: build downloads/boost_1_60_0 lib/libzip
	cp -r downloads/boost_1_60_0 build/boost
	cd $@ && ./bootstrap.sh --prefix=$(ROOT_DIR)/lib/boost

# breaks, but builds enough to work
lib/boost: lib build/boost lib/libzip
	cd build/boost && ./b2 -d0 -j"$$(nproc)" install || true

# not working
.PHONY: multiverso
multiverso: CNTK
	# git clone https://github.com/Microsoft/multiverso.git
	# mkdir multiverso/build && cd multiverso/build
	# cd multiverso/build && cmake .. && make && sudo make install

	cd CNTK && git submodule update --init -- Source/Multiverso

#work
downloads/3.1.0.zip:
	cd downloads && wget https://github.com/Itseez/opencv/archive/3.1.0.zip

#work
build/opencv-3.1.0: downloads/3.1.0.zip
	cd build && unzip ../downloads/3.1.0.zip
	mkdir -p build/opencv-3.1.0/release
	cd build/opencv-3.1.0/release && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(ROOT_DIR)/lib/opencv-3.1.0 -D WITH_CUDA=OFF ..
	cd build/opencv-3.1.0/release && make all

#work
lib/opencv-3.1.0: lib build/opencv-3.1.0
	cd build/opencv-3.1.0/release && make install

#work
downloads/swig-3.0.12.tar.gz: downloads
	cd $(@D) && wget http://downloads.sourceforge.net/swig/swig-3.0.12.tar.gz

#work
build/swig-3.0.12: build downloads/swig-3.0.12.tar.gz
	cd $(@D) && tar -xzf ../downloads/swig-3.0.12.tar.gz
	cd $@ && ./configure --prefix=$(ROOT_DIR)/lib/swig
	cd $@ && make

#work
lib/swig: build/swig-3.0.12
	cd build/swig-3.0.12 && make install


# CNTK/bindings/python/swig-3.0.10/root:
# 	cd CNTK/bindings/python && ./swig_install.sh

#work
lib/cub-1.4.1: lib
	cd downloads && wget https://github.com/NVlabs/cub/archive/1.4.1.zip
	cd build && unzip ../downloads/1.4.1.zip
	cp -r build/cub-1.4.1 lib

#work
lib/cudnn-5.1: lib
	cd downloads && wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-7.5-linux-x64-v5.1.tgz
	cd build && tar -xzvf ../downloads/cudnn-7.5-linux-x64-v5.1.tgz
	mkdir lib/cudnn-5.1
	cp -r build/cuda lib/cudnn-5.1

#work
# CNTK:
# 	git clone git@github.com:Microsoft/CNTK.git

#work

.PHONY: ../build/release
../build/release:
	mkdir ../build/release -p
	cd $@ \
	&& export PATH=$(ROOT_DIR)/lib/mpi/bin/:$$PATH \
	&& export LD_LIBRARY_PATH=$(ROOT_DIR)/lib/mpi/lib:$$LD_LIBRARY_PATH \
	&& export SWIG_LIB=$(ROOT_DIR)/lib/swig/share/swig/3.0.10/ \
	&& ../../configure \
	--with-mkl=$(ROOT_DIR)/lib/CNTKCustomMKL \
	--with-protobuf=$(ROOT_DIR)/lib/protobuf-3.1.0 \
	--with-libzip=$(ROOT_DIR)/lib/libzip \
	--with-opencv=$(ROOT_DIR)/lib/opencv-3.1.0 \
	--with-boost=$(ROOT_DIR)/lib/boost \
	--with-cuda=/usr/local/cuda \
	--with-mpi=$(ROOT_DIR)/lib/mpi \
	--1bitsgd=yes \
	--with-gdk-include=/usr/local/cuda-8.0/include \
	--with-gdk-nvml-lib=/usr/local/cuda-8.0/lib64/stubs \
	--with-cub=$(ROOT_DIR)/lib/cub-1.4.1 \
	--with-cudnn=$(ROOT_DIR)/lib/cudnn-5.1 \
	--with-swig=$(ROOT_DIR)/lib/swig

	# --with-py-versions=36
	# --with-py36-path=/opt/anaconda/

	# --with-swig=$(ROOT_DIR)/CNTK/bindings/python/swig-3.0.10/root --with-py36-path=/net/ssd/software/anaconda/envs/py36_test

	#../../configure --with-mkl=/net/nas/boeddeker/ms/lib/CNTKCustomMKL --with-protobuf=/net/nas/boeddeker/ms/lib/protobuf-3.1.0 --with-libzip=/net/nas/boeddeker/ms/lib/libzip  --with-opencv=/net/nas/boeddeker/ms/lib/opencv-3.1.0  --with-boost=/net/nas/boeddeker/ms/lib/boost --with-cuda=/usr/local/cuda
	# cd build/release
	# ../../configure --with-buildtype=release --with-mkl=/usr/local/CNTKCustomMKL --1bitsgd=yes 
	# --with-gdk-include=/usr/local/cuda-8.0/include --with-gdk-nvml-lib=/usr/local/cuda-8.0/lib64/stubs 
	# --with-swig=$HOME/python/cntk/bindings/python/swig-3.0.10/root


	# ../../configure --with-mkl=/net/nas/boeddeker/ms/lib/CNTKCustomMKL --with-protobuf=/net/nas/boeddeker/ms/lib/protobuf-3.1.0 --with-libzip=/net/nas/boeddeker/ms/lib/libzip  --with-opencv=/net/nas/boeddeker/ms/lib/opencv-3.1.0  --with-boost=/net/nas/boeddeker/ms/lib/boost --with-cuda=/usr/local/cuda --with-cuda=/usr/local/cuda --with-cudnn=/net/nas/boeddeker/ms/lib/cudnn-5.1 --with-gdk-include=/usr/local/cuda/targets/x86_64-linux/include --with-gdk-nvml-lib=/usr/local/cuda/targets/x86_64-linux/lib/stubs --with-cub=/net/nas/boeddeker/ms/lib/cub-1.4.1

	# cd $@ && make -j $$(nproc) all
	cd $@ && make -j 4 all

CNTK_cpp_build:
	cd ../build/release && make -j 4 all 

CNTK_cpp_debug: ../build/debug
	cd ../build/debug && make -j 16 all

.PHONY: ../build/debug
../build/debug:
	mkdir ../build/debug -p
	cd $@ \
	&& export PATH=$(ROOT_DIR)/lib/mpi/bin/:$$PATH \
	&& export LD_LIBRARY_PATH=$(ROOT_DIR)/lib/mpi/lib:$$LD_LIBRARY_PATH \
	&& export SWIG_LIB=$(ROOT_DIR)/lib/swig/share/swig/3.0.10/ \
	&& ../../configure \
	--with-buildtype=debug \
	--with-mkl=$(ROOT_DIR)/lib/CNTKCustomMKL \
	--with-protobuf=$(ROOT_DIR)/lib/protobuf-3.1.0 \
	--with-libzip=$(ROOT_DIR)/lib/libzip \
	--with-opencv=$(ROOT_DIR)/lib/opencv-3.1.0 \
	--with-boost=$(ROOT_DIR)/lib/boost \
	--with-cuda=/usr/local/cuda \
	--with-mpi=$(ROOT_DIR)/lib/mpi \
	--1bitsgd=yes \
	--with-gdk-include=/usr/local/cuda-8.0/include \
	--with-gdk-nvml-lib=/usr/local/cuda-8.0/lib64/stubs \
	--with-cub=$(ROOT_DIR)/lib/cub-1.4.1 \
	--with-cudnn=$(ROOT_DIR)/lib/cudnn-5.1 \
	--with-swig=$(ROOT_DIR)/lib/swig


python:
	# cd CNTK && make python
	export CNTK_ROOT=$(abspath $(ROOT_DIR)/..)

CNTK_setup.py_debug:
	cd ../bindings/python \
	&& export PATH=$(ROOT_DIR)/lib/swig/bin:$$PATH \
	&& export CNTK_LIB_PATH=$(ROOT_DIR)/../build/debug/lib \
	&& export CNTK_ROOT=$(abspath $(ROOT_DIR)/..) \
	&& pip install --user -e .
	cd ../bindings/python \
	&& export PATH=$(ROOT_DIR)/lib/swig/bin:$$PATH \
	&& export CNTK_LIB_PATH=$(ROOT_DIR)/../build/debug/lib \
	&& export CNTK_ROOT=$(abspath $(ROOT_DIR)/..) \
	&& python setup.py build_ext --inplace

CNTK_setup.py:
	cd ../bindings/python \
	&& export PATH=$(ROOT_DIR)/lib/swig/bin:$$PATH \
	&& export CNTK_LIB_PATH=$(ROOT_DIR)/../build/release/lib \
	&& export CNTK_ROOT=$(abspath $(ROOT_DIR)/..) \
	&& pip install --user -e .

CNTK_wheel:
	cd ../bindings/python \
	&& export PATH=$(ROOT_DIR)/lib/swig/bin:$$PATH \
	&& export CNTK_ROOT=$(abspath $(ROOT_DIR)/..) \
	&& python setup.py build_ext --inplace bdist_wheel --dist-dir wheelhouse

	# cd CNTK/bindings/python && export LIBRARY_PATH=$(ROOT_DIR)/CNTK/build/release/lib && export PATH=$(ROOT_DIR)/lib/swig/bin:$$PATH && export SWIG_LIB=$(ROOT_DIR)/lib/swig/share/swig/3.0.10 && pip install --user -e .
	# cd CNTK/bindings/python && export CNTK_COMPONENT_VERSION=2.0rc2 && export LIBRARY_PATH=$(ROOT_DIR)/CNTK/build/release/lib && export PATH=$(ROOT_DIR)/CNTK/bindings/python/swig-3.0.10/root/bin:$$PATH && export LD_LIBRARY_PATH=$(ROOT_DIR)/CNTK/bindings/python/swig-3.0.10/root/share/swig/3.0.10:$$LD_LIBRARY_PATH && export SWIG_LIB=$(ROOT_DIR)/CNTK/bindings/python/swig-3.0.10/root && pip install --user -e .
	# cd CNTK/bindings/python && export LIBRARY_PATH=$(ROOT_DIR)/CNTK/build/release/lib && export PATH=$(ROOT_DIR)/lib/swig/bin:$$PATH && export SWIG_LIB=$(ROOT_DIR)/lib/swig/share/swig/3.0.10/ && pip install --user -e .


CNTK_python_start:
	export LD_LIBRARY_PATH=/net/vol/boeddeker/ms/CNTK/build/release/lib/:$LD_LIBRARY_PATH && python

