#!/bin/bash

USE_SFPS=$1
lib_ps=$2
sfps_version=$3

# --------------------

# echo ${sfps_param}

if [ $USE_SFPS -eq 1 ]
then
    echo "begin install sfps"

    echo "USE_SFPS: ${USE_SFPS}"
    echo "lib_ps: ${lib_ps}"
    echo "sfps_version: ${sfps_version}"
    cd $lib_ps/capnproto-c++-0.10.2
    chmod -R 777 $lib_ps/capnproto-c++-0.10.2
    ./configure
    make -j6 check
    make install
    
    algo_dir=/opt/huawei/sfps
    
    
    export pybind11_DIR=/usr/local/lib/python3.7/site-packages/pybind11/share/cmake/pybind11/
    export ZMQ_DIR=$algo_dir/3rdparty/ps-lite/deps/
    export RDMA=1
    export LD_LIBRARY_PATH=${algo_dir}/3rdparty/ps-lite/deps/lib:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
    export USE_CAPNP=0

    if [ -e $lib_ps/${sfps_version}.zip ]
    then
        echo "use cache"
        rm -rf ${algo_dir}
        cp $lib_ps/${sfps_version}.zip /opt/huawei/
        cd /opt/huawei
        unzip ${sfps_version}.zip
        ls -lh

        cd ${algo_dir}
        ALL2ALL=0 pip install -e .
    else
        if [ ! -d $lib_ps/${sfps_version} ]
        then
            echo "$lib_ps/${sfps_version} not exists!!! exit"
            exit
        fi
        # rm -rf ${algo_dir}-old && mkdir -p ${algo_dir}-old && mv ${algo_dir}/* ${algo_dir}-old/
        # rm -rf ${algo_dir} && mkdir -p ${algo_dir}
        mkdir -p $algo_dir
        cp -r $lib_ps/${sfps_version}/* ${algo_dir}/
        cd ${algo_dir} && rm -rf ${algo_dir}/SFPS/tools/*.pb.h && protoc -I=${algo_dir}/SFPS/tools/ --cpp_out=${algo_dir}/SFPS/tools/ ${algo_dir}/SFPS/tools/KVEmbedding.proto
        
        if [ ! -d $algo_dir/3rdparty/ps-lite/deps/ ]; then
            cp -r $lib_ps/${sfps_version}/3rdparty/ps-lite/deps $algo_dir/3rdparty/ps-lite/
        fi

        cd  $algo_dir/3rdparty/ps-lite/deps/lib/
        ln -s libzmq.so.5.0.0 libzmq.so.5
        ln -s libzmq.so.5 libzmq.so

        cd ${algo_dir}
        ALL2ALL=0 pip install -e . -v
        
        # cache
        echo "zip cache lib"
        cd /opt/huawei/
        zip -r ${sfps_version}.zip sfps
        cp ${sfps_version}.zip $lib_ps
    fi

    echo "end install sfps"
fi