#!/bin/bash

echo "安装VDB生成工具依赖..."

# 更新包管理器
sudo apt-get update

# 安装系统依赖
echo "安装系统依赖..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libboost-all-dev \
    libtbb-dev \
    libblosc-dev \
    libopenexr-dev \
    libilmbase-dev \
    libeigen3-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev

# 安装Python依赖
echo "安装Python依赖..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 如果openvdb安装失败，尝试从源码编译
if ! python3 -c "import openvdb" 2>/dev/null; then
    echo "OpenVDB安装失败，尝试从源码编译..."
    
    # 克隆OpenVDB源码
    git clone https://github.com/AcademySoftwareFoundation/openvdb.git
    cd openvdb
    
    # 创建构建目录
    mkdir build && cd build
    
    # 配置CMake
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENVDB_BUILD_PYTHON_MODULE=ON \
        -DOPENVDB_BUILD_UNITTESTS=OFF \
        -DOPENVDB_BUILD_DOCS=OFF \
        -DOPENVDB_BUILD_VDB_PRINT=OFF \
        -DOPENVDB_BUILD_VDB_LOD=OFF \
        -DOPENVDB_BUILD_VDB_RENDER=OFF \
        -DOPENVDB_BUILD_VDB_VIEW=OFF
    
    # 编译
    make -j$(nproc)
    
    # 安装
    sudo make install
    
    # 更新库路径
    sudo ldconfig
    
    cd ../..
    rm -rf openvdb
fi

echo "依赖安装完成！"
echo "现在可以运行以下命令生成测试VDB文件："
echo "python3 generate_test_vdb.py"
echo "或者从OSM下载数据："
echo "python3 osm_to_vdb.py" 