自己构建 Python 绑定的 OpenVDB 能确保你获得最新版本和最佳兼容性。以下是**从源码构建 OpenVDB 并启用 Python 绑定**的完整步骤，适用于 **Linux 系统（如 Ubuntu 20.04/22.04/24.04）**。

---

## ✅ 一、前提条件

### 💡 推荐环境：

- Python 3.10
- CMake ≥ 3.28
- g++ ≥ 13.3
- Ubuntu 系统 20.04/22.04/24.04

---

## 📦 二、安装依赖项

```bash
sudo apt update
sudo apt install -y \
  git cmake g++ \
  libtbb-dev \
  libblosc-dev \
  libboost-all-dev \
  libilmbase-dev \
  libopenexr-dev \
  libz-dev \
  python3-dev \
  python3-pip \
  pybind11-dev

# 可选：构建更快
pip install ninja
```

---

## 📥 三、获取 OpenVDB 源码

```bash
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb
# git checkout v10.0.0   # 建议选一个稳定版本（如 10.0）
```

---

## ⚙️ 四、配置 CMake 构建

推荐在单独目录中构建：

```bash
mkdir build && cd build

cmake ..  \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENVDB_BUILD_PYTHON_MODULE=ON \
        -DOPENVDB_BUILD_NANOVDB=ON \
        -DNANOVDB_BUILD_TOOLS=ON
```

<!-- 如果你没装 ninja，可以删掉最后一行 `-GNinja` -->

---

## 🔨 五、编译 & 安装

```bash
make -j$(nproc)
sudo make install
```

---

## 📦 六、安装到 Python 模块路径（如果没自动装好）

构建后生成的 Python 模块通常位于：

```bash
build/openvdb/openvdb/python/openvdb*.so
```

你可以手动安装为 pip 包：

```bash
cd build/openvdb/openvdb/python/
conda activate <env_name_you_want>
pip install .
```

或将其路径添加到 PYTHONPATH：

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## ✅ 七、验证安装

```bash
python3 -c "import openvdb; print(openvdb.version())"
```

或运行：

```python
import openvdb

grid = openvdb.FloatGrid()
grid.name = "density"
grid.fill((0, 0, 0), 1.0)
print("Grid value:", grid.eval((0, 0, 0)))  # 应为 1.0
```

---

## 📘 八、可选建议

### 推荐添加环境变量（添加到 `~/.bashrc`）：

```bash
export OPENVDB_ROOT=/usr/local
export PYTHONPATH=$OPENVDB_ROOT/lib/python3.x/site-packages:$PYTHONPATH
```

### 安装开发工具：

```bash
pip install open3d numpy jupyter
```

---

## 🧠 九、常见问题排查

| 问题                                   | 解决方法                                  |
| -------------------------------------- | ----------------------------------------- |
| 找不到 `pybind11`                      | 确保安装了 `pybind11-dev` 包              |
| 找不到 Python 模块                     | 检查是否运行了 `pip install .`            |
| Boost 报错找不到                       | 确保安装了 `libboost-all-dev`，或指定路径 |
| Python 3.12 不兼容（如报错 `destroy`） | 使用 Python 3.10 或 3.9                   |

---

是否需要我为你生成一个一键构建脚本 `build_openvdb.sh` 并写好所有路径配置？
