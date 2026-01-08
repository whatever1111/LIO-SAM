# Noetic + Docker 分发与使用（仅二进制 + 配置）

本文档面向 **ROS1 Noetic / Ubuntu 20.04 / amd64** 部署，目标是把 `lio_sam` 的**可执行文件与配置文件**分发到其他服务器使用，而不是分发完整源码。

本仓库提供 `docker/Dockerfile.noetic`（multi-stage）：
- **builder stage**：拉依赖并编译，产出 `catkin install space`（`/ws/install`）
- **runtime stage**：只包含运行时依赖 + `/ws/install`（不包含源码）

---

## 1) 构建（Build）

### 1.1 前置条件
- 构建机已安装 Docker（20.10+ 推荐）
- 构建机能访问 Docker Hub、Ubuntu/ROS apt 源
- 构建机已准备 Fixposition 相关 catkin packages（用于编译 msgs/common；最终 runtime 镜像**不包含**这些源码）

### 1.2 获取 Fixposition 依赖（构建用，不随镜像分发源码）

`docker/Dockerfile.noetic` 默认启用 Fixposition/FPA（`LIO_SAM_ENABLE_FIXPOSITION=ON`），并期望你在构建上下文中提供以下目录：

```text
third_party/fixposition_driver/fixposition-sdk/fpsdk_common
third_party/fixposition_driver/fixposition-sdk/fpsdk_ros1
third_party/fixposition_driver/fixposition_driver_lib
third_party/fixposition_driver/fixposition_driver_msgs
third_party/fixposition_driver/rtcm_msgs
third_party/fixposition_driver/fixposition_driver_ros1
```

做法 1（推荐）：在构建机把 Fixposition 驱动仓库 clone 到 `third_party/`（不需要 commit 到你的仓库）：

```bash
mkdir -p third_party
git clone <YOUR_FIXPOSITION_DRIVER_REPO_URL> third_party/fixposition_driver
# 拉取 Fixposition SDK 子模块（必须）
git -C third_party/fixposition_driver submodule update --init --recursive
# 建议 pin 到你信任的 tag/commit，避免供应链漂移：
# (cd third_party/fixposition_driver && git checkout <TAG_OR_COMMIT>)
```

### 1.3 构建镜像（amd64）

在仓库根目录执行：

```bash
docker build -f docker/Dockerfile.noetic -t liosam-noetic:amd64 .
```

> 提示：你当前目标机是 amd64，不需要 `buildx --platform`。

### 1.4（可选）构建时走代理

如果构建机需要代理才能访问外网，可加 build args：

```bash
docker build -f docker/Dockerfile.noetic -t liosam-noetic:amd64 \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  .
```

---

## 2) 分发（Distribute）

推荐分发 **runtime 镜像 + 配置文件**。镜像里只有运行产物（`/ws/install`）与依赖，不含源码。

### 2.x 安全检查：确认镜像不包含源码

构建完成后，你可以在目标镜像内快速自检：

```bash
docker run --rm -it liosam-noetic:amd64 bash -lc 'ls -la /ws/src || echo "OK: no /ws/src"; find /ws -maxdepth 4 -name "*.cpp" | head'
```

预期：
- `/ws/src` 不存在（源码只在 builder stage）
- 运行时主要目录为 `/ws/install`

### 2.1 通过镜像仓库（推荐）

```bash
docker tag liosam-noetic:amd64 <REGISTRY>/<NAMESPACE>/liosam-noetic:<TAG>
docker push <REGISTRY>/<NAMESPACE>/liosam-noetic:<TAG>
```

目标机拉取：

```bash
docker pull <REGISTRY>/<NAMESPACE>/liosam-noetic:<TAG>
```

### 2.2 离线分发（不依赖仓库）

构建机导出：

```bash
docker save liosam-noetic:amd64 | gzip > liosam-noetic-amd64.tar.gz
```

拷贝到目标机后导入：

```bash
gunzip -c liosam-noetic-amd64.tar.gz | docker load
```

---

## 3) 配置文件管理（Config）

建议把配置放在宿主机目录中，用 volume 挂载到容器（只读）：

示例目录：

```text
/opt/lio_sam/
  config/
    params.yaml
```

你可以使用本仓库的 `config/params.yaml` 作为模板，按自己的传感器/话题/外参修改。

### 3.1 在目标机器上修改 `params.yaml`（推荐方式）

推荐在目标机维护一份“本机专用配置”，不要去容器里改文件（容器是一次性的，重建/升级会丢失修改）。

1) 在目标机创建配置目录并拷贝一份模板：

```bash
sudo mkdir -p /opt/lio_sam/config
sudo cp /path/to/repo/config/params.yaml /opt/lio_sam/config/params.yaml
sudo chown -R $USER:$USER /opt/lio_sam
```

2) 用编辑器修改（例如 nano/vim）：

```bash
nano /opt/lio_sam/config/params.yaml
```

常见需要调整的字段（见 `lio_sam:` 节点下）：
- `pointCloudTopic`：点云话题（与雷达驱动一致）
- `imuTopic`、`useFpaImu`：IMU 话题与格式（是否使用 `fixposition_driver_msgs/FpaImu`）
- `gpsTopic`：GPS/里程计话题（如使用 GNSS）
- `lidarFrame`/`baselinkFrame`/`odometryFrame`/`mapFrame`：坐标系命名（需与 TF 体系一致）
- `extrinsicTrans`、`extrinsicRot`、`extrinsicRPY`：外参（尤其是 IMU↔LiDAR）
- `sensor`、`N_SCAN`、`Horizon_SCAN`：传感器类型/线数/水平分辨率

3) 运行时挂载并指定 `params_file`：

```bash
docker run --rm --net=host \
  -v /opt/lio_sam/config:/config:ro \
  liosam-noetic:amd64 \
  roslaunch lio_sam run_headless.launch params_file:=/config/params.yaml
```

如需启用 `robot_localization`（navsat/ekf），请在启动时显式打开，并确保镜像/宿主机安装了 `ros-noetic-robot-localization`：

```bash
docker run --rm --net=host \
  -v /opt/lio_sam/config:/config:ro \
  liosam-noetic:amd64 \
  roslaunch lio_sam run_headless.launch params_file:=/config/params.yaml use_navsat:=true
```

### 3.2（可选）容器内快速验证参数是否加载

你可以临时进入容器，加载 launch 后用 `rosparam get` 检查：

```bash
docker run --rm -it --net=host -v /opt/lio_sam/config:/config:ro liosam-noetic:amd64 bash
# 容器内：
roslaunch lio_sam run_headless.launch params_file:=/config/params.yaml
```

另开一个终端（或同终端先后台运行 launch）：

```bash
rosparam get /lio_sam/pointCloudTopic
```

---

## 4) 运行（Run）

### 4.1 网络模式

ROS1 常用（推荐）：
- `--net=host`（最省心，避免 ROS_MASTER_URI/端口映射问题）

### 4.2 启动 Fixposition ROS1 驱动

本镜像包含 `fixposition_driver_ros1`（以及其依赖的 `fpsdk_*`、`fixposition_driver_*`、`rtcm_msgs`），可直接在容器里启动驱动节点。

驱动默认读取包内的 `config.yaml`：
- 容器内路径：`/ws/install/share/fixposition_driver_ros1/launch/config.yaml`

推荐做法是在目标机维护一份自己的配置，并用 volume 覆盖到该路径（避免在容器内改文件）：

```bash
sudo mkdir -p /opt/fixposition
docker create --name fp_cfg_extract liosam-noetic:amd64 >/dev/null
docker cp fp_cfg_extract:/ws/install/share/fixposition_driver_ros1/launch/config.yaml /opt/fixposition/config.yaml
docker rm fp_cfg_extract >/dev/null
sudo chown -R $USER:$USER /opt/fixposition
nano /opt/fixposition/config.yaml
```

如果是串口连接（示例 `/dev/ttyUSB0`），运行时把设备透传进去：

```bash
docker run --rm --net=host \
  --device=/dev/ttyUSB0 \
  -v /opt/fixposition/config.yaml:/ws/install/share/fixposition_driver_ros1/launch/config.yaml:ro \
  liosam-noetic:amd64 \
  roslaunch fixposition_driver_ros1 node.launch
```

如果是 TCP（例如传感器在网口上），通常不需要 `--device`，只要 `--net=host` 即可。

### 4.3 启动（无 GUI，推荐）

`ros:noetic-ros-base` 不包含 RViz/xacro 等 GUI 相关依赖，本仓库提供了无 GUI 的启动文件：
- `launch/run_headless.launch`：只启动 LIO-SAM（可选 navsat），不启动 RViz/robot_state_publisher

目标机执行：

```bash
docker run --rm --net=host \
  -v /opt/lio_sam/config:/config:ro \
  liosam-noetic:amd64 \
  roslaunch lio_sam run_headless.launch params_file:=/config/params.yaml
```

### 4.4 启动（Fixposition FPA）

前置：已按 **4.2** 启动 `fixposition_driver_ros1`，并确保：
- `output_ns` 输出在 `/fixposition`（默认 `config.yaml` 已设置）
- `/fixposition/fpa/odometry`、`/fixposition/fpa/corrimu` 等话题存在（按你的传感器配置/消息配置可能不同）

运行：

```bash
docker run --rm --net=host \
  -v /opt/lio_sam/config:/config:ro \
  liosam-noetic:amd64 \
  roslaunch lio_sam run_fpa.launch params_file:=/config/params.yaml
```

### 4.5（可选）把日志/产物写到宿主机

如果你的 `params.yaml` 里启用了 `savePCD` 或输出到某个目录，建议把目标目录挂载出来，例如：

```bash
docker run --rm --net=host \
  -v /opt/lio_sam/config:/config:ro \
  -v /opt/lio_sam/data:/data \
  liosam-noetic:amd64 \
  roslaunch lio_sam run_headless.launch params_file:=/config/params.yaml
```

并在 `params.yaml` 里把 `savePCDDirectory` 改到 `/data/...` 这类路径。

---

## 5)（可选）只导出“二进制 install/ + 配置”，不分发镜像

不推荐（因为目标机需要自行安装完整运行依赖），但如果你确实只想带走 `install/`：

在构建机从镜像里导出 `install/`：

```bash
docker create --name lio_extract liosam-noetic:amd64
docker cp lio_extract:/ws/install ./install
docker rm lio_extract

tar -czf lio_sam_install_noetic_amd64.tgz install
```

把 `lio_sam_install_noetic_amd64.tgz` 和 `params.yaml` 拷到目标机后：

```bash
tar -xzf lio_sam_install_noetic_amd64.tgz
source /opt/ros/noetic/setup.bash
source ./install/setup.bash
roslaunch lio_sam run_headless.launch params_file:=/path/to/params.yaml
```

---

## 6) 常见问题（FAQ）

### Q1：为什么 `run.launch` 在容器里跑不起来？
`run.launch` 会启动 RViz/robot_state_publisher（依赖 xacro、rviz 等）。当前 runtime 镜像基于 `ros:noetic-ros-base`，不包含这些 GUI/桌面依赖。

解决：
- 用 `run_headless.launch`（推荐）
- 或自行做一个 desktop 运行镜像（把 base 换成 `ros:noetic-desktop`/`desktop-full`，并安装 rviz/xacro）

### Q2：为什么构建时需要访问 GitHub？
本仓库的 `docker/Dockerfile.noetic` **不会**在 Docker build 过程中 `git clone` 任意仓库（避免把供应链下载逻辑藏在镜像构建里）。

但你仍然需要在“构建机宿主机”上准备 Fixposition 依赖源码（例如你自己 `git clone` 到 `third_party/fixposition_driver/`），用于编译 `fixposition_driver_msgs` 和 `fpsdk_common`。最终 runtime 镜像只包含 `/ws/install`，不含这些源码。

### Q3：`use_navsat` 是什么意思？
`use_navsat` 是 `launch/run_headless.launch` 的开关参数：
- `use_navsat:=false`（默认）：只跑 LIO-SAM，不启动 GNSS/robot_localization 融合
- `use_navsat:=true`：会额外 include `launch/include/module_navsat.launch`，启动 `robot_localization` 的 `ekf_localization_node` + `navsat_transform_node`，用于把 `gps/fix`（NavSatFix）与 IMU/里程计融合成 `/odometry/navsat`，给 LIO-SAM 的 GPS 因子使用

### Q4：构建时报 `Could not find a package configuration file provided by "nlohmann_json"`？
这是 Fixposition SDK 的 `fpsdk_common` 在配置阶段执行 `find_package(nlohmann_json REQUIRED)` 引起的。

本仓库的 `docker/Dockerfile.noetic` 已内置了兜底（`docker/cmake/Findnlohmann_json.cmake`），并通过 `-DCMAKE_MODULE_PATH=/opt/cmake_modules` 注入到构建中；如果你仍然看到该报错，通常是因为使用了旧缓存层。

建议：
- 先确认你用的就是当前仓库里的 `docker/Dockerfile.noetic`
- 重新构建时加 `--no-cache`（或至少清理相关层缓存）：

```bash
docker build --no-cache -f docker/Dockerfile.noetic -t liosam-noetic:amd64 .
```

### Q5：构建时报 `gtsam_unstable ... but this file does not exist`？
这是 GTSAM 的 CMake 导出文件引用了 `libgtsam_unstable.so`，但系统里只装了部分 GTSAM 包导致的。

本仓库的 `docker/Dockerfile.noetic` 已包含修复：
- runtime 安装 `libgtsam4` + `libgtsam-unstable4`
- builder 额外安装 `libgtsam-dev` + `libgtsam-unstable-dev`

如果你仍遇到该问题，同样建议用 `--no-cache` 重新构建，避免旧层里残留了不完整的 GTSAM 安装。
