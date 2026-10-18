# Physical AI Forklift - Goal-Conditioned SAC with Gazebo

**ENGLISH FOLLOWS**

このプロジェクトでは、Gazebo上の倉庫環境でフォークリフトを制御し、指定された棚位置まで自律移動した後、指定された高さまでフォークを上昇させます。

学習には Goal-Conditioned SAC を使用しています。  
同一の学習済みポリシーに対して `ROW / BIN / LEVEL` を変更することで、異なる目的地・棚位置・高さに対応します。

---

# 日本語

## 0) 20パターンGoal評価

同一の学習済み Goal-Conditioned SAC ポリシーに対して、ROW、BIN、LEVEL の組み合わせを変えた複数のGoalを与え、動作を評価しました。

[YouTubeで20-Goal評価動画を見る](https://www.youtube.com/watch?v=z2_p-0GHh6M)

※ この拡張評価動画は、全体の動作を短時間で確認できるよう16倍速で再生しています。

## 1) インストール方法と環境

### 想定環境

本プロジェクトは以下の構成を想定しています。

- Ubuntu Linux
- ROS 2 Jazzy
- Gazebo (`gz sim`)
- `ros_gz_bridge`
- Python virtual environment: `gazebo_wh`
- Stable-Baselines3
- PyTorch
- Gymnasium
- FFmpeg
- GUIのないサーバーの場合:
  - Xvfb
  - Openbox
  - x11vnc
  - wmctrl

Pythonパッケージの詳細は `requirements.txt` を参照してください。

### Python環境

このリポジトリ内のスクリプトでは、Python仮想環境を以下の場所に置くことを想定しています。

```bash
/opt/pyenv/gazebo_wh
```

例:

```bash
python3 -m venv /opt/pyenv/gazebo_wh
source /opt/pyenv/gazebo_wh/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

実行権限:

```bash
chmod +x scripts/*.sh
```

### GUIがないサーバーの場合

AWS EC2など、通常のデスクトップ画面がない環境では、仮想DISPLAYを起動します。

```bash
./scripts/setup_display.sh
```

このスクリプトは以下を起動します。

```text
Xvfb :99
Openbox
x11vnc
```

Gazeboはこの仮想DISPLAY上に表示されます。

```bash
./scripts/run_gazebo.sh
```

VNCクライアントから接続することで、Gazebo GUIを確認・操作できます。

### GUIがある環境の場合

通常のLinuxデスクトップ環境がある場合は `./scripts/setup_display.sh` を実行する必要はありません。

現在使用しているDISPLAYを設定したうえでGazeboを起動してください。

```bash
export DISPLAY=:0
./scripts/run_gazebo.sh
```

`DISPLAY` の値は環境に合わせて変更してください。

---

## 2) スクラッチから訓練する方法

まずGazeboを起動します。

GUIなしの場合:

```bash
./scripts/setup_display.sh
```

Gazebo起動:

```bash
./scripts/run_gazebo.sh
```

### ROS 2 - Gazebo Bridge

別ターミナルでROS 2とPython環境を有効化します。

```bash
source /opt/ros/jazzy/setup.bash
source /opt/pyenv/gazebo_wh/bin/activate
```

Bridgeを起動します。

```bash
ros2 run ros_gz_bridge parameter_bridge \
/model/forklift_simple/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
/model/forklift_simple/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
/model/forklift_simple/lift_force_cmd@std_msgs/msg/Float64]gz.msgs.Double \
/world/warehouse_world/model/forklift_simple/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model \
/world/warehouse_world/model/forklift_simple/link/base_link/sensor/base_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
/world/warehouse_world/model/forklift_simple/link/fork_carriage_link/sensor/fork_carriage_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
/world/warehouse_world/model/forklift_simple/link/left_fork_link/sensor/left_fork_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
/world/warehouse_world/model/forklift_simple/link/right_fork_link/sensor/right_fork_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts
```

### Scratch training

以下は、まず限定されたGoal範囲から学習を開始する例です。

```bash
python -u src/train_goal_conditioned.py \
  --world-name warehouse_world \
  --model-name forklift_simple \
  --model-sdf ./models/forklift_simple/model.sdf \
  --spawn-x 38.0 \
  --spawn-y 14.0 \
  --spawn-z 0.10 \
  --spawn-yaw -1.571 \
  --randomize-target \
  --target-rows 1 \
  --target-bins 8-12 \
  --target-levels 1 \
  --target-face auto \
  --xy-reference-point base \
  --timesteps 300000 \
  --save-dir ./models_ckpt_stageA \
  --checkpoint-freq 5000 \
  --model-out sac_forklift_stageA
```

`--resume-from` を指定しなければ、SACモデルはスクラッチから学習を開始します。

本プロジェクトでは、学習を安定させるため、Goalの範囲を段階的に拡張する curriculum training を使用できます。

```text
Stage A:
ROW=1
BIN=8-12
LEVEL=1

Stage B:
ROW=1
BIN=0-19
LEVEL=1

Stage C:
ROW=0-2
BIN=0-19
LEVEL=1

Stage D:
ROW=0-2
BIN=0-19
LEVEL=0-3
```

Stage B以降では、前Stageのcheckpointを指定して継続学習できます。

```bash
--resume-from ./models_ckpt_stageA/best_success_model.zip
```

学習済みモデルを利用する場合は、このリポジトリ内の以下を使用できます。

```text
checkpoints/best_success_model.zip
```

---

## 3) 推論方法

### 1つのGoalを指定する場合

GUIなしの場合:

```bash
./scripts/setup_display.sh
```

Gazebo:

```bash
./scripts/run_gazebo.sh
```

推論:

```bash
./scripts/infer.sh ROW BIN LEVEL
```

例:

```bash
./scripts/infer.sh 1 10 2
```

これは以下を意味します。

```text
ROW   = 1
BIN   = 10
LEVEL = 2
```

利用可能なGoal範囲:

```text
ROW   = 0-2
BIN   = 0-19
LEVEL = 0-3
```

`LEVEL` は0始まりです。

```text
LEVEL 0 = 1段目
LEVEL 1 = 2段目
LEVEL 2 = 3段目
LEVEL 3 = 4段目
```

推論ではデフォルトで以下を使用します。

```text
checkpoints/best_success_model.zip
```

### 動画を録画する場合

```bash
./scripts/record.sh start demo
./scripts/infer.sh 1 10 2
./scripts/record.sh stop
```

動画は以下に保存されます。

```text
videos/demo.mp4
```

### 複数のGoalを連続して推論する場合

`scripts/run_demo_goals.sh` の `GOALS` を編集してください。

```bash
GOALS=(
    "0 2 0"
    "1 10 2"
    "2 17 3"
)
```

これは同じ学習済みPolicyに対して以下を順番に与えます。

```text
Goal 1: ROW=0, BIN=2,  LEVEL=0
Goal 2: ROW=1, BIN=10, LEVEL=2
Goal 3: ROW=2, BIN=17, LEVEL=3
```

`GOALS` に行を追加すれば、より多くのGoalを連続して評価できます。20パターン評価したい場合も同じ形式で20行追加するだけです。

`TOTAL` は `GOALS` の要素数から自動計算されます。

実行:

```bash
CAMERA_WAIT_SEC=60 ./scripts/run_demo_goals.sh demo_multigoal
```

このスクリプトは以下を自動実行します。

```text
仮想DISPLAY / VNC起動
        ↓
Gazebo起動
        ↓
カメラ調整待機
        ↓
録画開始
        ↓
Goal 1 推論
        ↓
Goal 2 推論
        ↓
Goal 3 推論
        ↓
...
        ↓
録画終了
```

すべての推論は1つのMP4に録画されます。

結果は以下に保存されます。

```text
videos/demo_multigoal.mp4
videos/demo_multigoal_results.csv
```

---

# English

## 0) Extended 20-Goal Evaluation

The same trained Goal-Conditioned SAC policy was evaluated on multiple combinations of ROW, BIN, and LEVEL.

[Watch the 20-Goal Evaluation Video on YouTube](https://www.youtube.com/watch?v=z2_p-0GHh6M)

Note: This extended evaluation video is shown at 16x speed for compact visualization.

## 1) Installation and Environment

### Expected Environment

This project assumes the following environment:

- Ubuntu Linux
- ROS 2 Jazzy
- Gazebo (`gz sim`)
- `ros_gz_bridge`
- Python virtual environment: `gazebo_wh`
- Stable-Baselines3
- PyTorch
- Gymnasium
- FFmpeg
- For headless environments:
  - Xvfb
  - Openbox
  - x11vnc
  - wmctrl

See `requirements.txt` for the Python package requirements.

### Python Environment

The provided shell scripts assume that the Python virtual environment is located at:

```bash
/opt/pyenv/gazebo_wh
```

Example:

```bash
python3 -m venv /opt/pyenv/gazebo_wh
source /opt/pyenv/gazebo_wh/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Enable ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

Make the scripts executable:

```bash
chmod +x scripts/*.sh
```

### Headless Server

For a headless server such as an AWS EC2 instance, start the virtual display environment:

```bash
./scripts/setup_display.sh
```

This starts:

```text
Xvfb :99
Openbox
x11vnc
```

Then start Gazebo:

```bash
./scripts/run_gazebo.sh
```

The Gazebo GUI can be viewed and controlled through a VNC client.

### Desktop GUI Environment

If the machine already has a graphical desktop environment, the virtual display setup can be skipped.

Use the current graphical DISPLAY and start Gazebo directly.

```bash
export DISPLAY=:0
./scripts/run_gazebo.sh
```

Change the `DISPLAY` value as required by your environment.

---

## 2) Training from Scratch

Start Gazebo first.

For a headless environment:

```bash
./scripts/setup_display.sh
```

Then:

```bash
./scripts/run_gazebo.sh
```

### ROS 2 - Gazebo Bridge

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/pyenv/gazebo_wh/bin/activate
```

Start the bridge:

```bash
ros2 run ros_gz_bridge parameter_bridge \
/model/forklift_simple/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
/model/forklift_simple/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
/model/forklift_simple/lift_force_cmd@std_msgs/msg/Float64]gz.msgs.Double \
/world/warehouse_world/model/forklift_simple/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model \
/world/warehouse_world/model/forklift_simple/link/base_link/sensor/base_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
/world/warehouse_world/model/forklift_simple/link/fork_carriage_link/sensor/fork_carriage_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
/world/warehouse_world/model/forklift_simple/link/left_fork_link/sensor/left_fork_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
/world/warehouse_world/model/forklift_simple/link/right_fork_link/sensor/right_fork_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts
```

### Scratch Training Example

```bash
python -u src/train_goal_conditioned.py \
  --world-name warehouse_world \
  --model-name forklift_simple \
  --model-sdf ./models/forklift_simple/model.sdf \
  --spawn-x 38.0 \
  --spawn-y 14.0 \
  --spawn-z 0.10 \
  --spawn-yaw -1.571 \
  --randomize-target \
  --target-rows 1 \
  --target-bins 8-12 \
  --target-levels 1 \
  --target-face auto \
  --xy-reference-point base \
  --timesteps 300000 \
  --save-dir ./models_ckpt_stageA \
  --checkpoint-freq 5000 \
  --model-out sac_forklift_stageA
```

If `--resume-from` is not specified, a new SAC model is created and training starts from scratch.

A curriculum can be used to gradually expand the goal space:

```text
Stage A:
ROW=1
BIN=8-12
LEVEL=1

Stage B:
ROW=1
BIN=0-19
LEVEL=1

Stage C:
ROW=0-2
BIN=0-19
LEVEL=1

Stage D:
ROW=0-2
BIN=0-19
LEVEL=0-3
```

For later stages, training can be resumed from the previous checkpoint.

```bash
--resume-from ./models_ckpt_stageA/best_success_model.zip
```

A pretrained model is also included:

```text
checkpoints/best_success_model.zip
```

---

## 3) Inference

### Single Goal

For a headless environment:

```bash
./scripts/setup_display.sh
```

Start Gazebo:

```bash
./scripts/run_gazebo.sh
```

Run inference:

```bash
./scripts/infer.sh ROW BIN LEVEL
```

Example:

```bash
./scripts/infer.sh 1 10 2
```

Supported goal ranges:

```text
ROW   = 0-2
BIN   = 0-19
LEVEL = 0-3
```

Levels use zero-based indexing:

```text
LEVEL 0 = first shelf
LEVEL 1 = second shelf
LEVEL 2 = third shelf
LEVEL 3 = fourth shelf
```

By default, inference uses:

```text
checkpoints/best_success_model.zip
```

### Recording an Inference Run

```bash
./scripts/record.sh start demo
./scripts/infer.sh 1 10 2
./scripts/record.sh stop
```

The video will be saved as:

```text
videos/demo.mp4
```

### Multiple Goals

Edit the `GOALS` array in:

```text
scripts/run_demo_goals.sh
```

Example:

```bash
GOALS=(
    "0 2 0"
    "1 10 2"
    "2 17 3"
)
```

This evaluates:

```text
Goal 1: ROW=0, BIN=2,  LEVEL=0
Goal 2: ROW=1, BIN=10, LEVEL=2
Goal 3: ROW=2, BIN=17, LEVEL=3
```

Additional goals can be added simply by adding more lines to the `GOALS` array. A 20-goal evaluation can be performed by defining 20 entries.

The total number of runs is calculated automatically from the number of entries in `GOALS`.

Run:

```bash
CAMERA_WAIT_SEC=60 ./scripts/run_demo_goals.sh demo_multigoal
```

The script automatically performs:

```text
Virtual display / VNC startup
        ↓
Gazebo startup
        ↓
Camera adjustment wait
        ↓
Recording start
        ↓
Goal 1 inference
        ↓
Goal 2 inference
        ↓
Goal 3 inference
        ↓
...
        ↓
Recording stop
```

All inference runs are recorded into a single MP4 file.

Outputs:

```text
videos/demo_multigoal.mp4
videos/demo_multigoal_results.csv
```
