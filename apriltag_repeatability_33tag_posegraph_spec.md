# AprilTag(0~32) 기반 “경로 반복성(Repeatability) 평가” 구현 스펙 v2  
> **치수(보드 내부 오프셋) 없이**도 정밀 평가가 가능하도록, **태그 33개를 모두 ‘독립 랜드마크 노드’로 취급**하는 버전입니다.  
> Cursor(코딩 에이전트)에게 그대로 복붙해서 구현 지시할 수 있도록 **디테일 포함**으로 작성했습니다.

---

## 0) 핵심 결론(이 방식이 왜 맞는가)
- 현장 제작에서 “보드에 태그 3개를 정확한 간격으로 붙이기”가 어렵다면, 오프셋 기반 보드 모델은 **고정 바이어스**를 만들 수 있습니다.
- 대신 **태그 33개(0~32)를 전부 노드로** 두고, 캘리브 주행에서 얻는 **태그-태그 상대변환 엣지**로 작은 **Pose Graph(SE2)** 를 최적화하면:
  - 줄자/실측 없이도 “태그들의 실제 상대 배치”가 데이터로 추정됨
  - 이후 평가 주행에서 보이는 태그 아무거나로도 **tag0 기준 카메라 포즈**를 추정 가능
  - 한 프레임에서 태그가 여러 개 보이면 **가중 fusion**으로 흔들림을 줄일 수 있음

> 물리적으로는 `012, 345, 678, ...` 식으로 3개씩 같은 위치(보드)에 붙여도 OK.  
> **소프트웨어는 “33개 개별 태그”로만 취급**합니다(보드 내부 오프셋 없음).

---

## 1) 전제/환경
- ROS2 Humble
- 카메라: ZED2i (천장 방향), 실내 평지 → **2D(x,y,yaw)만 사용**
- 천장 거리: 2.0~2.5 m
- 태그: family `tag25h9`, size 0.2 m
- ID: **0~32 연속**, 물리 배치 예: `{0,1,2}`, `{3,4,5}`, `{6,7,8}` ...  
- 검출: `apriltag_ros`
  - `detections`: AprilTagDetectionArray (quality meta)
  - tag pose: `/tf`에 publish (TF lookup으로 `T_cam_tag` 획득)

---

## 2) 데이터 정의(좌표계/수학)
### 2.1 SE(2) 포즈 표현
- 포즈 `X = (x, y, theta)`  
- `theta`는 yaw (rad), 항상 `wrap(theta) ∈ [-pi, pi]`

### 2.2 SE(2) 연산(필수 유틸)
- `wrap(theta)`
- `compose(a,b)` : a ⊕ b
- `inv(a)`
- `between(a,b) = inv(a) ⊕ b`

### 2.3 측정(엣지) 정의
프레임 t에서 태그 i, j가 동시에 보이면:
1) TF로 `T_cam_tag_i`, `T_cam_tag_j`를 얻음(3D)
2) 상대변환:
   - `T_tag_i_tag_j = inv(T_cam_tag_i) * T_cam_tag_j`
3) 2D 축약:
   - `Z_ij = project_to_se2(T_tag_i_tag_j)` → (dx, dy, dtheta)

저장: `(i, j, dx, dy, dtheta, weight, stamp)`

---

## 3) Phase A: 캘리브(1~2회) — 33노드 Pose Graph 최적화로 “tag0 기준 태그 맵” 만들기
### 3.1 목표
- tag0을 기준 원점으로 고정:
  - `X0 = (0,0,0)`
- 미지수: `X1..X32` (총 32개 SE2 포즈)
- 관측: 모든 프레임에서 얻는 태그-태그 엣지 `Z_ij(t)`
- 최적화: 모든 엣지를 가장 잘 만족하는 태그 배치 `Xk` 추정

### 3.2 캘리브 주행 수집 조건(정밀에 매우 중요)
이 방식에서 **그래프 연결성**이 생명입니다.

필수 조건:
- 단일 태그만 계속 보이는 구간만 있으면 엣지가 부족합니다.
- 최소한 주행 중 “두 태그 이상 동시 관측” 프레임이 자주 나와야 합니다.

권장 운영:
- 태그를 3개씩 같은 위치에 붙이는 이유가 바로 “동시 관측 빈도”를 높이기 위함입니다.
- 인접 보드(예: 012 보드와 345 보드)가 **겹쳐 보이는 구간**이 반드시 있어야 그래프가 강하게 연결됩니다.
  - (0/1/2)만 보이다가 (3/4/5)만 보이는 식이면, 두 보드 사이 엣지가 0이라 그래프가 끊깁니다.
  - 해결: 보드 간 간격/카메라 FOV 조정해서 전환 구간에 2개 보드가 동시에 보이도록 배치.

추가로 정밀 강화:
- 가능하면 “왕복/재방문”으로 루프 제약을 만들어 잔차를 줄이기

### 3.3 엣지 수집 노드: `collect_edges.py`
**입력**
- topic: `detections` (AprilTagDetectionArray)
- TF: tag frames (apriltag_ros publish)

**출력**
- `edges.jsonl` 또는 `edges.npz`

**핵심 로직**
1) `detections`에서 검출된 tag id 리스트 추출
2) 각 tag id에 대해 TF lookup:
   - 기준 프레임: `camera_frame`(예: `zed_left_camera_optical_frame`)
   - 타겟 프레임: tag frame (예: `tag_12`)
   - 시간: `detections.header.stamp`를 우선으로 lookup (가능하면 exact)
3) 한 프레임에 성공적으로 lookup된 tag들의 집합 S 생성
4) 모든 쌍 (i<j) in S에 대해 엣지 생성:
   - `T_ij = inv(T_cam_i) * T_cam_j`
   - `Z_ij = project_to_se2(T_ij)`
   - weight 계산
   - 파일 append

**weight(가중치) 추천**
- `w = w_i * w_j`
- tag별 품질 `w_i`:
  - `hamming == 0` 아니면 drop (또는 weight=0.1)
  - `decision_margin` 기반: `w_margin = clamp((dm - DM_MIN)/(DM_GOOD - DM_MIN), 0..1)`
    - 예: `DM_MIN=40`, `DM_GOOD=70`
  - 태그 픽셀 크기 기반: corners로 평균 변 길이(px) 추정
    - `w_px = clamp((px - PX_MIN)/(PX_GOOD - PX_MIN), 0..1)`
    - 예: `PX_MIN=70`, `PX_GOOD=140`
  - 최종 `w_i = w_margin * w_px`

**이상치 제거(간단 게이트)**
- `abs(dtheta)`가 과도하거나 dx/dy가 비현실적으로 크면 skip
  - 예: `|dtheta| > 45deg`, `sqrt(dx^2+dy^2) > 5m` 등(환경에 맞게)

**TF lookup 실패 처리**
- timeout 50~100ms
- 실패한 tag는 그 프레임에서 제외
- 남은 태그가 2개 미만이면 그 프레임은 엣지 없음

---

## 4) Phase A 오프라인 최적화: `solve_pose_graph.py` (SciPy least_squares)
### 4.1 변수 벡터
- `p = [x1,y1,th1, x2,y2,th2, ..., x32,y32,th32]`

### 4.2 잔차(residual)
각 엣지 `Z_ij`에 대해:
- `Xi = X(i)`, `Xj = X(j)` (i=0이면 (0,0,0) 고정)
- 예측 상대변환: `pred = between(Xi, Xj)`
- 측정: `Z = Z_ij`
- 오차: `err = between(Z, pred)` = inv(Z) ⊕ pred
- residual:
  - `[sqrt(w)*err.dx, sqrt(w)*err.dy, sqrt(w)*k_theta*wrap(err.dtheta)]`

`k_theta` (각도 스케일) 추천:
- 초기값 `k_theta = 0.5`
- 만약 yaw가 너무 흔들리면 0.7~1.0으로 올리고,
- yaw가 과하게 강제되어 xy가 망가지면 0.3~0.5로 낮추기

### 4.3 최적화 설정
- `least_squares(..., loss='huber', f_scale=0.05~0.10)`
- max_nfev 충분히
- 결과 잔차 통계 출력(평균/95%/최대)

### 4.4 초기값(수렴 안정)
권장:
- 엣지들을 이용해 tag0에서 BFS로 확장하며 초기 포즈 구성
  - (0->a) 평균 엣지가 있으면 Xa 초기화
  - 이어서 (a->b)로 Xb 초기화
- 또는 전부 0으로 두되, 수렴이 느려질 수 있음

### 4.5 출력
- `tag_map.yaml` (tag0 기준, 모든 tag의 SE2 포즈)
```yaml
tags:
  0: [0.0, 0.0, 0.0]
  1: [x1, y1, th1]
  ...
  32: [x32, y32, th32]
```

---

## 5) Phase B: 평가 주행(10회) — tag0 기준 카메라 궤적 기록
### 5.1 핵심 수식(태그 k가 보이면)
- 고정 태그 맵에서 `Xk = T_{tag0 -> tagk}`가 주어짐
- TF에서 `T_cam_tagk`를 얻으면:
  - `T_tag0_cam^(k) = Xk ⊕ inv( project_to_se2(T_cam_tagk) )`
  - 주의: `T_cam_tagk`를 바로 SE2로 축약한 뒤 invert해도 되고,
          3D에서 invert 후 SE2로 축약해도 됨(일관되게만 사용)

### 5.2 멀티 태그 fusion(정밀 향상 핵심)
한 프레임에서 태그가 여러 개 보이면 후보 포즈가 여러 개:
- `C_k = T_tag0_cam^(k)` for k ∈ S

이를 SE2 평균으로 fuse:
- x,y: 가중 평균
- yaw: 원형 평균
  - `yaw = atan2(sum(w*sin(yaw_k)), sum(w*cos(yaw_k)))`

가중치 w는 Phase A의 tag 품질 계산과 동일(decision_margin + 픽셀 크기).

### 5.3 기록 포맷(CSV)
`run_01.csv`:
```
t,x,y,yaw,tags_used,quality
1767..., 0.00, 0.00, 0.001, "0|1|2", 0.88
...
```

### 5.4 선택 옵션(평가 신뢰성)
- “평가용”이므로 과한 필터링(스무딩)은 오차를 숨길 수 있음
- 권장:
  - **raw 궤적을 저장**
  - (옵션) 시각화용으로만 약한 low-pass 제공

---

## 6) 평가 지표(Repeatability)
속도 차이를 제거하기 위해 **arc-length(거리) 기준**으로 비교 권장.

### 6.1 arc-length 리샘플링
- 각 run에서 누적거리 s 계산
- 공통 s grid(예: ds=0.01m)로 보간
- yaw는 unwrap 후 보간하고 wrap

### 6.2 지표
1) ATE (ref 대비 위치 오차)
- `e(s) = ||p(s) - p_ref(s)||`
- mean / RMSE / 95% / max

2) CTE (ref 법선 방향 횡오차)
- ref의 접선 방향으로 normal 벡터 계산
- `cte(s) = dot(p - p_ref, normal_ref)`
- mean(|cte|), RMSE(cte), 95%(|cte|), max(|cte|)

합격 기준 예:
- `95%(|CTE|) < 0.02m` (2cm)

---

## 7) 구현 파일 구조(권장)
```
apriltag_repeatability_eval/
  config/
    apriltag_ros_25h9.yaml
    tag_map.yaml            # Phase A 결과
  nodes/
    collect_edges.py
    localize_and_record.py
  scripts/
    solve_pose_graph.py
    evaluate_runs.py
  utils/
    se2.py
    quality.py
    tf_utils.py
  README.md
```

---

## 8) apriltag_ros 설정 팁(프레임명 안정화)
TF frame에 콜론(`tag25h9:12`)이 들어가면 처리하기 번거로울 수 있습니다.  
`tag.frames`를 지정해서 `tag_12` 형태로 통일하는 것을 권장.

`config/apriltag_ros_25h9.yaml` 예(개념):
```yaml
apriltag:
  ros__parameters:
    family: 25h9
    size: 0.2
    max_hamming: 0
    tag:
      ids:    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
      frames: ["tag_0","tag_1","tag_2","tag_3","tag_4","tag_5","tag_6","tag_7","tag_8","tag_9","tag_10","tag_11","tag_12","tag_13","tag_14","tag_15","tag_16","tag_17","tag_18","tag_19","tag_20","tag_21","tag_22","tag_23","tag_24","tag_25","tag_26","tag_27","tag_28","tag_29","tag_30","tag_31","tag_32"]
      sizes:  [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
```

---

## 9) 실행 절차(예시)
### 9.1 apriltag_ros 실행
```bash
ros2 run apriltag_ros apriltag_node --ros-args \
  -r image_rect:=/zed/zed_node/left/image_rect_color \
  -r camera_info:=/zed/zed_node/left/camera_info \
  --params-file ./config/apriltag_ros_25h9.yaml
```

### 9.2 Phase A: 엣지 수집
```bash
ros2 run apriltag_repeatability_eval collect_edges \
  --ros-args -p camera_frame:=zed_left_camera_optical_frame \
            -p out_edges:=./data/edges.jsonl
```

### 9.3 Phase A: 태그 맵 최적화
```bash
python3 scripts/solve_pose_graph.py \
  --edges ./data/edges.jsonl \
  --out ./config/tag_map.yaml
```

### 9.4 Phase B: run 기록(10회 반복)
```bash
ros2 run apriltag_repeatability_eval localize_and_record \
  --ros-args -p camera_frame:=zed_left_camera_optical_frame \
            -p tag_map:=./config/tag_map.yaml \
            -p out_csv:=./data/run_01.csv
```

### 9.5 평가
```bash
python3 scripts/evaluate_runs.py \
  --ref ./data/run_01.csv \
  --runs ./data/run_*.csv \
  --ds 0.01 \
  --out_dir ./out \
  --pass_cte95 0.02
```

---

## 10) Cursor 작업 지시(체크리스트)
- [ ] ROS2 Python package 생성 + 의존성 정리(`rclpy`, `tf2_ros`, `apriltag_msgs`, `numpy`, offline: `scipy`, `matplotlib`)
- [ ] `collect_edges.py`: detections + TF lookup으로 edges.jsonl 저장
- [ ] `solve_pose_graph.py`: SciPy least_squares(huber)로 tag_map.yaml 생성
- [ ] `localize_and_record.py`: tag_map 기반으로 매 프레임 tag0->cam 계산 + 멀티 태그 fuse + CSV 기록
- [ ] `evaluate_runs.py`: arc-length resample + ATE/CTE 통계 + 플롯 + pass/fail 옵션
- [ ] 각도 wrap/unwrap, 원형 평균 구현(필수)
- [ ] TF lookup 시간은 `detections.header.stamp` 우선, 실패 시 skip
- [ ] `tag.frames`로 프레임명 `tag_#` 통일(권장)
- [ ] 디버그 모드: 프레임당 감지 태그 수 / 평균 weight / TF 실패율 출력

---

## 11) 참고(사용자 배치 요약)
- ID: 0~32 연속 사용
- 물리적으로 3개씩 묶어서 설치(동시 관측 증가 목적):
  - (0,1,2), (3,4,5), (6,7,8), …, (30,31,32)

끝.
