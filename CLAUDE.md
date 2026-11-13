# 專題：4 × Orange Pi 6 Plus 邊緣 LLM 管線平行推論叢集

本檔案給 Claude Code 讀取。你在使用者的本機電腦上執行，透過 SSH 操作 4 張 Orange Pi 6 Plus。
專題目標：以管線平行（Pipeline Parallelism）在 4 張板子上執行 Gemma 31B Dense，並證明其並發容量與總吞吐量優於「單板」與「4 板各跑一份模型（資料平行）」。
企劃書與數據分析在另一個 Claude 對話中維護；你負責在板子上動手做實驗、改程式、回報結果。

---

## 1. 連線方式

- 以 `ssh node1 '指令'` 操作板子（設定在本機 `~/.ssh/config`）。不需要密碼；若跳出密碼提示，停下來請使用者處理。
- **每次 `ssh` 都是新的 shell**：變數、`cd` 不會保留。每個指令都要寫完整，例如：
  `ssh node1 'cd ~/llama.cpp && ./build-cpu-opt/bin/llama-bench -m models/gemma-4-31B-it-Q4_0.gguf ...'`
- 本機若是 Windows（Git Bash / PowerShell），注意引號：外層用單引號包住遠端指令。

| 節點 | IP | 角色 | 已知模型 | llama.cpp commit |
| --- | --- | --- | --- | --- |
| node1 | 192.168.137.30（`enp97s0`，100Mb/s） | Master + Stage 0 | gemma-4-31B-it-Q4_0、gemma-4-26B-A4B-it-Q4_0 | 330f0189c（已確認，2026-11-08） |
| node2 | 192.168.137.31（`enp97s0`，100Mb/s）；hostname 已改為 `node2` | Stage 1 | gemma-4-31B-it-Q4_0（2026-11-10 從 node1 複製，SHA-256 相同）、Llama-3.3-70B-Instruct-IQ2_M | 330f0189c（已確認，2026-11-10） |
| node3 | TODO | Stage 2 | TODO | TODO |
| node4 | TODO | Stage 3 | TODO | TODO |

node2 即先前記錄的「IP 曾為 192.168.137.31、放有 Qwen2.5-7B 與 Llama-3.3-70B」的那張板子。Qwen2.5-7B 已於 2026-11-10 經使用者同意刪除。系統碟同為 USB 隨身碟（54 GB），無 swap，時鐘無法 NTP 同步（已手動校正）。

切層版本：兩張板子都有 `~/llama.cpp-pp`（git worktree，branch `pp-layer-range`，尚未 commit）與 `build-pp`（同第 4 節參數）。內含 `llama-stage-test`（同程序兩段驗證）與 `llama-pp-stage`（跨程序 / 跨板 TCP 兩段）。改動也存在本機 `results/patches/`。
第一次連線時，請先在每張板子執行 `hostname; ip -br addr; git -C ~/llama.cpp rev-parse --short HEAD; ls -lh ~/llama.cpp/models/*.gguf | grep -v vocab; df -h /`，把上表補完並回報。

---

## 2. 硬體事實（已實測）

- SoC：CIX P1（CD8160）。CPU：8 × Cortex-A720（2.2–2.6 GHz，編號 0、1、6–11）+ 4 × Cortex-A520（1.8 GHz，編號 2–5）。
- 記憶體：系統可見 30 GiB，開機後可用約 24 GiB。
- GPU：Mali-G720，Vulkan 驅動不支援整數點積，**比 CPU 慢，不使用**。
- 網路：雙 5GbE（`enp49s0`、`enp97s0`，Realtek RTL8126，驅動 `r8126`）。`enp97s0` 接交換器（只支援 100Mb/s，待換），用於管理與上網。
- **2026-11-13 起改用 2.5GbE 交換器**：兩板 `enp97s0` 協商到 2500Mb/s，iperf3 雙向 2.35 Gbit/s、0 重傳，小封包 RTT 0.51 ms。**板子之間的資料傳輸走交換器的 192.168.137.x**。剛開機的前 1–2 分鐘頻寬不穩，量測前先等開機完成。
- 直連線（`enp49s0`）目前已拔除。設定仍保留在 NetworkManager「Wired connection 1」（node1 `10.0.0.1`、node2 `10.0.0.2`），插回就能用。曾測到 5000Mb/s、4.71 Gbit/s。
- 板子沒有 RTC 電池且無法 NTP，**每次開機時鐘都會停在上次關機的時間**，log 的時間戳記會錯，需要手動對時。
- 系統：Ubuntu 24.04、GCC 13.3。

---

## 3. 必守規則（這些都是踩過的坑）

1. **只用 `build-cpu-opt`。** 含 Vulkan 的 `build` 會讓 CPU 運算慢約 35 倍；舊的 `build-cpu` 沒有開加速指令。
2. **量測前確認 CPU 調頻模式**：`cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor` 必須全部是 `performance`。若不是，請使用者在板子上執行 `~/set_perf.sh`（需要 sudo 密碼，你無法代為輸入）。
3. **標準量測參數**：`taskset -c 0,1,6-11`、`-t 8`、`-r 5`、`-o json`（看每次的個別數值）。`llama-bench` 加 `-lm none`（此版本用 `-lm`，不是 `-mmp`）。MoE 的 prompt 處理一律用 `-p 512`（短 prompt 會因專家負載不均而大幅波動）。
4. **長時間實驗在板子背景執行**，不依賴 SSH 連線：
   `ssh node1 'cd ~/llama.cpp && mkdir -p results && nohup <指令> > results/<名稱>_$(date +%m%d_%H%M).log 2>&1 &'`
   之後用 `ssh node1 'tail -5 ~/llama.cpp/results/<檔名>'` 或 `pgrep -a llama` 檢查進度。
5. **永遠先存完整輸出，再從檔案 grep。** 不要直接 `指令 | grep`，錯誤訊息會被過濾掉（已發生兩次）。每次都檢查結束代碼：`127` 找不到指令、`130` 被中斷、`137` 記憶體不足。
6. **同一張板子一次只跑一個量測。** 開始前先 `pgrep -a llama` 確認沒有其他測試在跑。
7. **31B 的上下文不要設到 65536 以上。** 接近記憶體上限時，模型權重會反覆從硬碟讀回，單次載入要 10–21 分鐘；131072 會直接記憶體不足。
8. **每筆結果都記錄**：節點名稱、llama.cpp commit、指令全文、時間。
9. **安全**：不要刪除 `models/` 下的任何檔案；`git checkout` 前先 `git status`，有本地修改就停下來問使用者；不要修改系統設定；任何刪除或不可逆的動作先詢問。RPC server 沒有驗證機制，只能在叢集獨立網段使用。

---

## 4. 編譯參數（四張板子相同）

GCC 13.3 不認得 Cortex-A720，`-mcpu=native` 會退回基本 ARMv8，所以必須手動指定：

```bash
cd ~/llama.cpp
cmake -B build-cpu-opt -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+fp16+i8mm
cmake --build build-cpu-opt --config Release -j 8
grep -r "armv8.6-a" build-cpu-opt --include=flags.make | head -1   # 有輸出才算成功
```

RPC worker 啟動：`taskset -c 0,1,6-11 ./build-cpu-opt/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -c`（commit 330f0189c 的執行檔名稱是 `ggml-rpc-server`，不是 `rpc-server`）

---

## 5. 目前的實測結果（摘要）

| 項目 | 結果 |
| --- | --- |
| 加速指令修正（Gemma 26B-A4B） | pp128 16.2 → 約 48–66 t/s；tg64 10.09 → 13.67 t/s |
| 31B 單板批次（llama-batched-bench，每人 pp128 / tg32） | 總生成吞吐：B=1 2.07、B=2 2.50、B=4 7.44、B=8 8.51、B=16 11.15、B=32 10.51 t/s |
| 31B prompt 處理 | 約 13–15.7 t/s，不隨並發數變化 |
| 31B KV Cache | 10 層完整注意力（第 5、11…59 層）每 token 80 KiB；50 層滑動視窗（1024）每 token 800 KiB，每人封頂約 800 MiB |
| 31B 單板 KV 上限 | 約 6–7 GB；c=65536 成功，c=131072 記憶體不足 |
| 31B 模型參數 | 60 層、hidden size 5376、Q4_0 約 17 GB |
| 31B 跨板兩段（node1 0–30 層 → 5G 直連 → node2 30–60 層） | 與單板逐位元相同；每 token 前段 0.227 s + 後段 0.271 s + 網路 0.001 s = 2.00 t/s（B=1，未流水）；每板 RSS 約 9.4 GB，剩約 15 GB（`results/6.7_cross_board_pp.md`） |
| 31B 跨板流水（`performance`） | 1x1 2.10 t/s → 2 個請求流水 2x2 **3.83 t/s**（1.82 倍）（`results/6.8_pipeline_sweep.md`） |
| 31B 跨板批次 + 流水（`kv_unified` 修正後，切在第 30 層） | 4x1 8.08、8x2 14.79、16x1 13.08、**16x2 17.08 t/s**（單板 B=16 的 1.38 倍）；後段 99% 忙碌是瓶頸。32x2 因 node2 當機未完成（`results/6.9_pipeline_sweep_fixed.md`） |

**進度總報告**：`results/REPORT_2026-11-13.md`。

**node2 供電不足（2026-11-13 確認）**：node2 的電源線只有 60 W，長時間 8 核滿載（流水的後段 99% 忙碌）時會整台當機，兩次都發生在大批次組態（當機前溫度最高 74°C、記憶體充足）。**換成足夠的電源（例如 100 W）之前，不要在 node2 跑長時間滿載的實驗**；當機後 node1 的前段會一直等待，要用 `pgrep -x llama-pp-stage` 找到 PID 停掉。

**換好電源後，先做這兩件事：**
1. **補跑 32x2**：切在第 30 層，`--configs 32x2`，其他參數同 `results/6.9_pipeline_sweep_fixed.md`。
2. **跑 pp5（非對稱切層）**：`-s 33`，`--configs 1x1,2x2,16x2,32x2`。後段走 `192.168.137.31:9000`。標準答案 `/tmp/stage_ref_31b.bin` 開機後會消失，要用 `llama-stage-test --mode ref` 重新產生。

兩張板子開機後都要執行 `~/set_perf.sh`（需要 sudo），調頻才會是 `performance`。監控腳本 `~/wd_mem.sh` 會記錄最高溫度，並每筆立即寫入磁碟。

---

## 6. 本週待辦（依優先順序）

完成每一項後，在 `~/llama.cpp/results/` 留下原始記錄，並在本機產生一份簡短的 markdown 摘要（表格 + 原始檔路徑 + 你的解讀），方便使用者帶回規劃對話。

### 6.1 補齊節點資訊
執行第 1 節的檢查指令，補完節點表。

### 6.2 並發 2 異常偏低的原因
B=2 只比 B=1 快 20%，B=4 卻快 3.6 倍，兩次量測皆重現。假設：`i8mm` 加速路徑以 4 列為單位，批次未達 4 時退回慢路徑。

```bash
taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m models/gemma-4-31B-it-Q4_0.gguf \
  -c 4096 -npp 128 -ntg 32 -npl 1,1,2,3,4,5,6,8 -t 8
```

第一列為暖機，不採用。回報 S_TG 是否在 4 的倍數時明顯較高。若成立，可再到 ggml CPU 後端原始碼中找出對應的批次門檻作為佐證。

### 6.3 四板統一 llama.cpp 版本
目標 commit：`330f0189c`。每張板子：`git status`（有修改就停下來問）→ `git fetch origin` → `git checkout 330f0189c` → 刪除並重建 `build-cpu-opt`（第 4 節）→ 驗證。若出現 `reference is not a tree`，先 `git fetch --unshallow`。需要板子能連外網。

### 6.4 修正後版本的單板數據重測（在 node1）
- 31B 執行緒掃描：`llama-bench -t 2,4,6,8 -p 64 -n 32 -r 3 -lm none`（綁 8 大核）
- MoE 8 核 vs 12 核：`taskset -c 0,1,6-11 ... -t 8` 與 `taskset -c 0-11 ... -t 12`，各 `-p 512 -n 64 -r 5 -lm none`
- 分開量測 `performance` 與 `-lm none` 各自的效果（請使用者切換調頻模式）

### 6.5 節點間網路延遲
在兩張板子之間量 `ping -c 50` 與 `iperf3`（若未安裝，先確認能否連外網再請使用者安裝）。記錄 `ethtool enp49s0 | grep Speed`。

### 6.6 切層可行性驗證（風險最高，上限 2–3 天）
目標：讓一個程序只執行模型的第 a 到 b 層，輸入隱藏狀態、輸出隱藏狀態，串接兩段後的最終 logits 與單板完整模型一致（容許數值誤差）。
候選路線：`llama_batch` 的 `embd` 欄位可直接輸入向量；修改 Gemma 的模型建構程式碼，使其支援指定層範圍並跳過首尾的 Embedding / LM Head。注意 Gemma 的滑動視窗注意力、各層 KV head 數不同、embedding 縮放等細節。
先閱讀原始碼並提出實作方案與工作量估計，經使用者同意後再開始修改。改動放在獨立的 git branch，不要動 `master`。
若 3 天內無法讓兩段輸出與完整模型一致，回報卡住的地方，專題將改走 llama.cpp 原生 RPC 的備案。

---

## 7. 回報方式

- 數字要附單位、重複次數、標準差或 5 次個別數值。
- 明確區分「實測」與「推測」。推測要說明如何驗證。
- 出現異常結果時，先檢查第 3 節的規則是否都有遵守（版本、調頻、綁核、是否有其他程式在跑），再下結論。
