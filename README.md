# LMDeploy
## 1.大模型部署背景

### 模型部署

定义：将训练好的模型在特定软硬件环境中启动的过程，使模型能够接收输入并返回预测结果。为了性能和效率要求需要对模型进行优化，如模型压缩和硬件加速。

产品形态：云端、边缘计算端、移动端

计算设备：CPU、GPU、NPU、TPU

大模型特点：内存开销巨大，庞大参数量，需要缓存之前生成的K/V；动态shape;相对视觉模型，LLM结构简单。

### 部署挑战

设备：低存储设备部署；推理：加速token生成、动态shape推理不间断、有效管理利用内存；服务：吞吐量提高、平均响应时长

部署方案
![0](https://github.com/hui1feng/LMDeploy/assets/126125104/169d3576-49a8-43e6-8004-422ae1357033)


## 2.LMDeploy简介

LMDeploy是LLM在英伟达设备上部署的全流程解决方案。包括模型轻量化、推理和服务。
![1](https://github.com/hui1feng/LMDeploy/assets/126125104/5aa686f7-cb17-4413-8f95-db4141481429)


推理性能：LMDeploy遥遥领先
![2](https://github.com/hui1feng/LMDeploy/assets/126125104/dbed4ac4-610c-478c-8a95-be7165962a33)


LMDeploy核心功能-量化
![3](https://github.com/hui1feng/LMDeploy/assets/126125104/74dc76b7-4a28-4ce5-a391-50c75ae166d0)


做Weight Only量化原因：LLMs是显存密集型任务，大多数实践在生成Token阶段。一举两多得，将FP16模型权重降到1/4，降低访存成本，还增加了显存。
![4](https://github.com/hui1feng/LMDeploy/assets/126125104/9797c3bc-d814-4e7e-8b6f-8abc386933ee)


如何做？AWQ算法：4bi模型推理时权重反量化为FP16。比GPTQ更快。
![5](https://github.com/hui1feng/LMDeploy/assets/126125104/db0c8392-2123-42dd-ba59-89aec0039868)


核心功能-推理引擎TuboMind

1.持续批处理
![6](https://github.com/hui1feng/LMDeploy/assets/126125104/bdd66958-f2df-4584-a88f-827ed2a236e5)


2.有状态推理
![7](https://github.com/hui1feng/LMDeploy/assets/126125104/b1ac914c-948b-4875-8d79-b55006cb86cc)


3.高性能 cuda kernel
![8](https://github.com/hui1feng/LMDeploy/assets/126125104/5874860a-7164-4e23-8386-6df9784286d3)


4.Block k/v cache
![9](https://github.com/hui1feng/LMDeploy/assets/126125104/6651ec6b-d3c2-4653-a7a1-0fdc29ecd925)


LMDeploy核心功能-推理服务 api server

![10](https://github.com/hui1feng/LMDeploy/assets/126125104/3a3d2952-7d78-4466-9134-1e88f5a601b2)

## 3.手动实现
### 3.1创建环境
```python
#1.创建并激活环境
 conda activate lmdeploy
#2.安装lmdeploy,先下载依赖
# 解决 ModuleNotFoundError: No module named 'packaging' 问题
pip install packaging
# 使用 flash_attn 的预编译包解决安装过慢问题
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install 'lmdeploy[all]==v0.1.0'
```
### 3.2服务部署
![捕获1](https://github.com/hui1feng/LMDeploy/assets/126125104/f31136ca-c805-4b2a-914b-d5aa801c3b1c)
我们把从架构上把整个服务流程分成下面几个模块。

* 模型推理/服务。主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。
* Client。可以理解为前端，与用户交互的地方。
* API Server。一般作为前端的后端，提供与产品和服务相关的数据和功能支持。
值得说明的是，以上的划分是一个相对完整的模型，但在实际中这并不是绝对的。比如可以把“模型推理”和“API Server”合并，有的甚至是三个流程打包在一起提供服务。

接下来，我们看一下lmdeploy提供的部署功能。

### 3.3 模型转换
使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，目前支持在线转换和离线转换两种形式。在线转换可以直接加载 Huggingface 模型，离线转换需需要先保存模型再加载。

TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 FasterTransformer 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。

* 在线转换
直接启动本地的 Huggingface 模型，如下所示。
```python
lmdeploy chat turbomind /share/temp/model_repos/internlm-chat-7b/  --model-name internlm-chat-7b
```
以上命令都会启动一个本地对话界面，通过 Bash 可以与 LLM 进行对话,效果如下图所示。
![在线转化](https://github.com/hui1feng/LMDeploy/assets/126125104/3e472d89-1550-4e4f-8801-c087dd055c4f)

* 离线转换
```python
#离线转换需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。
# 转换模型（FastTransformer格式） TurboMind
lmdeploy convert internlm-chat-7b /path/to/internlm-chat-7b
#这里我们使用官方提供的模型文件，就在用户根目录执行，如下所示。
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/ 
```
执行完成后将会在当前目录生成一个 workspace 的文件夹。这里面包含的就是 TurboMind 和 Triton “模型推理”需要到的文件。weights 和 tokenizer 目录分别放的是拆分后的参数和 Tokenizer。如果我们进一步查看 weights 的目录，就会发现参数是按层和模块拆开的

### 3.4 TurboMind 推理+命令行本地对话
模型转换完成后，我们就具备了使用模型推理的条件，接下来就可以进行真正的模型推理环节。

我们先尝试本地对话（Bash Local Chat），下面用（Local Chat 表示）在这里其实是跳过 API Server 直接调用 TurboMind。简单来说，就是命令行代码直接执行 TurboMind。所以说，实际和前面的架构图是有区别的。

这里支持多种方式运行，比如Turbomind、PyTorch、DeepSpeed。但 PyTorch 和 DeepSpeed 调用的其实都是 Huggingface 的 Transformers 包，PyTorch表示原生的 Transformer 包，DeepSpeed 表示使用了 DeepSpeed 作为推理框架。Pytorch/DeepSpeed 目前功能都比较弱，不具备生产能力，不推荐使用。

建立workspace工作目录后执行命令如下。
```python
# Turbomind + Bash Local Chat
lmdeploy chat turbomind ./workspace
```
启动后就可以和它进行对话了。

### 3.5 TurboMind推理+API服务
在上面的部分我们尝试了直接用命令行启动 Client，接下来我们尝试如何运用 lmdepoy 进行服务化。

”模型推理/服务“目前提供了 Turbomind 和 TritonServer 两种服务化方式。此时，Server 是 TurboMind 或 TritonServer，API Server 可以提供对外的 API 服务。我们推荐使用 TurboMind，TritonServer 使用方式详见《附录1》。

首先，通过下面命令启动服务。
```python
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1
```
上面的参数中 server_name 和 server_port 分别表示服务地址和端口，tp 参数我们之前已经提到过了，表示 Tensor 并行。还剩下一个 instance_num 参数，表示实例数，可以理解成 Batch 的大小。
然后，我们可以新开一个窗口，执行下面的 Client 命令。如果使用官方机器，可以打开 vscode 的 Terminal，执行下面的命令。
```python
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:23333
```
刚刚启动的是 API Server，自然也有相应的接口。可以直接打开 http://{host}:23333 查看，如下图所示。
![5 1](https://github.com/hui1feng/LMDeploy/assets/126125104/d60605cc-3863-4b5e-a60b-ec683b17b6e7)

![5 2](https://github.com/hui1feng/LMDeploy/assets/126125104/59297aa3-faba-43e8-9cc5-21d3185b6921)

### 3.6 网页 Demo 演示
这一部分主要是将 Gradio 作为前端 Demo 演示。在上一节的基础上，我们不执行后面的 api_client 或 triton_client，而是执行 gradio。
API Server 的启动和上一节一样，这里直接启动作为前端的 Gradio。
```python
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True
```
结果如下图所示。
![5 5](https://github.com/hui1feng/LMDeploy/assets/126125104/f752f04f-8e48-46e8-ab85-3761a1ca606e)

当然，Gradio 也可以直接和 TurboMind 连接，如下所示。
```python
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```
可以直接启动 Gradio，此时没有 API Server，TurboMind 直接与 Gradio 通信。如下图所示。

### 3.7 模型量化
本部分内容主要介绍如何对模型进行量化。主要包括 KV Cache 量化和模型参数量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。

正式介绍 LMDeploy 量化方案前，需要先介绍两个概念：
* 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速。
* 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。
常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。

那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用 KV Cache 量化和 4bit Weight Only 量化（W4A16）。KV Cache 量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。4bit Weight 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

#### 3.7.1 KV Cache 量化
步骤:KV Cache 量化是将已经生成序列的 KV 变成 Int8，使用过程一共包括三步：

第一步：计算 minmax。主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。

* 对于 Attention 的 K 和 V：取每个 Head 各自维度在所有Token的最大、最小和绝对值最大值。对每一层来说，上面三组值都是 (num_heads, head_dim) 的矩阵。这里的统计结果将用于本小节的 KV Cache。
* 对于模型每层的输入：取对应维度的最大、最小、均值、绝对值最大和绝对值均值。每一层每个位置的输入都有对应的统计值，它们大多是 (hidden_dim, ) 的一维向量，当然在 FFN 层由于结构是先变宽后恢复，因此恢复的位置维度并不相同。这里的统计结果用于下个小节的模型参数量化，主要用在缩放环节（回顾PPT内容）。

第一步执行命令如下：
```python
# 计算 minmax
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_output
```
在这个命令行中，会选择 128 条输入样本，每条样本长度为 2048，数据集选择 C4，输入模型后就会得到上面的各种统计值。值得说明的是，如果显存不足，可以适当调小 samples 的数量或 sample 的长度。
第二步：通过 minmax 获取量化参数。主要就是利用下面这个公式，获取每一层的 K V 中心值（zp）和缩放值（scale）。有这两个值就可以进行量化和解量化操作了。具体来说，就是对历史的 K 和 V 存储 quant 后的值，使用时在 dequant。
```python
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```
第二步的执行命令如下：
```python
# 通过 minmax 获取量化参数
lmdeploy lite kv_qparams \
  --work_dir ./quant_output  \
  --turbomind_dir workspace/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1
```
在这个命令中，num_tp 的含义前面介绍过，表示 Tensor 的并行数。每一层的中心值和缩放值会存储到 workspace 的参数目录中以便后续使用。kv_sym 为 True 时会使用另一种（对称）量化方法，它用到了第一步存储的绝对值最大值，而不是最大值和最小值。

第三步：修改配置。也就是修改 weights/config.ini 文件，这个我们在《2.6.2 模型配置实践》中已经提到过了（KV int8 开关），只需要把 quant_policy 改为 4 即可。

这一步需要额外说明的是，如果用的是 TurboMind1.0，还需要修改参数 use_context_fmha，将其改为 0。

接下来就可以正常运行前面的各种服务了，只不过咱们现在可是用上了 KV Cache 量化，能更省（运行时）显存了。

量化会导致一定的误差，有时候这种误差可能会减少模型对训练数据的拟合，从而提高泛化性能。量化可以被视为引入轻微噪声的正则化方法。或者，也有可能量化后的模型正好对某些数据集具有更好的性能。总结一下，KV Cache 量化既能明显降低显存占用，还有可能同时带来精准度（Accuracy）的提升。

#### 3.7.2 W4A16 量化
W4A16中的A是指Activation，保持FP16，只对参数进行 4bit 量化。使用过程也可以看作是三步。

第一步：同 1.3.1，不再赘述。

第二步：量化权重模型。利用第一步得到的统计值对参数进行量化，具体又包括两小步：

缩放参数。主要是性能上的考虑（回顾 PPT）。
整体量化。
第二步的执行命令如下：
```python
# 量化权重模型
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output 
```
命令中 w_bits 表示量化的位数，w_group_size 表示量化分组统计的尺寸，work_dir 是量化后模型输出的位置。这里需要特别说明的是，因为没有 torch.int4，所以实际存储时，8个 4bit 权重会被打包到一个 int32 值中。所以，如果你把这部分量化后的参数加载进来就会发现它们是 int32 类型的。

最后一步：转换成 TurboMind 格式。
```python
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128
```
这个 group-size 就是上一步的那个 w_group_size。如果不想和之前的 workspace 重复，可以指定输出目录：--dst_path，比如：
```python
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128 \
    --dst_path ./workspace_quant
```
量化模型和 KV Cache 量化也可以一起使用，以达到最大限度节省显存。W4A16 参数量化后能极大地降低显存，同时相比其他框架推理速度具有明显优势。

参考资料：
https://github.com/InternLM/tutorial/blob/main/lmdeploy/lmdeploy.md
https://www.bilibili.com/video/BV1iW4y1A77P/?spm_id_from=333.999.0.0&vd_source=1451efee9af1bf2214b4d072f5760564
