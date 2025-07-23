# 半小时本地运行自己的“ChatGPT”


平时我们在手机和电脑上使用 AI 助手比如 ChatGPT 或者豆包、kimi 等都是在线使用的。但是部分场景下我们也需要模型能够运行在本地，比如办公电脑访问外部网站不方便的时候，我们可以在自己的电脑上运行一个合适的大模型。或者自己开发的应用、网站中需要使用大模型的API，如果用付费的 API，可能会花费不少的成本还有并发数量等的限制，这时候可以在本地或者服务器部署一个大模型的服务，提供 API 出来供上层的应用使用。

这篇文章写一下本地运行大模型的方案，其中这里面的多种方案，都是和 OpenAI 的 API 协议兼容的，运行在本地之后，相当于我们本地有了一个离线版的 ChatGPT。

这里有多种方案，我们按照客户端和 API 的方式大致做个分类。

## 客户端的方式运行
这种方式一般是在本地下载一个应用，在应用中下载大模型，然后选择对应的大模型来使用。GitHub 上有不少类似的项目，比如 [Jan](https://github.com/janhq/jan) 和 [Ollama](https://github.com/ollama/ollama)。Ollama 官方应用没有附带直接使用的前端界面，Jan 应用带了类 ChatGPT 的前端界面，我们这里主要以 Jan 来举例，其实他们的原理是类似的。

我们在 Jan 的[官网](https://jan.ai/)下载客户端，Jan 提供了 Mac 和 Windows、Linux 的版本，选择适合自己系统的版本下载即可。安装完成之后，第一次打开使用是没有模型的，我们可以在下面的界面中选择对应的模型下载（下面的截图是 Mac 版，Windows 和 Linux 中可能会有细微的差异）。

如果下面的列表中没有我们想要的模型，可以提供一个 Hugging Face 的 URL 来下载模型。

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1736474156185-ec6c981e-b0f0-4ead-955f-f52536b5f940.png)

不同配置的设备，能运行的模型规模不同，模型后面有相应的性能提示，我们可以根据自己的设备配置选择合适的模型下载。在没有显卡的设备上，使用 CPU 进行推理，速度会比较慢，有 GPU 的机器上，会用 GPU 进行推理。CPU 推理底层是使用了 [llama.cpp](https://github.com/ggerganov/llama.cpp) 来提供支持。

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1736474204816-f2034790-5c6c-4e6d-8d1d-2d5417047197.png)

下载完成模型之后，点击上图的 Use 去使用，这里用了 Qwen 2.5 7B 的 Q4 量化模型做了举例，在 Mac 的 m1 pro 芯片上运行速度还可以接受。（Apple Silicon系列的芯片默认使用 Metal 进行 GPU 加速，推理速度比 Intel 版本的会更快。）

在下图右侧，我们可以直接调整 Temperature、Max Tokens 长度等参数来影响模型的输出结果。

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1736474427205-184ee27f-50e8-4d6f-9005-dc65008891dc.png)

按照这种方式，我们可以很快的获得一个“离线版的 ChatGPT 客户端”。

## API的方式运行
如果我们想要本地运行的大模型，还要能提供 API 接口供上层应用使用，也有多种方式可供选择，下面列举几种。

### Python + Web 框架
这种方式是我们自己开发 Python 应用，在代码中和大模型进行交互，通过 Web 框架提供 API 出来。这种方式开发成本相对高一些。

下面是 Python 使用 Qwen 模型的方式，我们再结合 FastAPI 等 Web 框架即可提供 API 供前端应用使用，这种方式灵活度比较高，但是开发和使用成本会高一些。

```jsx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Ollama
上面我们有说过 Ollama，下载 Ollama 客户端运行之后，我们可以在终端中运行 ollama run llama3.3 来和大模型对话，同时 Ollama 也提供了 API 可供调用，使用方式：

```jsx
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

### Cortex
[Cortex](https://cortex.so/) 是 Jan 底层依赖的一个库，Cortex 底层依赖了 llama.cpp。Cortex 也提供了安装包，下载运行之后，我们可以按照 [文档](https://cortex.so/api-reference#tag/chat) 中的方式访问 API。Cortex 提供的 API 是兼容 OpenAI 的 API 格式的，使用起来也比较方便。

```jsx
import requests

url = "http://127.0.0.1:39281/v1/chat/completions"

payload = {
    "messages": [],
    "model": "mistral",
    "stream": True,
    "max_tokens": 4096,
    "stop": ["End"],
    "frequency_penalty": 0.2,
    "presence_penalty": 0.6,
    "temperature": 0.8,
    "top_p": 0.95,
    "modalities": ["text"],
    "audio": {
        "voice": "",
        "format": "mp3"
    },
    "store": False,
    "metadata": { "type": "conversation" },
    "logit_bias": {
        "15496": -100,
        "51561": -100
    },
    "logprobs": False,
    "n": 1,
    "response_format": { "type": "text" },
    "seed": 123,
    "stream_options": None,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "",
                "parameters": {},
                "strict": False
            }
        }
    ],
    "parallel_tool_calls": True
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
```

### vLLM
[vLLM](https://github.com/vllm-project/vllm) 是一个高性能的大模型推理引擎，不只是可以在本地使用，在生产环境也可以使用 vLLM 来部署大模型对外提供 API 服务。vLLM 对高并发等场景做了优化，性能比较好，同时 API 也兼容了 OpenAI 的 SDK，我们优先使用这种方式对外提供 API。

我们继续用 Qwen 的模型举例子，可以在这里看 [参考文档](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)。

先安装 python 的包：

```jsx
pip install vllm
```

运行 Qwen 2.5 7B模型：

```jsx
vllm serve Qwen/Qwen2.5-7B-Instruct
```

注意，上面举例的 Jan 客户端中运行的模型是 Q4 量化后的，这里的模型是原始的版本，效果会更好，也会更消耗设备的性能。 如果设备配置不够，可以运行更小参数规模的版本，比如 0.5B 或1.5B等。

通过上面这个命令，我们就运行起来了一个和 OpenAI 的 API 格式兼容的服务，我们可以按照下面的方式调用 API：

```jsx
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```

因为 API 和 OpenAI 兼容，我们也可以直接使用 OpenAI 的 SDK，只需要替换掉包中的 base_url 和 api_key，同时指定 model 名称即可：

```jsx
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
```

按照上面这种方式，我们可以很方便的切换 OpenAI 的远端模型和本地设备运行的离线模型。比如对效果要求比较高的场景，我们可以指定模型为 OpenAI 的 gpt4、o1 等模型，其他的场景指定模型为本地运行的模型，来节省一定的 API 费用。

vLLM 运行起来之后，我们在代码中写一个异步调用的循环，来简单测试一下 vLLM 的并发优化效果（非严谨测试）。

Node.js 代码：

```jsx
import { OpenAI } from "openai";

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'EMPTY',
});

// 问题列表是用 gpt 生成的，100 条数据，这里不再全部列出
let questionList = [
  "What is the capital of France?",
  "Explain the concept of quantum entanglement.",
  "What are the differences between HTTP and HTTPS?",
  // ...,
  // ...,
]

async function init(question: string) {
  let chat_response = openai.chat.completions.create({
    model: "Qwen/Qwen2.5-1.5B-Instruct", // 这里用的 1.5B 做测试
    messages: [
      { "role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." },
      { "role": "user", "content": question || "hello" },
    ],
    temperature: 0.7,
    top_p: 0.8,
    max_tokens: 5000,
  }).then((res) => {
    log('res ' + (suc++), res);
  }).catch((err) => {
    log('err' + (err++), err);
  })
}

for (let i = 0; i < questionList.length; i++) {
  let q = questionList[i];
  init(q);
}
```

上面是使用 Node.js 在 for 循环中，发起了对 vLLM 服务的请求调用，questionList 的长度为 100，相当于短时间内发起了 100 个请求，请求很快都全部返回了。

下面是 vLLM 服务在控制台输出的日志，可以看到，生成的速度为：2069.1 tokens/s，运行请求：46 个，这个速度是非常快的。上面我用 Mac 版本的 Jan 发送消息，大致为 30 tokens/s。（备注：该对比为非严谨对比，使用的模型版本、调用方式等都不同，但是可以看出来 vLLM 的并发性能是很好的。）

```jsx
Avg prompt throughput: 730.8 tokens/s, Avg generation throughput: 2069.1 tokens/s, Running: 46 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 1.6%, CPU KV cache usage: 0.0%.
```

本次测试设备配置：Nvidia 4090 GPU + Intel 13 代 i7 CPU + 32GB RAM，操作系统：Ubuntu 22.04。

通过这个简单的测试，看来通过 vLLM 部署大模型，单台 4090 设备就可以支持小用户规模下的并发使用了。

## 小结
上面讲了通过客户端和 API 的方式，使用本地运行的大模型，整个过程半小时左右就可以搞定。同时 API 兼容 OpenAI 的格式，这样我们可以直接使用 OpenAI 官方的 SDK 来开发我们自己的应用，非常方便，相当于拥有了一个本地运行的 ChatGPT。

这篇文章就先更新到这里，如果有问题可以评论区留言，也欢迎大家点赞收藏关注，后续我会继续更新其他内容，谢谢～

