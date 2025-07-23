## 借助 ChatGPT 开发智能家居助手 - 指令解析篇

现在的语音助手应用很广泛，比如小米的音响可以控制小米的智能家居，新能源汽车比如理想、小鹏等在车中可以用语音控制车中的设备，打开/关闭车窗、设置空调温度、播放音乐等。

之前的时候，想把语音助手开发的比较好，难度还是比较大的。想象一下一些老款的燃油车中带的语音助手，说的稍微复杂点或者没有严格按照它的提示词来说，就根本识别不了我想让它做什么。

现在有了 chatGPT，我们就可以相对比较低门槛的去完成一个语音助手，来帮助我们控制智能设备。

我们这一篇暂时不涉及到语音，先通过文字描述我们的需求，最终转换成想要的指令。有了指令之后，我们就可以通过智能设备提供的接口去控制相应的设备了。

### 关键词匹配指令
现在我们以空调举例子，比如我会说：打开空调，把空调调到 26 度；把空调关掉；调度再调高 2 度；温度调低2 度。最简单的方式，可能是通过关键词匹配，我们可以去搜索句子中的关键词，比如“打开”和“关闭”，发现有关键词之后，我们就可以知道用户想要做什么。但是可能不同的人表达方式可能不同，比如说：打开空调，开启空调，启动空调，把空调开起来等等，通过关键词匹配我们需要去处理每一种表达方式，也很难匹配完整。所以我们需要换一种方式。

### 使用ChatGPT提取指令
ChatGPT 可以帮我们做很多事情，我们试下看 ChatGPT 能否帮我们完成这件事。

我们描述下我们的需求，在网页中发送给 ChatGPT，看下 ChatGPT 的处理结果。

下面是我发送给它的话。

```plain
我需要你帮我完成一个语音助手，分析用户的话，并给我指令。比如用户说：打开空调，设置到 26 摄氏度。你分析这句话，给我结构化的数据，让我能完成用户的命令。
```

忽略掉多余的内容，只关注指令，它发送给我的指令是这样的：

```plain
{
  "action": "turn_on",
  "device": "air_conditioner",
  "settings": {
    "temperature": 26
  }
}
```

我再给它说：

```plain
把温度调高 2 度
```

它发送给我的指令：

```plain
{
  "action": "increase_temperature",
  "device": "air_conditioner",
  "value": 2
}
```

看起来是基本符合我们的需求的。

上面我是通过网页发送给 ChatGPT 的，但是我没办法把网页上的回答和后续的控制设备的功能进行串联起来，所以我们需要调用 OpenAI 的 API 来完成这个事情。

### 对接 API
如果没有账号，需要先注册一个账号，然后在[这里](https://platform.openai.com/settings/organization/api-keys)申请 API key。同时需要关联一个信用卡或者充值金额到账户中。

有了 API 之后，就可以在代码中调用来看模型给我们的回复了。

我们的代码使用 Node.js + TypeScript 来做演示。

在项目中创建一个.env文件，内容如下：

```bash
OPENAI_API_KEY=sk-xxxx
```

把里面的 key 替换成你自己的。

下面是我们的代码，TypeScript 版本。

```typescript
import { OpenAI } from "openai";
import * as dotenv from 'dotenv';
dotenv.config();
const openai = new OpenAI({
  apiKey: process.env['OPENAI_API_KEY'],
});

async function main() {
  const stream = await openai.chat.completions.create({
    messages: [{ role: 'user', content: '你好' }],
    model: 'gpt-4o',
    stream: false,
  });
  console.log('steam', stream);
}

main();

```

输出的结果大致是下面这样：

```typescript
{
  id: 'id',
  object: 'chat.completion',
  created: 1735400613,
  model: 'gpt-4o-2024-08-06',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: '你好！有什么我可以帮助你的吗？', refusal: null },
      logprobs: null,
      finish_reason: 'stop'
    }
  ],
  usage: {
    prompt_tokens: 8,
    completion_tokens: 10,
    total_tokens: 18,
    prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
    completion_tokens_details: {
      reasoning_tokens: 0,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0
    }
  },
  system_fingerprint: 'fp_xxxx'
}
```

现在我们开始描述我们的需求：

```typescript
const stream = await openai.chat.completions.create({
    messages: [{ role: 'user', content: '你是个智能家居的设备，可以分析用户的话，拆解成动作指令。现在开始：打开空调，并设置到 26度，' }],
    model: 'gpt-4o',
    stream: false,
  });
```

忽略掉上面重复的部分，输出如下：

```typescript
choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: '好的，我来帮你拆解这个指令：\n' +
          '\n' +
          '1. **打开空调**：启动空调设备。\n' +
          '2. **设置温度到26度**：将空调的温度调节到26摄氏度。\n' +
          '\n' +
          '请确认这些步骤是否正确，或者是否有其他需要调整的地方。',
        refusal: null
      },
      logprobs: null,
      finish_reason: 'stop'
    }
  ],
```

输出的步骤是正确的，但是我们需要更结构化的格式，供我们的代码调用，现在这种格式，我们后续的程序无法识别并执行。

### 改造为Function calling
查看 OpenAI 的文档，文档中有描述 [Function calling](https://platform.openai.com/docs/guides/function-calling?lang=node.js) 的功能，可以按照我们指定的格式输出想要的指令。相比上面的调用，需要再增加一个 tools 字段。

#### 输出为对象
我们按照官方文档进行改造，现在是这样的：

```typescript
let tools = [
  {
    type: "function",
    function: {
      name: 'smartHome',
      description: 'smart home device action',
      parameters: {
        type: "object",
        properties: {
          action: { type: "string" },
          device: { type: 'string' },
          setting: { type: 'string' },
          value: { type: 'string' },
        },
      },
    },
  },
];

const stream = await openai.chat.completions.create({
    messages: [{ role: 'user', content: '你是个智能家居的设备，可以分析用户的话，拆解成动作指令。现在开始：打开空调，并设置到 26度，' }],
    model: 'gpt-4o',
    stream: false,
    // @ts-ignore
    tools,
  });

```

模型的输出：

```typescript
choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_111', // 备注：id 这里做了修改，真实的输出是一个随机字符串
            type: 'function',
            function: {
              name: 'smartHome',
              arguments: '{"action": "打开", "device": "空调"}'
            }
          },
          {
            id: 'call_112',
            type: 'function',
            function: {
              name: 'smartHome',
              arguments: '{"action": "设置", "device": "空调", "setting": "温度", "value": "26度"}'
            }
          }
        ],
        refusal: null
      },
      logprobs: null,
      finish_reason: 'tool_calls'
    }
  ],
```

在输出的tool_calls字段中，我们可以看到模型输出了指令，并且格式是我们在 tools 中设置的格式。

上面的 tools 目前类型是个对象，因为我们的指令可能很长，需要有多个步骤，所以我们把它改造成数组。

#### 输出为数组
主要是修改下面的 tools 的格式，添加 list 字段指定为 array 类型，在里面增加 items 字段。

```typescript
let tools = [
  {
    type: "function",
    function: {
      name: 'smartHome',
      description: 'smart home device action',
      parameters: {
        type: "object",
        properties: {
          list: {
            type: 'array', // 增加数组类型
            items: { // 增加items字段
              properties: {
                action: { type: "string" },
                device: { type: 'string' },
                setting: { type: 'string' },
                value: { type: 'string' },
              }
            }
          }
        },
      },
    },
  },
];

```

同时完善下我们的提示词：

```typescript
你是个智能家居的设备，可以分析用户的话，拆解成动作指令。需要有action，setting，device，value。可能会有多个动作。把set_temperature变成action: set, setting: temperature。现在开始：打开空调，并设置到 26度，
```

现在的输出：

```typescript
choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_1',
            type: 'function',
            function: {
              name: 'smartHome',
              arguments: '{"list": [{"action": "turn_on", "device": "air_conditioner"}]}'
            }
          },
          {
            id: 'call_2',
            type: 'function',
            function: {
              name: 'smartHome',
              arguments: '{"list": [{"action": "set", "setting": "temperature", "value": 26, "device": "air_conditioner"}]}'
            }
          }
        ],
        refusal: null
      },
      logprobs: null,
      finish_reason: 'tool_calls'
    }
  ],
```

#### 增加多条指令
现在继续增加我们的指令，让它完成多件事情，看看效果。

```typescript
你是个智能家居的设备，可以分析用户的话，拆解成动作指令。需要有action,setting,device,value字段。可能会有多个动作以及多个设备。空调的action: set_temperature变成action: set, setting: temperature。现在开始：打开空调，并设置到26度，再增加 2 度。再关闭窗帘，再关闭灯光。
```

```typescript
choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_333',
            type: 'function',
            function: {
              name: 'smartHome',
              arguments: '{"list": [{"action": "turn_on", "device": "air_conditioner"}, {"action": "set", "setting": "temperature", "value": 26, "device": "air_conditioner"}, {"action": "increase", "setting": "temperature", "value": 2, "device": "air_conditioner"}]}'
            }
          },
          {
            id: 'call_444',
            type: 'function',
            function: {
              name: 'smartHome',
              arguments: '{"list": [{"action": "turn_off", "device": "curtain"}, {"action": "turn_off", "device": "light"}]}'
            }
          }
        ],
        refusal: null
      },
      logprobs: null,
      finish_reason: 'tool_calls'
    }
  ],
```

现在可以看到，这些指令基本上可以满足我们的需求。

#### 解析指令
模型输出的是字符串，我们做一定的解析处理。

下面是解析的 util 代码。

```typescript
interface IFuncItem {
    name: string;
    arguments: { [key: string]: any } | null
}
export function getFuncInfoFromMsg(msg: API.ChatCompletionMessage): (IFuncItem[]) {
    let arr = [];
    if (msg.tool_calls) {
        for (const funcItem of msg.tool_calls) {
            let functionName: string = funcItem.function?.name as string;
            let argStr = funcItem.function?.arguments;
            let funcInfo = {
                name: functionName,
                arguments: null,
            }
            if (argStr) {
                let argObj = JSON.parse(argStr);
                funcInfo.arguments = argObj;
                arr.push(funcInfo);
            }
        } 
    }
    return arr;
}
```

在原来的的代码中添加上解析指令的代码：

```typescript
async function main() {
  const stream = await openai.chat.completions.create({
    messages: [{ role: 'user', content: '你是个智能家居的设备，可以分析用户的话，拆解成动作指令。需要有action,setting,device,value字段。可能会有多个动作以及多个设备。空调的action: set_temperature变成action: set, setting: temperature。现在开始：打开空调，并设置到26度，再增加 2 度。再关闭窗帘，再关闭灯光。' }],
    model: 'gpt-4o',
    stream: false,
    // @ts-ignore
    tools,
  });
  console.log('steam', stream);
  let msg = stream.choices[0].message;
  let funcInfoList = getFuncInfoFromMsg(msg);
  console.log('funcInfoList', funcInfoList);
}
```

解析后的指令是这样的：

```typescript
[
  {
    name: 'smartHome',
    arguments: {
      list: [
        { action: 'turn_on', device: 'air_conditioner' },
        {
          action: 'set',
          setting: 'temperature',
          device: 'air_conditioner',
          value: 26
        },
        {
          action: 'increase',
          setting: 'temperature',
          device: 'air_conditioner',
          value: 2
        },
        { action: 'turn_off', device: 'curtains' },
        { action: 'turn_off', device: 'lights' }
      ]
    }
  }
]

```

模型的输出每次调用不太一样，有时候指令会分散到多个 tool_calls 中，类似这样：

```typescript
[
  {
    name: 'smartHome',
    arguments: {
      list: [
        { action: 'turn_on', device: 'air_conditioner' },
        {
          action: 'set',
          setting: 'temperature',
          device: 'air_conditioner',
          value: 26
        },
        {
          action: 'increase',
          setting: 'temperature',
          device: 'air_conditioner',
          value: 2
        },
      ]
    }
  },
  {
    name: 'smartHome',
    arguments: {
      list: [
        { action: 'turn_off', device: 'curtains' },
        { action: 'turn_off', device: 'lights' },
      ]
    }
  }
];

```

不过里面的指令都是一样的，不影响我们的功能调用。

#### 控制设备执行指令
有了解析后的指令之后，我们就可以根据指令去让设备执行具体的动作了，我们可以写类似下面的代码：

```typescript
const smartHome = async (action: string, device: string, setting: string, value: any) => {
    if (device === 'air_conditioner') {
        if (action === 'turn_on') {
            // 打开空调
        } else if (action === 'turn_off') {
            // 关闭空调
        }
    }
};
```

这样就可以控制我们的设备了。

### 总结
通过上面的短短百来行代码，我们就可以准确解析出用户语言中的指令，并且在一句话中控制多个设备，让设备做出相应的响应。是不是比那些老设备的语音助手功能更强大？

本篇先更新到这里，后面的文章中我会更新语音识别、以及真实的智能家居设备控制等环节的代码。最终我们开发完之后，会有一个属于我们自己的简版的智能家居助手App。

感兴趣的朋友可以点赞关注收藏等，后面会继续更新，谢谢～



