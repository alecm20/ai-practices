GitHub 上有一个比较热门的项目，[screenshot-to-code](https://github.com/abi/screenshot-to-code)，目前已经 66.5k star，可以根据一个截图生成前端代码，生成的代码支持 React、Vue、Tailwind CSS、Bootstrap、和HTML 以及 CSS。

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1735971207325-e9920fb4-e852-46a8-96f7-76fb6430c80c.png)



我们一起来看看他的源码，看这个项目是怎么实现把一个截图变成代码的。

项目下的代码主要在 frontend 和 backend 两个目录下，前端技术栈为 React + Tailwind CSS，后端项目是 Python + fastapi web 框架。把截图变成代码的逻辑主要在 backend 下，所以我们主要来看他的后端代码逻辑。

生成代码的逻辑主要在 backend/routes/generate_code.py文件中。

这里主要是处理参数相关的逻辑，继续往下看。

```python
extracted_params = await extract_params(params, throw_error)
stack = extracted_params.stack
input_mode = extracted_params.input_mode
openai_api_key = extracted_params.openai_api_key
openai_base_url = extracted_params.openai_base_url
anthropic_api_key = extracted_params.anthropic_api_key
should_generate_images = extracted_params.should_generate_images
generation_type = extracted_params.generation_type
```

这里是选取要使用的模型，这个项目中使用的模型主要为 gpt-4o、claude-3.5。

```python
if generation_type == "create":
    claude_model = Llm.CLAUDE_3_5_SONNET_2024_10_22
else:
    claude_model = Llm.CLAUDE_3_5_SONNET_2024_06_20

if openai_api_key and anthropic_api_key:
    variant_models = [
        claude_model,
        Llm.GPT_4O_2024_11_20,
    ]
elif openai_api_key:
    variant_models = [
        Llm.GPT_4O_2024_11_20,
        Llm.GPT_4O_2024_11_20,
    ]
elif anthropic_api_key:
    variant_models = [
        claude_model,
        Llm.CLAUDE_3_5_SONNET_2024_06_20,
    ]
else:
    await throw_error(
        "No OpenAI or Anthropic API key found. Please add the environment variable OPENAI_API_KEY or ANTHROPIC_API_KEY to backend/.env or in the settings dialog. If you add it to .env, make sure to restart the backend server."
    )
    raise Exception("No OpenAI or Anthropic key")
```

下面是构建请求参数的代码，不同的模型需要的参数和请求的地址不同，使用对应的参数构建请求之后，统一放到 tasks 中。

```python
for index, model in enumerate(variant_models):
    if model == Llm.GPT_4O_2024_11_20 or model == Llm.O1_2024_12_17:
        if openai_api_key is None:
            await throw_error("OpenAI API key is missing.")
            raise Exception("OpenAI API key is missing.")
        tasks.append(
            stream_openai_response(
                prompt_messages,
                api_key=openai_api_key,
                base_url=openai_base_url,
                callback=lambda x, i=index: process_chunk(x, i),
                model=model,
            )
        )
    elif model == Llm.GEMINI_2_0_FLASH_EXP and GEMINI_API_KEY:
        tasks.append(
            stream_gemini_response(
                prompt_messages,
                api_key=GEMINI_API_KEY,
                callback=lambda x, i=index: process_chunk(x, i),
                model=Llm.GEMINI_2_0_FLASH_EXP,
            )
        )
    elif (
        model == Llm.CLAUDE_3_5_SONNET_2024_06_20
        or model == Llm.CLAUDE_3_5_SONNET_2024_10_22
    ):
        if anthropic_api_key is None:
            await throw_error("Anthropic API key is missing.")
            raise Exception("Anthropic API key is missing.")

        # For creation, use Claude Sonnet 3.6 but it can be lazy
        # so for updates, we use Claude Sonnet 3.5
        if params["generationType"] == "create":
            claude_model = Llm.CLAUDE_3_5_SONNET_2024_10_22
        else:
            claude_model = Llm.CLAUDE_3_5_SONNET_2024_06_20

        tasks.append(
            stream_claude_response(
                prompt_messages,
                api_key=anthropic_api_key,
                callback=lambda x, i=index: process_chunk(x, i),
                model=claude_model,
            )
        )
```

然后发起并发请求，获取模型 API 返回的响应结果。

```python
completions = await asyncio.gather(*tasks, return_exceptions=True)

```

这里面比较重要的是prompt_messages的内容，我们打印一下prompt_messages的数据，看下：

```json
[
  {
    "role": "system",
    "content": "\nYou are an expert React/Tailwind develo... (1895 chars)"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA... (274607 chars)",
          "detail": "high"
        }
      },
      {
        "type": "text",
        "text": "\nGenerate code for a web page that looks... (60 chars)"
      }
    ]
  }
]
```

完整的提示语在项目中可以找到，我们做个替换，下面就是完整的prompt_messages内容。

```python
[
  {
    "role": "system",
    "content": """
    You are an expert React/Tailwind developer
    - Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
    - Repeat elements as needed. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
    - For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

    In terms of libraries,

    - Use these script to include React so that it can run on a standalone page:
    <script src="https://unpkg.com/react/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.js"></script>
    - Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
    - You can use Google Fonts
    - Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

    Return only the full code in <html></html> tags.
    Do not include markdown "```" or "```html" at the start or end.
    """
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA... (274607 chars)",
          "detail": "high"
        }
      },
      {
        "type": "text",
        "text": "Generate code for a web page that looks exactly like this."
      }
    ]
  }
]
```

提示语翻译成中文：

```python

你是一名专业的 React/Tailwind 开发人员
	•	不要在代码中添加类似 <!-- Add other navigation links as needed --> 或 <!-- ... other news items ... --> 的注释来代替完整的代码，请编写完整代码。
	•	如果需要重复元素，例如有 15 个项目，代码中必须包含 15 个项目。不要留类似 <!-- Repeat for each news item --> 这样的注释，否则会导致问题。
	•	对于图片，请使用 https://placehold.co 的占位图片，并在 alt 属性中详细描述图片内容，以便图像生成 AI 能够随后生成这些图片。

关于使用的库：
	•	使用以下脚本引入 React，使其可以在独立页面上运行：

<script src="https://unpkg.com/react/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom/umd/react-dom.development.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.js"></script>


	•	使用以下脚本引入 Tailwind：

<script src="https://cdn.tailwindcss.com"></script>


	•	你可以使用 Google Fonts。
	•	使用 Font Awesome 图标库：

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>



只返回包含完整代码的 <html></html> 标签。
不要在代码前后包含 markdown 的 “" 或 "html”。

生成一个网页代码，使其外观与此完全相同。
```

可以看到，这就是一个 prompt，让大模型根据图片内容，生成对应的代码，这里的 image_url 为 base64 格式的，提供一个 HTTP 格式的 Url 也可以。

现在有了prompt_messages的内容之后，我们可以使用这个结构自己调用 openAI 的 API 来完成截图到代码的工作。

下面这个是使用 Node.js 版本的  OpenAI SDK 发起 API 调用的代码，messages 内容就是上面的prompt_messages。

```javascript
async function screenshot2code() {
  // @ts-ignore
  const stream = await openai.chat.completions.create({
    messages: msg,
    model: 'gpt-4o-2024-11-20',
    stream: false,
    // @ts-ignore
    tools,
  });
}
screenshot2code();
```

 API 返回了对应的代码，大致的内容是下面这种：

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1735973865522-d032d42f-08df-4eef-8395-8e81ff57ef5f.png)

下面我们对比下原图、自己调用的Node.js版本的 SDK 返回的代码渲染效果、还有 screenshot-to-code 项目生成代码的渲染结果。

原图：

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1735974001878-74537bf6-eeb1-4e7f-b0ac-c5b316ca6a4e.png)

Node.js 调用 API 返回的网页代码渲染的效果：

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1735974057969-7731b7a9-d010-4f82-809e-bb9fdb65bbd5.png)

screenshot-to-code 这个项目返回的网页代码渲染结果：

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1735976231328-24b1472f-b31d-4711-8e1c-b4181e1a7d71.png)

可以看到，基本的布局是差不多的，但是上面有两个图片不一样。我们自己的版本中，图片是一个类似占位符的图片，下面是一个重新生成的图片。

这个占位符图片是这样的：

```javascript
https://placehold.co/1400x200
```

screenshot-to-code 这个项目，会把这种格式的图片使用 dalle3 模型生成一张新的图片，然后做替换。图片生成加替换的代码也比较简单，这里不再贴出来了。替换之后就会变成上面的带图片的效果了。

这个项目还做了其他的一些优化，但是主要原理其实就是构建合适的 prompt，让大模型识别图片来帮我们生成代码。如果想要自己实现类似的功能，我们去调用对应的大模型 API 也能完成这个工作。

上面就是这篇文档的内容了，觉得有问题的可以在评论区留言。也欢迎大家点赞收藏关注，后续我会继续更新其他内容，谢谢～

