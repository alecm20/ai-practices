# 使用大模型一句话生成故事短视频

## 功能介绍
借助 AI 大模型，只需要输入一句话，就可以生成一个故事短视频，生成的视频截图如下图。具体的视频效果可以去 github 的[项目主页](https://github.com/alecm20/story-flicks)体验。

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1737337277810-99de06d7-5415-482c-b32f-04b0a44e85a3.png)

这个是使用界面，我们选择需要使用的文本模型和图片生成模型，然后填写一句话的主题后稍等一两分钟就可以生成带字幕、语音的视频了。

![](https://cdn.nlark.com/yuque/0/2025/png/288653/1737338401542-419b2555-0f60-4e34-acec-f68417d7ae9a.png)

## 功能实现说明
下面我们一起来看下这个功能是如何实现的。

### 视频文本生成
首先，我们需要用户输入一个故事的主题，比如说：生成一个关于小白兔钓鱼的故事。

然后我们让大模型帮我们基于这个主题生成一个故事，故事需要分段，比如说分为 3 段，然后让模型给我们返回一个 JSON 数组。

大致的提示词是这样的：

```plain
if story_prompt:
            base_prompt = f"讲一个故事，主题是：{story_prompt}"
        
        return f"""
        {base_prompt}. The story needs to be divided into {segments} scenes, and each scene must include descriptive text and an image prompt.

        Please return the result in the following JSON format, where the key `list` contains an array of objects:

        **Expected JSON format**:
        {{
            "list": [
                {{
                    "text": "Descriptive text for the scene",
                    "image_prompt": "Detailed image generation prompt, described in English"
                }},
                {{
                    "text": "Another scene description text",
                    "image_prompt": "Another detailed image generation prompt in English"
                }}
            ]
        }}

        **Requirements**:
        1. The root object must contain a key named `list`, and its value must be an array of scene objects.
        2. Each object in the `list` array must include:
            - `text`: A descriptive text for the scene, written in {languageValue}.
            - `image_prompt`: A detailed prompt for generating an image, written in English.
        3. Ensure the JSON format matches the above example exactly. Avoid extra fields or incorrect key names like `cimage_prompt` or `inage_prompt`.
        """

```

完整的提示词可以去看项目的源码，整体就是让大模型输出一个固定格式的 JSON。

不过在实际测试效果的时候，发现不同的大模型效果不一致，阿里云的 qwen-plus 在输出 JSON 的时候会把 image_prompt 这个 key 输出为错误的 key，每次输出的 key 都不太一样，调整提示词也没有效果。后面加了个工具类，对非 text 字段的 key 做了重命名，才满足需求。

```python
def normalize_keys(self, data):
    if isinstance(data, dict):
        # 如果是字典，处理键值
        if "text" in data:
            # 找到非 `text` 的键
            other_keys = [key for key in data.keys() if key != "text"]
            # 确保只处理一个非 `text` 键的情况
            if len(other_keys) == 1:
                data["image_prompt"] = data.pop(other_keys[0])
            elif len(other_keys) > 1:
                raise ValueError(f"Unexpected extra keys: {other_keys}. Only one non-'text' key is allowed.")
        return data
    elif isinstance(data, list):
        # 如果是列表，递归处理每个对象
        return [self.normalize_keys(item) for item in data]
    else:
        raise TypeError("Input must be a dict or list of dicts")
```

在这个步骤，我们可以获取一个故事的分段数组，数组中的 item 有字幕文本和后续生成图片用的英文版的提示词。获得的数据如下：

```python
[
    {
      "text": "在茂密的森林深处，小白兔正在阳光明媚的草地上尽情地蹦跳。四周鲜花盛开，蝶儿翩翩飞舞。小白兔的眼中充满了无限的好奇与喜悦，它的白毛在阳光下显得格外光洁闪亮。",
      "image_prompt": "A vibrant forest clearing with blooming flowers and fluttering butterflies, where a small white rabbit is joyfully hopping under the bright sunlight, its fur gleaming and its eyes wide with curiosity and joy.",
    },
    {
      "text": "就在这时，大灰狼从阴影中悄然出现。它藏在一棵大树后，目光中带着些许狡黠。小白兔警觉地竖起了耳朵，但它并未发现大灰狼的存在。森林中弥漫着一丝紧张的气氛。",
      "image_prompt": "A shadowy part of the forest where a cunning grey wolf lurks behind a large tree, watching a small white rabbit intently. The wolf's eyes sparkle with slyness, while the rabbit stands alert with its ears perked up, unaware of the danger. A slight tension hangs in the air.",
    }
]
```

### 图片生成
根据上一步的 image_prompt，我们去调用图像生成模型，让模型输出相应的图片。OpenAI 的模型为 dall-e-3，阿里云的百炼上面有很多模型，包括通义的模型，还有三方的模型。目前开源模型中效果比较好的为 flux 模型，百炼平台目前有一定的免费额度可以用，具体的可以看[阿里云百炼文档](https://help.aliyun.com/zh/model-studio/getting-started/models#a1a9f05a675m4)。

OpenAI 和阿里云都有对应的 python SDK 可以使用，使用 SDK 可以比较方便的生成一张图片。

```python
if settings.image_provider == "aliyun":
    rsp = ImageSynthesis.call(model=settings.image_llm_model,
                  prompt=prompt,
                  size='1024*1024')
    if rsp.status_code == HTTPStatus.OK:
        # print("aliyun image response", rsp.output)
        for result in rsp.output.results:
            return result.url
    else:
        error_message = f'Failed, status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}'
        logger.error(error_message)
        raise Exception(error_message)
elif settings.image_provider == "openai":
    response = self.image_client.images.generate(
        model=self.image_llm_model,
        prompt=safe_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    logger.info("image generate res", response.data[0].url)
    return response.data[0].url
```

获取图片之后，我们先把图片下载到本地供后续使用。

### 语音生成
现在有了视频文本和图片，还需要生成视频的语音文件。

我们这里使用 edge-tts 这个 python 包，可以输入一个文本，选择语音，生成一段 mp3 格式的语音。

```python
communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
    sub_maker = edge_tts.SubMaker()
    
    with open(voice_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
                    
```

生成语音的时候，voice_name 需要和文本匹配，比如说中文文本，需要选择中文的语音，类似这种：zh-CN-XiaoyiNeural，否则生成语音会有问题。

这里生成语音的环节还是比较简单的。

### 字幕生成
字幕这里会稍微麻烦一些，比如下面的一段话：

```python
在茂密的森林深处，小白兔正在阳光明媚的草地上尽情地蹦跳。四周鲜花盛开，蝶儿翩翩飞舞。小白兔的眼中充满了无限的好奇与喜悦，它的白毛在阳光下显得格外光洁闪亮。
```

在视频中展示的时候，没办法在一个字幕上一次性展示完，需要分成多句话展示。

比如说我们拆分成下面的几句话：

```python
在茂密的森林深处
小白兔正在阳光明媚的草地上尽情地蹦跳
四周鲜花盛开
蝶儿翩翩飞舞
小白兔的眼中充满了无限的好奇与喜悦
它的白毛在阳光下显得格外光洁闪亮
```

我们需要知道，这些拆分后的文本，在语音文件中对应的开始时间和结束时间。这样声音在读到下一句话的时候，我们才能及时切换字幕，保持声音和字幕的同步。

```python
communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
    sub_maker = edge_tts.SubMaker()
    with open(voice_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                sub_maker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])
```

这里的逻辑如上面的代码所示，communicate.stream() 的 chunk 中有一个 WordBoundary 的类型，包含了每个词在语音文件中的开始时间和时长、以及每个词的文本。

```python
在茂密的森林深处
这句话可能会分为：
在 茂密 的 森林 深处
```

我们遍历这个communicate.stream()，获取每一个词的信息，和我们上面拆分的每句话做匹配，匹配开始和结束的词，记录对应的开始和结束时间，然后生成字幕文件，字幕文件如下：

```python
1
00:00:00,100 --> 00:00:01,700
在茂密的森林深处

2
00:00:02,013 --> 00:00:05,787
小白兔正在阳光明媚的草地上尽情地蹦跳

3
00:00:06,362 --> 00:00:07,725
四周鲜花盛开

4
00:00:07,975 --> 00:00:09,213
蝶儿翩翩飞舞

5
00:00:09,775 --> 00:00:13,037
小白兔的眼中充满了无限的好奇与喜悦

6
00:00:13,412 --> 00:00:16,688
它的白毛在阳光下显得格外光洁闪亮


```

这里面有每句话的开始时间和结束时间。

### 视频生成
上面有了图片、字幕文本、音频文件，接下来要把这些内容制作成视频。

我们用 moviepy 这个 python 的库，这里的代码比较多，下面贴一下精简的代码：

```python
# 创建图片剪辑
image_clip = ImageClip(image_file)
# 设置时长
image_clip = image_clip.with_duration(subtitle_duration)
# 创建音频剪辑
audio_clip = AudioFileClip(audio_file)
image_clip = image_clip.with_audio(audio_clip)
# 使用系统字体
font_path = os.path.join(utils.resource_dir(), "fonts", "STHeitiLight.ttc")
# 处理字幕
text_clips = []
for item in sub.subtitles:
    clip = create_text_clip(subtitle_item=item)
    text_clips.append(clip)
video_clip = CompositeVideoClip([image_clip, *text_clips])

# 合并所有视频片段
final_clip = concatenate_videoclips(clips)
video_file = os.path.join(task_dir, "video.mp4")
final_clip.write_videofile(video_file, fps=24, codec='libx264', audio_codec='aac')
```

处理视频的流程：根据图片创建 ImageClip，再添加音频数据、字幕数据，变成 CompositeVideoClip，最后再使用 concatenate_videoclips 整合多个视频片段为一个完整的视频，得到最终的视频。

## 总结
上面就是这个项目实现的大致思路，先使用大模型生成文本和图片，然后使用 edge_tts 生成语音，再生成字幕文件，然后使用 moviepy 整合素材生成完整的视频。目前视频中的图像素材只有图片，后续再接入文本或图片生成视频的 API，视频就可以有动态画面了。

这里为 github 的 [项目地址](https://github.com/alecm20/story-flicks)，感兴趣的可以去 clone 下来试用一下，觉得有帮助的可以在 github 帮我点个 star，我会继续迭代功能，谢谢～

