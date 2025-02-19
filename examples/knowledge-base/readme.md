## 项目启动方式

### 后端

```
conda create -n knowledge-base python=3.10
conda activate knowledge-base

cd examples/knowledge-base/backend
pip install -r requirements.txt

cp .env.example .env
### 设置 .env 中的api_key
openai_api_key=xxx
python main.py
```


### 前端

```
### 备注：目前暂时没有实现前端，需要通过接口进行创建知识库和查询资料
```

## 接口文档

启动项目之后，打开下面的网址即可：

http://localhost:8000/docs
