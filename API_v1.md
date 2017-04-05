# Demo V1 API

## 基本信息接口

**输入**：公众号ID

**返回值**：公众号基本信息，json格式。 返回字段说明

| 字段        | 类型   | 字段说明                                     |
|:-------------|:-------|:---------------------------------------------|
| id          | string | 公众号ID                                     |
| name        | string | 公众号名                                     |
| description | string | 公众号描述                                   |
| image       | string | 公众号头像所在的url地址                      |
| articles    | list   | 公众号文章列表                               |
| title       | string | 文章题目                                     |
| content     | list   | 公众号文章内容，多段，列表中每一个元素为一段 |

## 公众号名称搜索接口

**输入**：公众号名称

**返回值**：名称为该输入的所有公众号ID，json格式。 返回字段说明

| 字段        | 类型   | 字段说明                |
|:------------|:-------|:------------------------|
| name        | string | 公众号名                |
| accounts    | list   | 公众号列表              |
| id          | string | 公众号ID                |
| description | string | 公众号描述              |
| image       | string | 公众号头像所在的url地址 |

## 关键词提取接口

**输入**：公众号ID

**返回值**：公众号对应关键词，json格式。 返回字段说明

| 字段     | 类型   | 字段说明   |
|:---------|:-------|:-----------|
| id       | string | 公众号ID   |
| keywords | list   | 关键词列表 |
| word     | string | 关键词     |
| weight   | string | 关键词权重 |

## 产品类别推荐接口

**输入**：公众号ID，无效关键词。公众号ID为字符串格式，无效关键词为字符串格式，每个关键词之间用`||||`分隔

**返回值**：推荐的产品类别名称，json格式。 返回字段说明

| 字段       | 类型   | 字段说明     |
|:-----------|:-------|:-------------|
| categories | list   | 产品目录列表 |
| category1  | string | 一级类别     |
| category2  | string | 二级类别     |
| category3  | string | 三级类别     |
| urls       | list   | 产品url     |

## 产品详细信息接口

**输入**：产品URL

**返回值**：产品详情，json格式。 返回字段说明：

| 字段  | 类型   | 字段说明      |
|:------|:-------|---------------|
| url   | string | 产品URL       |
| title | string | 产品名        |
| image | string | 产品封面的URL |