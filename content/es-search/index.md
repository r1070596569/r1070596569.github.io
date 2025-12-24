---
title: ES查询
date: 2025-12-18T20:57:46Z
lastmod: 2025-12-19T14:48:40Z
---

# ES查询

```ruby
# 查询所有索引
GET _cat/indices

# 创建索引
PUT company

# 查询索引
GET company

# 索引新增数据
PUT company/_doc/1
{
  "name": "小公司",
  "city": "小地方",
  "contact": "010-123456"
}

PUT company/_doc/2
{
  "name": "大公司",
  "city": "大地方",
  "contact": "010-888888"
}

# 自动生成ID：07ptMZsBX2pqnbq8Fiw8
POST company/_doc
{
  "name": "一般公司",
  "city": "一般地方",
  "contact": "010-666666"
}

# 查询数据
GET company/_doc/1

GET company/_doc/2

# 查询所有数据
GET company/_search

# 修改数据
PUT company/_doc/1
{
  "name": "小公司",
  "city": "小地方",
  "contact": "010-1234567"
}

# 修改数据
POST company/_doc/1
{
  "name": "小公司",
  "city": "小地方",
  "contact": "010-12345678"
}

# 删除数据
DELETE company/_doc/07ptMZsBX2pqnbq8Fiw8

# 批量添加
PUT company/_bulk
{"index":{"_index":"company", "_id":"03"}}
{"name": "C公司","city": "C地方","contact": "010-111111", "boss":"Lao Zhang", "age": 26}
{"index":{"_index":"company", "_id":"04"}}
{"name": "D公司","city": "D地方","contact": "010-222222", "boss":"Lao Wang", "age": 30}


# 条件查询
GET company/_search?explain=true
{
  "query": {
    "match": {
      "boss": "Lao Yang"
    }
  }
}

# 条件查询 - 无结果
GET company/_search
{
  "query": {
    "term": {
      "city": {
        "value": "大地方"
      }
    }
  }
}
# 条件查询 - 有结果
GET company/_search
{
  "query": {
    "term": {
      "city.keyword": {
        "value": "大地方"
      }
    }
  }
}

# 条件查询 or
GET company/_search
{
  "query": {
    "bool": {
      "should": [
        {"match": {
          "name": "A"
        }},
        {"match": {
          "contact": "010"
        }}
      ]
    }
  },
  "from": 0,
  "size": 10, 
  "sort": [
    {
      "age": {
        "order": "asc"
      }
    }
  ]
}

# 分组查询 分组后加总
GET company/_search
{
  "aggs": {
    "ageGroup": {
      "terms": {
        "field": "age"
      },
      "aggs": {
        "ageSum": {
          "sum": {
            "field": "age"
          }
        }
      }
    }
  },
  "size": 0
}

# 平均年龄
GET company/_search
{
  "aggs": {
    "ageAvg": {
      "avg": {
        "field": "age"
      }
    }
  },
  "size": 0
}

# 前2个
GET company/_search
{
  "aggs": {
    "top2": {
      "top_hits": {
        "sort": [
          {
            "age": {
              "order": "asc"
            }
          }
        ], 
        "size": 2
      }
    }
  },
  "size": 0
}

# 分词器
## 1. 最少切分 ik_smart
## 2. 最细粒度切分 ik_max_word

# 英文分词器
GET _analyze
{
  "analyzer": "standard",
  "text": ["i am a good boy"]
}

# 分词器
GET _analyze
{
  "analyzer": "ik_smart",
  "text": ["我来自偶然，像一颗尘土"]
}

# 分词器
GET _analyze
{
  "analyzer": "ik_max_word",
  "text": ["我来自偶然，像一颗尘土"]
}

# 搜索结果评分机制
#TF-IDF公式
#  boost * idf * tf
#  boost 权重默认2.2
#  可以手动设置查询权重值
#TF:词频
#IDF: 逆文档频率
GET company/_search?explain=true
{
  "query": {
    "match": {
      "contact": "010"
    }
  }
}

# 增加查询权重
GET company/_search
{
  "query": {
    "match": {
      "contact": {
        "query": "010",
        "boost": 1
      }
    }
  }
}

# SQL操作
POST _sql?format=txt
{
  "query": """
  SELECT * FROM "company" where age >= 26
  """
}

# SQL转成DSL
POST _sql/translate
{
  "query": """
  SELECT * FROM "company" where age >= 26
  """
}

# SQL操作
POST _sql?format=txt
{
  "query": """
  describe "company"
  """
}

# SQL操作 - json会返回游标
POST _sql?format=json
{
  "query": """
  SELECT * FROM "company" where age >= 26
  """,
  "fetch_size": 1
}

# SQL操作 - 游标定位
POST _sql?format=json
{
  "cursor": "gLuKBERGTACEUEtOwzAU9EsdIXXDng1XSCkgsUDIpiRBRZEI+TTZVM6XfJqIxFFhx1E4BPcrdlCAHbPxzJunseZBiOAZKYAkDgLHkoDasSZPb94Fn7E8lfbJ4XsJAcA8K9I62fZtx/E2aePJku4nzJCiCDKGYsFGYDGR+aDKQKTgqO178cQFf0PKUdw2nMVcDBq2EzZ8QP4wrCm5M4OzKx4b+pDa4frRt2p3RS8tXX/xNF4mVX1hr8Iu8D1KaEjT0nuKKs8IvXueuMHSKV0t0SwWbWhDJtwOXK+sDXM9wzHD3t3Zpl0s9o5WL5m2OPeNmJD//iLkGs2nk6mQjVfCddvkomc2lgPM01c+SlnyV05l/yzI0j/yFKEvAAAA//8DAA=="
}

# SQL与DSL混用
# ES会先用SQL查询，再对查询结果用DSL进行二次查询
POST _sql?format=json
{
  "query": """
  SELECT * FROM "company"
  """,
  "filter": {
    "range": {
      "age": {
        "gte": 26
      }
    }
  }
}

## Elasticsearch SQL 的LIKE：
##    1. like 'JA*A' 
##    2. like 'JAVA%'
##    3. like 'JAVA_'
##    4. rlike 'JA*A' 正则匹配
## 尽管LIKE在 Elasticsearch SQL 中搜索或过滤时是一个有效的选项,但全文搜索 MATCH和 QUERY速度更快、功能更强大，并且是首选替代方案。




```
