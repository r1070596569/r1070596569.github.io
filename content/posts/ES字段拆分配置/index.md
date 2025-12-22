---
title: ES字段拆分配置
date: 2025-12-17T16:53:17Z
lastmod: 2025-12-19T17:10:44Z
---

# ES字段拆分配置

filebeat配置

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /opt/logs/debug.log
  multiline:
    pattern: '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
    negate: true
    match: after
  fields:
    app_name: "order-service"
    environment: "testing"
    server_ip: "10.0.2.10"
  fields_under_root: true

output.logstash:
  hosts: ["logstash-server:5044"]
```

logstash配置

```ruby
input {
  beats {
    port => 5044
    host => "0.0.0.0"
  }
}

filter {
  # 1. 去除多行合并后的换行符（可选）
  # mutate {
  #   gsub => [
  #     "message", "\r", " ",
  #     "message", "\n", "\\n"
  #   ]
  # }

  # 1. 从Filebeat的fields中获取应用名和环境
  # 由于设置了fields_under_root: true，可以直接使用
  if ![environment] {
    mutate {
      add_field => { "environment" => "unknown" }
    }
  }
  
  if ![app_name] {
    mutate {
      add_field => { "app_name" => "unknown-app" }
    }
  }
  
  # 2. 如果environment是unknown，尝试通过IP判断环境
  # if [environment] == "unknown" and [server_ip] {
  #   # 定义IP范围与环境映射
  #   if [server_ip] =~ /^10\.0\.1\./ {
  #     mutate {
  #       replace => { "environment" => "production" }
  #     }
  #   } else if [server_ip] =~ /^10\.0\.2\./ {
  #     mutate {
  #       replace => { "environment" => "testing" }
  #     }
  #   } else if [server_ip] =~ /^10\.0\.3\./ {
  #     mutate {
  #       replace => { "environment" => "development" }
  #     }
  #   }
  # }
  # 
  # # 3. 也可以通过host字段判断（Filebeat会自动添加host字段）
  # if [environment] == "unknown" and [host] and [host][name] {
  #   # 根据主机名判断环境
  #   if [host][name] =~ /prod-/ {
  #     mutate {
  #       replace => { "environment" => "production" }
  #     }
  #   } else if [host][name] =~ /test-/ {
  #     mutate {
  #       replace => { "environment" => "testing" }
  #     }
  #   } else if [host][name] =~ /dev-/ {
  #     mutate {
  #       replace => { "environment" => "development" }
  #     }
  #   }
  # }
  
  # 4. 标准化环境名称
  mutate {
    lowercase => ["environment"]
    gsub => [
      "environment", "production", "prod",
      "environment", "testing", "test",
      "environment", "development", "dev"
    ]
  }

  
  # 2. 基础解析：所有日志都有的格式
  grok {
    match => { 
      "message" => [
        # 主模式，匹配基础日志格式
        "%{TIMESTAMP_ISO8601:timestamp} \[%{DATA:thread}\] %{LOGLEVEL:level}  \[%{DATA:logger}\] %{DATA:file}:%{DATA:line} \[%{DATA:trace_id}\] uid:%{NUMBER:user_id} - %{GREEDYDATA:log_message}",
        # 备用模式，没有uid的情况
        "%{TIMESTAMP_ISO8601:timestamp} \[%{DATA:thread}\] %{LOGLEVEL:level}  \[%{DATA:logger}\] %{DATA:file}:%{DATA:line} \[%{DATA:trace_id}\] %{DATA:user_info} - %{GREEDYDATA:log_message}"
      ]
    }
    tag_on_failure => ["_grokparsefailure"]
  }
  
  # 3. 处理user_id
  if [user_id] {
    mutate {
      convert => { "user_id" => "long" }
    }
  } else if [user_info] {
    # 从user_info中提取uid
    grok {
      match => { 
        "user_info" => [
          "uid:%{NUMBER:user_id}",
          "%{GREEDYDATA}"
        ]
      }
    }
    
    if [user_id] {
      mutate {
        convert => { "user_id" => "long" }
      }
    }
  }
  
  # 4. 判断是否为需要提取的日志格式
  if [log_message] =~ /^\[.*? \| (REQUEST|RESPONSE)\]/ {
    # 4.1 提取通用部分
    grok {
      match => { 
        "log_message" => "^\[%{DATA:api_path} \| %{WORD:request_type}\] \| %{GREEDYDATA:details}"
      }
    }
    
    # 4.2 根据request_type分别处理
    if [request_type] == "REQUEST" {
      grok {
        match => { 
          "details" => "Method: %{WORD:http_method} \| Params: %{DATA:request_params} \| LogTag: %{DATA:log_tag}"
        }
      }
    } else if [request_type] == "RESPONSE" {
      grok {
        match => { 
          "details" => "Time: %{NUMBER:response_time}ms \| Result: %{DATA:response_result} \| LogTag: %{DATA:log_tag}"
        }
      }
      mutate {
        convert => { "response_time" => "integer" }
      }
    }
    
    # 4.3 解析LogTag
    if [log_tag] {
      # 分离前缀和参数
      grok {
        match => { "log_tag" => "^%{DATA:log_tag_prefix}\{(?<log_params>[^}]+)\}" }
      }
      
      # 清理字段
      mutate {
        remove_field => ["details"]
        add_field => { "log_category" => "api_%{request_type}" }
      }
    }
  } else {
    # 5. 处理异常堆栈
    if [level] == "ERROR" or [log_message] =~ /(Exception|Error|at\s+[a-zA-Z0-9_.]+\(|Caused by:|java\.|org\.)/ {
      mutate {
        add_field => { "log_category" => "exception" }
      }
      
      # 从异常信息中提取异常类
      if [log_message] =~ /^(?<exception_class>[a-zA-Z0-9_.]+Exception):/ {
        grok {
          match => { "log_message" => "^(?<exception_class>[a-zA-Z0-9_.]+Exception):%{GREEDYDATA:exception_message}" }
        }
      }
    } else {
      # 其他普通日志
      mutate {
        add_field => { "log_category" => "general" }
      }
    }
  }
  
  # 6. 将异常堆栈的换行符还原（便于查看）
  if [log_category] == "exception" {
    mutate {
      gsub => [
        "log_message", "\\\\n", "\n"
      ]
    }
  }
  
  # 7. 添加时间戳
  date {
    match => ["timestamp", "ISO8601"]
    target => "@timestamp"
    timezone => "Asia/Shanghai"
  }
  
  # 8. 清理不需要的字段
  mutate {
    remove_field => ["log_message", "user_info", "@version", "host", "path"]
  }

  # 10. 为索引准备元数据
  mutate {
    add_field => {
      "[@metadata][index_app]" => "%{app_name}"
      "[@metadata][index_env]" => "%{environment}"
    }
  }

}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    # 索引格式: {环境}-{应用名}-logs-{日期}
    index => "%{[@metadata][index_env]}-%{[@metadata][index_app]}-logs-%{+YYYY.MM.dd}"
    # 如果有认证，需要配置用户和密码
    # user => "elastic"
    # password => "changeme"
  }
  
  # 调试时输出到控制台
  stdout { 
    codec => rubydebug 
  }
}
```

‍
