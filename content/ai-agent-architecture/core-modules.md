# 核心功能模块和接口规范

## 1. AI能力模块详细设计

### 1.1 作业结构化提取模块 (Homework Extraction Module)

**功能描述：**
- 接收家长输入的作业文本（可能包含图片OCR结果）
- 提取作业的关键信息：科目、题目类型、难度等级、截止时间、具体要求等
- 输出结构化的JSON格式数据

**核心组件：**
```java
// 作业提取请求DTO
@Data
public class HomeworkExtractionRequest {
    private String homeworkText;        // 作业原始文本
    private String subject;            // 科目（可选，用于上下文）
    private String gradeLevel;         // 年级（可选）
    private List<String> attachments;   // 附件URL列表（图片等）
}

// 作业提取响应DTO
@Data
public class HomeworkExtractionResponse {
    private String subject;            // 科目
    private String title;              // 作业标题
    private String description;        // 作业描述
    HomeworkType type;         // 作业类型（选择题、填空题、解答题等）
    DifficultyLevel difficulty; // 难度等级
    LocalDateTime deadline;    // 截止时间
    List<Question> questions;  // 具体题目列表
    Map<String, Object> metadata; // 元数据
}

// 作业提取服务接口
public interface HomeworkExtractionService {
    HomeworkExtractionResponse extract(HomeworkExtractionRequest request);
    HomeworkExtractionResponse extractWithValidation(HomeworkExtractionRequest request);
}
```

**提示词工程：**
```java
@Component
public class HomeworkExtractionPromptBuilder {
    
    public String buildPrompt(HomeworkExtractionRequest request) {
        return """
        你是一个专业的作业分析助手，请从以下作业描述中提取结构化信息。
        
        要求：
        1. 准确识别科目（数学、语文、英语、物理、化学、生物、历史、地理、政治等）
        2. 分析作业类型（选择题、填空题、判断题、解答题、作文、实验报告等）
        3. 评估难度等级（简单、中等、困难）
        4. 提取截止时间（如果有）
        5. 将作业内容分解为具体的题目
        
        作业描述：
        %s
        
        请以严格的JSON格式返回结果，包含以下字段：
        {
          "subject": "科目",
          "title": "作业标题", 
          "description": "作业描述",
          "type": "作业类型",
          "difficulty": "难度等级",
          "deadline": "截止时间(ISO8601格式)",
          "questions": [
            {
              "questionNumber": "题号",
              "content": "题目内容",
              "type": "题目类型"
            }
          ]
        }
        """.formatted(request.getHomeworkText());
    }
}
```

### 1.2 AI聊天答疑模块 (Chat AI Module)

**功能描述：**
- 提供多轮对话能力，模拟AI老师角色
- 支持知识库问答、解题指导、学习建议
- 具备情感识别和适当的情感回应
- 维护对话上下文和用户学习档案

**核心组件：**
```java
// 聊天请求DTO
@Data
public class ChatRequest {
    private String userId;             // 用户ID
    private String sessionId;          // 会话ID
    private String message;            // 用户消息
    private ChatContext context;       // 对话上下文
    private List<KnowledgeSource> knowledgeSources; // 知识源
}

// 聊天响应DTO
@Data
public class ChatResponse {
    private String messageId;          // 消息ID
    private String response;           // AI回复
    ResponseType responseType; // 响应类型（文本、图片、链接等）
    List<SuggestedAction> suggestedActions; // 建议操作
    Emotion emotion;           // 情感状态
    ChatContext updatedContext; // 更新后的上下文
}

// 聊天服务接口
public interface ChatAIService {
    ChatResponse converse(ChatRequest request);
    ChatResponse continueConversation(String sessionId, String message);
    void endConversation(String sessionId);
}
```

**对话管理策略：**
```java
@Component
public class ChatConversationManager {
    
    // 多轮对话状态机
    public enum ConversationState {
        GREETING,          // 问候阶段
        TOPIC_IDENTIFICATION, // 话题识别
        PROBLEM_SOLVING,   // 问题解决
        KNOWLEDGE_REINFORCEMENT, // 知识强化
        CLOSING            // 结束阶段
    }
    
    // 上下文管理
    public ChatContext manageContext(ChatRequest request) {
        // 从Redis获取用户历史对话
        // 更新当前对话状态
        // 维护用户学习档案
        // 返回更新后的上下文
    }
    
    // 知识库检索
    public List<KnowledgeChunk> retrieveRelevantKnowledge(String query) {
        // 使用向量数据库进行语义搜索
        // 返回相关知识片段
    }
}
```

### 1.3 作业批改模块 (Grading AI Module)

**功能描述：**
- 支持主观题和客观题的自动批改
- 提供详细的批改反馈和改进建议
- 生成学习报告和知识点掌握情况分析
- 支持多种作业格式（文本、图片、手写识别）

**核心组件：**
```java
// 批改请求DTO
@Data
public class GradingRequest {
    private String homeworkId;         // 作业ID
    private String studentId;          // 学生ID
    private List<StudentAnswer> answers; // 学生答案
    private GradingCriteria criteria;  // 批改标准
    private boolean includeFeedback;   // 是否包含详细反馈
}

// 批改响应DTO
@Data
public class GradingResponse {
    private String gradingId;          // 批改ID
    private double score;              // 得分
    GradingStatus status;      // 批改状态
    List<QuestionGrade> questionGrades; // 题目得分详情
    List<FeedbackItem> feedback; // 反馈建议
    KnowledgeAnalysis knowledgeAnalysis; // 知识点分析
}

// 批改服务接口
public interface GradingAIService {
    GradingResponse evaluate(GradingRequest request);
    GradingResponse reevaluate(String gradingId, GradingRequest request);
    KnowledgeAnalysis analyzeKnowledgeMastery(String studentId, String subject);
}
```

**批改策略实现：**
```java
@Component
public class GradingStrategyFactory {
    
    public GradingStrategy getStrategy(QuestionType type) {
        switch (type) {
            case MULTIPLE_CHOICE:
                return new MultipleChoiceGradingStrategy();
            case FILL_IN_BLANK:
                return new FillInBlankGradingStrategy();
            case ESSAY:
                return new EssayGradingStrategy();
            case MATH_PROBLEM:
                return new MathProblemGradingStrategy();
            default:
                throw new IllegalArgumentException("Unsupported question type: " + type);
        }
    }
}

// 主观题批改策略
@Component
public class EssayGradingStrategy implements GradingStrategy {
    
    @Override
    public QuestionGrade grade(StudentAnswer answer, ModelAnswer modelAnswer) {
        // 内容完整性检查
        double completenessScore = evaluateCompleteness(answer.getContent(), modelAnswer.getKeyPoints());
        
        // 逻辑性评估
        double logicScore = evaluateLogic(answer.getContent());
        
        // 语言表达评估
        double languageScore = evaluateLanguage(answer.getContent());
        
        // 综合评分
        double finalScore = calculateWeightedScore(completenessScore, logicScore, languageScore);
        
        // 生成反馈
        List<FeedbackItem> feedback = generateFeedback(answer.getContent(), modelAnswer);
        
        return new QuestionGrade(finalScore, feedback);
    }
}
```

## 2. 接口规范详细定义

### 2.1 RESTful API 规范

**API版本控制：**
- 使用URL路径版本控制：`/api/v1/ai/capabilities/...`
- 支持向后兼容，旧版本API在新版本发布后保持6个月

**请求/响应格式：**
```json
// 通用请求格式
{
  "requestId": "unique-request-id",
  "timestamp": "2026-02-06T16:57:51Z",
  "data": {
    // 具体业务数据
  },
  "metadata": {
    "userId": "user-123",
    "sessionId": "session-456",
    "source": "mobile-app"
  }
}

// 通用响应格式
{
  "requestId": "unique-request-id",
  "timestamp": "2026-02-06T16:57:52Z",
  "code": 200,
  "message": "success",
  "data": {
    // 具体业务数据
  },
  "metadata": {
    "processingTime": 125,
    "modelUsed": "qwen-max",
    "cacheHit": false
  }
}
```

**错误处理规范：**
```json
// 错误响应格式
{
  "requestId": "unique-request-id",
  "timestamp": "2026-02-06T16:57:52Z",
  "code": 400,
  "message": "Invalid homework text format",
  "errorDetails": {
    "errorCode": "INVALID_INPUT",
    "field": "homeworkText",
    "reason": "Text length exceeds maximum limit of 10000 characters"
  }
}
```

### 2.2 具体API接口定义

#### 2.2.1 作业结构化提取接口
```yaml
POST /api/v1/ai/capabilities/homework/extract
Content-Type: application/json

Request Body:
{
  "homeworkText": "数学作业：完成课本第25页的练习题1-5",
  "subject": "数学",
  "gradeLevel": "初中二年级"
}

Response Body:
{
  "requestId": "req-123",
  "timestamp": "2026-02-06T16:57:51Z",
  "code": 200,
  "message": "success",
  "data": {
    "subject": "数学",
    "title": "课本练习题",
    "description": "完成课本第25页的练习题1-5",
    "type": "EXERCISE",
    "difficulty": "MEDIUM",
    "deadline": null,
    "questions": [
      {
        "questionNumber": "1",
        "content": "练习题1",
        "type": "EXERCISE"
      },
      {
        "questionNumber": "2", 
        "content": "练习题2",
        "type": "EXERCISE"
      }
    ]
  }
}
```

#### 2.2.2 AI聊天答疑接口
```yaml
POST /api/v1/ai/capabilities/chat/converse
Content-Type: application/json

Request Body:
{
  "userId": "user-123",
  "sessionId": "session-456",
  "message": "这道数学题怎么做？",
  "context": {
    "currentHomeworkId": "homework-789"
  }
}

Response Body:
{
  "requestId": "req-456",
  "timestamp": "2026-02-06T16:57:52Z", 
  "code": 200,
  "message": "success",
  "data": {
    "messageId": "msg-789",
    "response": "让我帮你分析这道题...",
    "responseType": "TEXT",
    "suggestedActions": [
      {
        "type": "SHOW_HINT",
        "label": "显示提示"
      },
      {
        "type": "SHOW_SOLUTION", 
        "label": "显示解答"
      }
    ],
    "emotion": "HELPFUL"
  }
}
```

#### 2.2.3 作业批改接口
```yaml
POST /api/v1/ai/capabilities/grading/evaluate
Content-Type: application/json

Request Body:
{
  "homeworkId": "homework-789",
  "studentId": "user-123", 
  "answers": [
    {
      "questionId": "q1",
      "content": "我的答案是..."
    }
  ],
  "includeFeedback": true
}

Response Body:
{
  "requestId": "req-789",
  "timestamp": "2026-02-06T16:57:53Z",
  "code": 200,
  "message": "success",
  "data": {
    "gradingId": "grade-123",
    "score": 85.5,
    "status": "COMPLETED",
    "questionGrades": [
      {
        "questionId": "q1",
        "score": 85.5,
        "maxScore": 100
      }
    ],
    "feedback": [
      {
        "type": "IMPROVEMENT",
        "content": "建议在解题过程中更详细地写出步骤"
      }
    ],
    "knowledgeAnalysis": {
      "weakAreas": ["代数运算"],
      "strongAreas": ["几何证明"]
    }
  }
}
```

### 2.3 WebSocket 实时交互接口

**智能体实时交互协议：**
```javascript
// 连接建立
const ws = new WebSocket('wss://api.luckyboot.com/ai/agents/{agentId}/ws');

// 客户端发送消息
ws.send(JSON.stringify({
  type: 'USER_MESSAGE',
  payload: {
    messageId: 'msg-123',
    content: '这道数学题怎么做？',
    timestamp: '2026-02-06T16:57:51Z'
  }
}));

// 服务端响应消息
{
  type: 'AGENT_RESPONSE',
  payload: {
    messageId: 'msg-456',
    content: '让我帮你分析这道题...',
    timestamp: '2026-02-06T16:57:52Z',
    responseType: 'TEXT',
    suggestedActions: [
      { type: 'SHOW_HINT', label: '显示提示' },
      { type: 'SHOW_SOLUTION', label: '显示解答' }
    ]
  }
}

// 智能体状态更新
{
  type: 'AGENT_STATE_UPDATE',
  payload: {
    state: 'THINKING',
    progress: 0.5,
    message: '正在分析题目...'
  }
}