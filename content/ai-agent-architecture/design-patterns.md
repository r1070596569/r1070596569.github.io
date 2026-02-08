# 设计模式和架构模式应用

## 1. 核心设计模式应用

### 1.1 策略模式（Strategy Pattern）

**应用场景：** AI能力的多样化实现、智能体策略选择

```java
// AI能力策略接口
public interface AICapabilityStrategy {
    AIResponse execute(AIRequest request);
    CapabilityMetadata getMetadata();
    boolean supports(CapabilityType type);
}

// 具体策略实现
@Component
public class HomeworkExtractionStrategy implements AICapabilityStrategy {
    @Override
    public AIResponse execute(AIRequest request) {
        // 作业提取具体实现
        return extractHomework(request);
    }
    
    @Override
    public boolean supports(CapabilityType type) {
        return type == CapabilityType.HOMEWORK_EXTRACTION;
    }
}

@Component
public class ChatTutorStrategy implements AICapabilityStrategy {
    @Override
    public AIResponse execute(AIRequest request) {
        // 聊天辅导具体实现
        return tutorChat(request);
    }
    
    @Override
    public boolean supports(CapabilityType type) {
        return type == CapabilityType.CHAT_TUTOR;
    }
}

// 策略上下文
@Service
public class AICapabilityContext {
    private final Map<CapabilityType, AICapabilityStrategy> strategies;
    
    public AICapabilityContext(List<AICapabilityStrategy> strategyList) {
        this.strategies = strategyList.stream()
            .collect(Collectors.toMap(
                strategy -> strategy.getMetadata().getType(),
                Function.identity()
            ));
    }
    
    public AIResponse execute(CapabilityType type, AIRequest request) {
        AICapabilityStrategy strategy = strategies.get(type);
        if (strategy == null) {
            throw new UnsupportedOperationException("Unsupported capability type: " + type);
        }
        return strategy.execute(request);
    }
}
```

### 1.2 工厂模式（Factory Pattern）

**应用场景：** 智能体创建、AI能力实例化

```java
// 抽象工厂
public interface AgentFactory {
    BaseAgent createAgent(AgentConfig config);
    AgentType getSupportedType();
}

// 具体工厂实现
@Component
public class HomeworkAgentFactory implements AgentFactory {
    private final ApplicationContext applicationContext;
    
    @Override
    public BaseAgent createAgent(AgentConfig config) {
        return applicationContext.getBean(HomeworkAgent.class, config);
    }
    
    @Override
    public AgentType getSupportedType() {
        return AgentType.HOMEWORK;
    }
}

@Component
public class ChatAgentFactory implements AgentFactory {
    private final ApplicationContext applicationContext;
    
    @Override
    public BaseAgent createAgent(AgentConfig config) {
        return applicationContext.getBean(ChatAgent.class, config);
    }
    
    @Override
    public AgentType getSupportedType() {
        return AgentType.CHAT;
    }
}

// 工厂管理器
@Service
public class AgentFactoryManager {
    private final Map<AgentType, AgentFactory> factories;
    
    public AgentFactoryManager(List<AgentFactory> factoryList) {
        this.factories = factoryList.stream()
            .collect(Collectors.toMap(
                AgentFactory::getSupportedType,
                Function.identity()
            ));
    }
    
    public BaseAgent createAgent(AgentType type, AgentConfig config) {
        AgentFactory factory = factories.get(type);
        if (factory == null) {
            throw new IllegalArgumentException("No factory found for agent type: " + type);
        }
        return factory.createAgent(config);
    }
}
```

### 1.3 装饰器模式（Decorator Pattern）

**应用场景：** AI能力增强（日志、监控、缓存、重试）

```java
// 基础AI能力接口
public interface AICapability {
    AIResponse execute(AIRequest request);
}

// 日志装饰器
@Component
public class LoggingAICapabilityDecorator implements AICapability {
    private final AICapability delegate;
    private final Logger logger = LoggerFactory.getLogger(LoggingAICapabilityDecorator.class);
    
    public LoggingAICapabilityDecorator(AICapability delegate) {
        this.delegate = delegate;
    }
    
    @Override
    public AIResponse execute(AIRequest request) {
        logger.info("Executing AI capability with request: {}", request);
        long startTime = System.currentTimeMillis();
        
        try {
            AIResponse response = delegate.execute(request);
            long duration = System.currentTimeMillis() - startTime;
            logger.info("AI capability executed successfully in {}ms", duration);
            return response;
        } catch (Exception e) {
            logger.error("AI capability execution failed", e);
            throw e;
        }
    }
}

// 缓存装饰器
@Component
public class CachingAICapabilityDecorator implements AICapability {
    private final AICapability delegate;
    private final Cache<String, AIResponse> cache;
    
    public CachingAICapabilityDecorator(AICapability delegate, Cache<String, AIResponse> cache) {
        this.delegate = delegate;
        this.cache = cache;
    }
    
    @Override
    public AIResponse execute(AIRequest request) {
        String cacheKey = generateCacheKey(request);
        AIResponse cached = cache.getIfPresent(cacheKey);
        
        if (cached != null) {
            return cached;
        }
        
        AIResponse response = delegate.execute(request);
        if (response.isSuccess()) {
            cache.put(cacheKey, response);
        }
        return response;
    }
}

// 重试装饰器
@Component
public class RetryAICapabilityDecorator implements AICapability {
    private final AICapability delegate;
    private final int maxRetries;
    private final long retryDelayMs;
    
    public RetryAICapabilityDecorator(AICapability delegate, 
                                    @Value("${ai.retry.max-retries:3}") int maxRetries,
                                    @Value("${ai.retry.delay-ms:1000}") long retryDelayMs) {
        this.delegate = delegate;
        this.maxRetries = maxRetries;
        this.retryDelayMs = retryDelayMs;
    }
    
    @Override
    public AIResponse execute(AIRequest request) {
        Exception lastException = null;
        
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return delegate.execute(request);
            } catch (Exception e) {
                lastException = e;
                if (attempt < maxRetries) {
                    try {
                        Thread.sleep(retryDelayMs * (1 << attempt)); // 指数退避
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted during retry", ie);
                    }
                }
            }
        }
        
        throw new AIExecutionException("Failed after " + maxRetries + " retries", lastException);
    }
}

// 装饰器链构建器
@Component
public class AICapabilityDecoratorChainBuilder {
    private final List<AICapabilityDecorator> decorators;
    
    public AICapability buildChain(AICapability baseCapability) {
        AICapability current = baseCapability;
        for (AICapabilityDecorator decorator : decorators) {
            current = decorator.decorate(current);
        }
        return current;
    }
}
```

### 1.4 观察者模式（Observer Pattern）

**应用场景：** 智能体状态变化通知、事件驱动架构

```java
// 事件接口
public interface AgentEvent {
    AgentEventType getType();
    String getAgentId();
    LocalDateTime getTimestamp();
}

// 事件类型
public enum AgentEventType {
    CREATED, ACTIVATED, DEACTIVATED, ERROR, COMPLETED
}

// 观察者接口
public interface AgentEventListener {
    void onAgentEvent(AgentEvent event);
}

// 事件发布器
@Component
public class AgentEventPublisher {
    private final List<AgentEventListener> listeners = new CopyOnWriteArrayList<>();
    
    public void addListener(AgentEventListener listener) {
        listeners.add(listener);
    }
    
    public void removeListener(AgentEventListener listener) {
        listeners.remove(listener);
    }
    
    public void publishEvent(AgentEvent event) {
        listeners.forEach(listener -> {
            try {
                listener.onAgentEvent(event);
            } catch (Exception e) {
                log.error("Error handling agent event", e);
            }
        });
    }
}

// 具体事件监听器
@Component
public class AgentMetricsEventListener implements AgentEventListener {
    private final MeterRegistry meterRegistry;
    
    @Override
    public void onAgentEvent(AgentEvent event) {
        Counter.builder("agent.events")
            .tag("eventType", event.getType().name())
            .register(meterRegistry)
            .increment();
    }
}

@Component
public class AgentPersistenceEventListener implements AgentEventListener {
    private final AgentStateRepository stateRepository;
    
    @Override
    public void onAgentEvent(AgentEvent event) {
        // 持久化状态变更
        stateRepository.saveAgentEvent(event);
    }
}
```

### 1.5 命令模式（Command Pattern）

**应用场景：** 智能体操作的封装、操作历史记录、撤销重做

```java
// 命令接口
public interface AgentCommand {
    AgentResponse execute();
    void undo();
    boolean isUndoable();
    CommandMetadata getMetadata();
}

// 具体命令实现
public class SubmitHomeworkCommand implements AgentCommand {
    private final HomeworkSubmissionService submissionService;
    private final HomeworkSubmissionRequest request;
    private HomeworkSubmissionResult result;
    
    @Override
    public AgentResponse execute() {
        this.result = submissionService.submit(request);
        return convertToAgentResponse(result);
    }
    
    @Override
    public void undo() {
        if (result != null) {
            submissionService.cancelSubmission(result.getSubmissionId());
        }
    }
    
    @Override
    public boolean isUndoable() {
        return true;
    }
}

// 命令执行器
@Component
public class AgentCommandExecutor {
    private final Deque<AgentCommand> commandHistory = new ArrayDeque<>();
    private static final int MAX_HISTORY_SIZE = 10;
    
    public AgentResponse executeCommand(AgentCommand command) {
        AgentResponse response = command.execute();
        
        if (command.isUndoable()) {
            commandHistory.push(command);
            if (commandHistory.size() > MAX_HISTORY_SIZE) {
                commandHistory.removeLast();
            }
        }
        
        return response;
    }
    
    public AgentResponse undoLastCommand() {
        if (commandHistory.isEmpty()) {
            throw new IllegalStateException("No commands to undo");
        }
        
        AgentCommand lastCommand = commandHistory.pop();
        lastCommand.undo();
        return createUndoResponse(lastCommand);
    }
}
```

## 2. 架构模式应用

### 2.1 分层架构（Layered Architecture）

```
┌─────────────────────────────────────────────────────────┐
│                   表现层 (Presentation Layer)           │
│  - REST API Controllers                                │
│  - WebSocket Endpoints                                 │
│  - API Gateway Integration                             │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   应用层 (Application Layer)            │
│  - AI编排服务 (AI Orchestration Service)               │
│  - 智能体管理服务 (Agent Management Service)           │
│  - 业务逻辑协调 (Business Logic Coordination)          │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   领域层 (Domain Layer)                 │
│  - AI能力抽象 (AI Capability Abstraction)              │
│  - 智能体核心逻辑 (Agent Core Logic)                   │
│  - 领域实体和值对象 (Domain Entities & Value Objects)  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   基础设施层 (Infrastructure Layer)     │
│  - 模型网关 (Model Gateway)                            │
│  - 向量存储 (Vector Store)                             │
│  - 消息队列 (Message Queue)                            │
│  - 缓存系统 (Cache System)                             │
│  - 持久化存储 (Persistence Storage)                    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 微服务架构（Microservices Architecture）

**服务划分原则：**
- **单一职责原则**：每个服务只负责一个业务领域
- **高内聚低耦合**：服务内部高内聚，服务间低耦合
- **独立部署**：每个服务可以独立部署和扩展

**服务边界定义：**

| 服务名称 | 职责 | 接口 |
|---------|------|------|
| AI-Orchestrator | AI能力编排和路由 | `/api/v1/orchestrate` |
| Homework-Agent | 作业智能体管理 | `/api/v1/agents/homework` |
| Chat-Agent | 聊天智能体管理 | `/api/v1/agents/chat` |
| Grading-Agent | 批改智能体管理 | `/api/v1/agents/grading` |
| Model-Gateway | 统一模型访问网关 | `/api/v1/models` |
| Knowledge-Base | 知识库管理 | `/api/v1/knowledge` |

### 2.3 事件驱动架构（Event-Driven Architecture）

```java
// 事件定义
@Data
public class HomeworkSubmittedEvent {
    private String homeworkId;
    private String studentId;
    private String subject;
    private LocalDateTime timestamp;
}

@Data
public class AgentCreatedEvent {
    private String agentId;
    private AgentType type;
    private String userId;
    private LocalDateTime timestamp;
}

// 事件处理器
@Component
@RocketMQMessageListener(topic = "homework-topic", consumerGroup = "ai-consumer-group")
public class HomeworkSubmittedEventHandler implements RocketMQListener<HomeworkSubmittedEvent> {
    
    private final AgentLifecycleManager agentManager;
    
    @Override
    public void onMessage(HomeworkSubmittedEvent event) {
        // 创建作业智能体来处理新提交的作业
        AgentCreateRequest request = AgentCreateRequest.builder()
            .type(AgentType.HOMEWORK)
            .userId(event.getStudentId())
            .config(buildHomeworkAgentConfig(event))
            .build();
            
        agentManager.createAgent(request);
    }
}

// 事件发布
@Service
public class HomeworkService {
    private final RocketMQTemplate rocketMQTemplate;
    
    public void submitHomework(HomeworkSubmissionRequest request) {
        // 保存作业
        HomeworkEntity homework = saveHomework(request);
        
        // 发布事件
        HomeworkSubmittedEvent event = new HomeworkSubmittedEvent();
        event.setHomeworkId(homework.getId());
        event.setStudentId(homework.getStudentId());
        event.setSubject(homework.getSubject());
        event.setTimestamp(LocalDateTime.now());
        
        rocketMQTemplate.convertAndSend("homework-topic", event);
    }
}
```

### 2.4 CQRS模式（Command Query Responsibility Segregation）

**写模型（Commands）：**
```java
// 写操作接口
public interface AgentWriteService {
    String createAgent(AgentCreateCommand command);
    void updateAgent(String agentId, AgentUpdateCommand command);
    void deleteAgent(String agentId);
    AgentResponse interact(String agentId, AgentInteractionCommand command);
}

// 写操作实现
@Service
@Transactional
public class AgentWriteServiceImpl implements AgentWriteService {
    private final AgentRepository agentRepository;
    private final AgentEventPublisher eventPublisher;
    
    @Override
    public String createAgent(AgentCreateCommand command) {
        // 验证命令
        validateCommand(command);
        
        // 创建智能体
        BaseAgent agent = agentFactory.createAgent(command.getConfig());
        
        // 保存到数据库
        agentRepository.save(agent);
        
        // 发布事件
        eventPublisher.publishEvent(new AgentCreatedEvent(agent.getId(), agent.getType()));
        
        return agent.getId();
    }
}
```

**读模型（Queries）：**
```java
// 读操作接口
public interface AgentReadService {
    AgentDTO getAgent(String agentId);
    List<AgentDTO> getAgentsByUser(String userId);
    List<AgentDTO> getActiveAgents();
    AgentStatistics getAgentStatistics();
}

// 读操作实现（使用专门的读取优化表）
@Repository
public class AgentReadServiceImpl implements AgentReadService {
    private final JdbcTemplate jdbcTemplate;
    
    @Override
    public AgentDTO getAgent(String agentId) {
        // 从专门的读取优化视图查询
        return jdbcTemplate.queryForObject(
            "SELECT * FROM agent_read_view WHERE agent_id = ?", 
            new AgentRowMapper(), 
            agentId
        );
    }
    
    @Override
    public List<AgentDTO> getActiveAgents() {
        // 使用缓存优化频繁查询
        return cache.get("active_agents", () -> 
            jdbcTemplate.query("SELECT * FROM active_agent_view", new AgentRowMapper())
        );
    }
}
```

### 2.5 管道-过滤器模式（Pipe-Filter Pattern）

**应用场景：** AI请求处理流水线

```java
// 过滤器接口
public interface AIRequestFilter {
    AIRequest process(AIRequest request);
    int getOrder();
}

// 具体过滤器实现
@Component
public class ValidationFilter implements AIRequestFilter {
    @Override
    public AIRequest process(AIRequest request) {
        // 验证请求参数
        validateRequest(request);
        return request;
    }
    
    @Override
    public int getOrder() {
        return 1; // 最先执行
    }
}

@Component
public class ContextEnrichmentFilter implements AIRequestFilter {
    @Override
    public AIRequest process(AIRequest request) {
        // 丰富请求上下文
        enrichContext(request);
        return request;
    }
    
    @Override
    public int getOrder() {
        return 2;
    }
}

@Component
public class RateLimitingFilter implements AIRequestFilter {
    @Override
    public AIRequest process(AIRequest request) {
        // 限流检查
        checkRateLimit(request);
        return request;
    }
    
    @Override
    public int getOrder() {
        return 3;
    }
}

// 管道执行器
@Component
public class AIRequestPipeline {
    private final List<AIRequestFilter> filters;
    
    public AIRequestPipeline(List<AIRequestFilter> filterList) {
        this.filters = filterList.stream()
            .sorted(Comparator.comparingInt(AIRequestFilter::getOrder))
            .collect(Collectors.toList());
    }
    
    public AIRequest process(AIRequest request) {
        AIRequest current = request;
        for (AIRequestFilter filter : filters) {
            current = filter.process(current);
        }
        return current;
    }
}
```

## 3. 设计模式组合应用

### 3.1 智能体生命周期管理组合模式

```java
// 结合工厂模式 + 策略模式 + 观察者模式
@Service
public class AgentLifecycleManager {
    
    // 工厂模式：创建不同类型的智能体
    private final AgentFactoryManager factoryManager;
    
    // 策略模式：不同的生命周期策略
    private final Map<AgentType, AgentLifecycleStrategy> lifecycleStrategies;
    
    // 观察者模式：通知状态变化
    private final AgentEventPublisher eventPublisher;
    
    public String createAgent(AgentCreateRequest request) {
        // 1. 工厂模式创建智能体
        BaseAgent agent = factoryManager.createAgent(request.getType(), request.getConfig());
        
        // 2. 策略模式初始化
        AgentLifecycleStrategy strategy = lifecycleStrategies.get(request.getType());
        strategy.initialize(agent);
        
        // 3. 观察者模式发布事件
        eventPublisher.publishEvent(new AgentCreatedEvent(agent.getId(), agent.getType()));
        
        return agent.getId();
    }
}
```

### 3.2 AI能力执行组合模式

```java
// 结合策略模式 + 装饰器模式 + 命令模式
@Service
public class AICapabilityExecutor {
    
    // 策略模式：选择合适的AI能力
    private final AICapabilityContext capabilityContext;
    
    // 装饰器模式：增强AI能力
    private final AICapabilityDecoratorChainBuilder decoratorBuilder;
    
    // 命令模式：封装执行操作
    private final AgentCommandExecutor commandExecutor;
    
    public AIResponse executeCapability(CapabilityType type, AIRequest request) {
        // 1. 策略模式获取基础能力
        AICapability baseCapability = capabilityContext.getCapability(type);
        
        // 2. 装饰器模式构建增强链
        AICapability enhancedCapability = decoratorBuilder.buildChain(baseCapability);
        
        // 3. 命令模式封装执行
        AgentCommand command = new ExecuteCapabilityCommand(enhancedCapability, request);
        return commandExecutor.executeCommand(command);
    }
}
```

## 4. 架构决策记录（ADR）

### 4.1 关键架构决策

| 决策ID | 决策描述 | 选择方案 | 替代方案 | 理由 |
|--------|----------|----------|----------|------|
| ADR-001 | 服务架构 | 微服务架构 | 单体架构 | 支持独立扩展AI能力，便于维护和部署 |
| ADR-002 | AI能力抽象 | 策略模式 | 继承模式 | 更好的开闭原则，易于扩展新能力 |
| ADR-003 | 智能体创建 | 工厂模式 | 直接实例化 | 解耦创建逻辑，支持依赖注入 |
| ADR-004 | 性能优化 | 多级缓存 + 异步处理 | 同步处理 | 满足高并发需求，降低响应时间 |
| ADR-005 | 通信模式 | REST + WebSocket | 仅REST | 支持实时交互和推送场景 |

### 4.2 模式选择总结

**核心设计原则：**
1. **开闭原则**：通过策略模式和工厂模式，系统对扩展开放，对修改关闭
2. **单一职责**：每个类和方法只负责一个功能
3. **依赖倒置**：高层模块不依赖低层模块，都依赖抽象
4. **里氏替换**：子类型必须能够替换其基类型
5. **接口隔离**：客户端不应该依赖它不需要的接口

**模式应用效果：**
- **可扩展性**：新增AI能力只需实现策略接口
- **可维护性**：各组件职责清晰，耦合度低
- **可测试性**：依赖注入和接口抽象便于单元测试
- **性能**：异步处理和缓存机制保证高性能
- **可靠性**：装饰器模式提供重试、降级等可靠性保障