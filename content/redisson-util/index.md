---
title: redisson配置
date: 2025-09-03T23:30:14Z
lastmod: 2025-09-04T00:01:39Z
---

# redisson配置

```xml
<!-- actuator与shiro冲突  -->
<dependency>
        <groupId>org.redisson</groupId>
        <artifactId>redisson-spring-boot-starter</artifactId>
        <exclusions>
             <exclusion>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-actuator</artifactId>
             </exclusion>
        </exclusions>
</dependency>
```

```typescript
package com.luckyboot.config;

import org.redisson.Redisson;
import org.redisson.api.RedissonClient;
import org.redisson.codec.JsonJacksonCodec;
import org.redisson.config.Config;
import org.redisson.config.TransportMode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;

@Configuration
public class RedissonConfig {

    @Value("${spring.data.redis.host:localhost}")
    private String host;

    @Value("${spring.data.redis.port:6379}")
    private String port;

    @Value("${spring.data.redis.password:}")
    private String password;

    @Value("${spring.data.redis.database:0}")
    private int database;

    @Value("${spring.data.redis.timeout:3000}")
    private int timeout;

    /**
     * 单机模式配置
     */
    @Bean(destroyMethod = "shutdown")
    public RedissonClient redissonClient() {
        Config config = new Config();
        config.setTransportMode(TransportMode.NIO);

        // 单节点配置
        config.useSingleServer()
                .setAddress("redis://" + host + ":" + port)
                .setPassword(password.isEmpty() ? null : password)
                .setDatabase(database)
                .setTimeout(timeout)
                // 连接池配置
                .setConnectionPoolSize(64)
                .setConnectionMinimumIdleSize(24)
                // 重试配置
                .setRetryAttempts(3)
                .setRetryInterval(1500)
                // 心跳检测
                .setPingConnectionInterval(30000);

        // 使用JSON序列化
        config.setCodec(new JsonJacksonCodec());

        return Redisson.create(config);
    }
}
```

```typescript
package com.luckyboot.util;

import org.redisson.api.*;
import org.redisson.client.codec.Codec;
import org.redisson.client.codec.StringCodec;
import org.redisson.codec.JsonJacksonCodec;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * 功能丰富的Redisson工具类
 */
@Component
public class RedissonUtil {

    @Autowired
    private RedissonClient redissonClient;

    // ============================== 分布式锁 ==============================

    /**
     * 可重入锁（普通锁）
     */
    public RLock getLock(String lockKey) {
        return redissonClient.getLock(lockKey);
    }

    /**
     * 公平锁
     */
    public RLock getFairLock(String lockKey) {
        return redissonClient.getFairLock(lockKey);
    }

    /**
     * 读写锁
     */
    public RReadWriteLock getReadWriteLock(String lockKey) {
        return redissonClient.getReadWriteLock(lockKey);
    }

    /**
     * 联锁（多个锁同时加锁）
     */
    public RLock getMultiLock(RLock... locks) {
        return redissonClient.getMultiLock(locks);
    }

    /**
     * 红锁（RedLock）
     */
    public RLock getRedLock(RLock... locks) {
        return redissonClient.getRedLock(locks);
    }

    /**
     * 尝试获取锁（可设置等待时间和leaseTime）
     */
    public boolean tryLock(String lockKey, long waitTime, long leaseTime, TimeUnit unit) throws InterruptedException {
        RLock lock = getLock(lockKey);
        return lock.tryLock(waitTime, leaseTime, unit);
    }

    /**
     * 加锁并执行，自动释放锁
     */
    public <T> T executeWithLock(String lockKey, long waitTime, long leaseTime, TimeUnit unit, Supplier<T> supplier) {
        RLock lock = getLock(lockKey);
        try {
            if (lock.tryLock(waitTime, leaseTime, unit)) {
                try {
                    return supplier.get();
                } finally {
                    if (lock.isHeldByCurrentThread()) {
                        lock.unlock();
                    }
                }
            }
            throw new RuntimeException("获取锁失败");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("获取锁被中断", e);
        }
    }

    /**
     * 加锁并执行，无返回值
     */
    public void executeWithLock(String lockKey, long waitTime, long leaseTime, TimeUnit unit, Runnable runnable) {
        executeWithLock(lockKey, waitTime, leaseTime, unit, () -> {
            runnable.run();
            return null;
        });
    }

    // ============================== 基本对象操作 ==============================

    /**
     * 获取Bucket对象
     */
    public <T> RBucket<T> getBucket(String key) {
        return redissonClient.getBucket(key);
    }

    /**
     * 获取Bucket对象（指定编解码器）
     */
    public <T> RBucket<T> getBucket(String key, Codec codec) {
        return redissonClient.getBucket(key, codec);
    }

    /**
     * 设置对象值
     */
    public <T> void setValue(String key, T value) {
        getBucket(key).set(value);
    }

    /**
     * 设置对象值（带过期时间）
     */
    public <T> void setValue(String key, T value, long timeToLive, TimeUnit timeUnit) {
        getBucket(key).set(value, timeToLive, timeUnit);
    }

    /**
     * 获取对象值
     */
    public <T> T getValue(String key) {
        return (T) getBucket(key).get();
    }

    /**
     * 如果不存在则设置
     */
    public <T> boolean setIfAbsent(String key, T value) {
        return getBucket(key).trySet(value);
    }

    /**
     * 如果不存在则设置（带过期时间）
     */
    public <T> boolean setIfAbsent(String key, T value, long timeToLive, TimeUnit timeUnit) {
        return getBucket(key).trySet(value, timeToLive, timeUnit);
    }

    /**
     * 删除对象
     */
    public boolean delete(String key) {
        return getBucket(key).delete();
    }

    /**
     * 判断key是否存在
     */
    public boolean hasKey(String key) {
        return redissonClient.getBucket(key).isExists();
    }

    // ============================== 原子操作 ==============================

    /**
     * 获取原子Long
     */
    public RAtomicLong getAtomicLong(String key) {
        return redissonClient.getAtomicLong(key);
    }

    /**
     * 获取原子Double
     */
    public RAtomicDouble getAtomicDouble(String key) {
        return redissonClient.getAtomicDouble(key);
    }

    /**
     * 自增并返回新值
     */
    public long incrementAndGet(String key) {
        return getAtomicLong(key).incrementAndGet();
    }

    /**
     * 自减并返回新值
     */
    public long decrementAndGet(String key) {
        return getAtomicLong(key).decrementAndGet();
    }

    /**
     * 增加指定值并返回新值
     */
    public long addAndGet(String key, long delta) {
        return getAtomicLong(key).addAndGet(delta);
    }

    // ============================== 集合操作 ==============================

    /**
     * 获取Map对象
     */
    public <K, V> RMap<K, V> getMap(String key) {
        return redissonClient.getMap(key);
    }

    /**
     * 获取本地缓存Map
     */
    public <K, V> RLocalCachedMap<K, V> getLocalCachedMap(String key, LocalCachedMapOptions<K, V> options) {
        return redissonClient.getLocalCachedMap(key, options);
    }

    /**
     * 获取Set对象
     */
    public <T> RSet<T> getSet(String key) {
        return redissonClient.getSet(key);
    }

    /**
     * 获取SortedSet对象
     */
    public <T> RSortedSet<T> getSortedSet(String key) {
        return redissonClient.getSortedSet(key);
    }

    /**
     * 获取List对象
     */
    public <T> RList<T> getList(String key) {
        return redissonClient.getList(key);
    }

    /**
     * 获取Queue对象
     */
    public <T> RQueue<T> getQueue(String key) {
        return redissonClient.getQueue(key);
    }

    /**
     * 获取Deque对象
     */
    public <T> RDeque<T> getDeque(String key) {
        return redissonClient.getDeque(key);
    }

    /**
     * 获取BlockingQueue对象
     */
    public <T> RBlockingQueue<T> getBlockingQueue(String key) {
        return redissonClient.getBlockingQueue(key);
    }

    // ============================== 发布订阅 ==============================

    /**
     * 发布消息
     */
    public <T> long publish(String topicName, T message) {
        RTopic topic = redissonClient.getTopic(topicName);
        return topic.publish(message);
    }

    /**
     * 订阅消息
     */
    public <T> int subscribe(String topicName, Class<T> messageType, Consumer<T> consumer) {
        RTopic topic = redissonClient.getTopic(topicName);
        return topic.addListener(messageType, (channel, msg) -> consumer.accept(msg));
    }

    // ============================== 布隆过滤器 ==============================

    /**
     * 获取布隆过滤器
     */
    public <T> RBloomFilter<T> getBloomFilter(String key) {
        return redissonClient.getBloomFilter(key);
    }

    /**
     * 初始化布隆过滤器
     */
    public <T> void initBloomFilter(String key, long expectedInsertions, double falseProbability) {
        RBloomFilter<T> bloomFilter = getBloomFilter(key);
        bloomFilter.tryInit(expectedInsertions, falseProbability);
    }

    /**
     * 添加元素到布隆过滤器
     */
    public <T> boolean addToBloomFilter(String key, T value) {
        return getBloomFilter(key).add(value);
    }

    /**
     * 检查元素是否可能在布隆过滤器中
     */
    public <T> boolean mightContainInBloomFilter(String key, T value) {
        return getBloomFilter(key).contains(value);
    }

    // ============================== 限流器 ==============================

    /**
     * 获取限流器
     */
    public RRateLimiter getRateLimiter(String key) {
        return redissonClient.getRateLimiter(key);
    }

    /**
     * 初始化限流器
     */
    public void initRateLimiter(String key, RateType type, long rate, long rateInterval, RateIntervalUnit intervalUnit) {
        RRateLimiter limiter = getRateLimiter(key);
        limiter.trySetRate(type, rate, rateInterval, intervalUnit);
    }

    /**
     * 尝试获取许可
     */
    public boolean tryAcquire(String key) {
        return getRateLimiter(key).tryAcquire();
    }

    /**
     * 尝试获取许可（带超时）
     */
    public boolean tryAcquire(String key, long timeout, TimeUnit unit) {
        return getRateLimiter(key).tryAcquire(timeout, unit);
    }

    // ============================== 其他高级功能 ==============================

    /**
     * 获取地理空间对象
     */
    public <T> RGeo<T> getGeo(String key) {
        return redissonClient.getGeo(key);
    }

    /**
     * 获取位图
     */
    public RBitSet getBitSet(String key) {
        return redissonClient.getBitSet(key);
    }

    /**
     * 获取HyperLogLog
     */
    public <T> RHyperLogLog<T> getHyperLogLog(String key) {
        return redissonClient.getHyperLogLog(key);
    }

    /**
     * 获取FIFO队列
     */
    public <T> RScoredSortedSet<T> getScoredSortedSet(String key) {
        return redissonClient.getScoredSortedSet(key);
    }

    // ============================== 批量操作 ==============================

    /**
     * 批量执行命令（提高性能）
     */
    public void executeInBatch(Runnable runnable) {
        RBatch batch = redissonClient.createBatch();
        try {
            runnable.run();
            batch.execute();
        } finally {
            batch.discard();
        }
    }

    // ============================== 工具方法 ==============================

    /**
     * 获取Redisson客户端
     */
    public RedissonClient getClient() {
        return redissonClient;
    }

    /**
     * 关闭Redisson客户端
     */
    public void shutdown() {
        redissonClient.shutdown();
    }
}
```
