### 4. target-only，再放开 (\theta)，最后才放开 (y)

我同意你这个顺序，而且建议比你写得更明确一点：

#### 4a. 放开 (x,\theta)，锁死 (y)

这一步专门看：

* 自发偏航是否过强
* `theta(t)` 是否连续
* 推进是否因为角漂移被破坏

#### 4b. 再放开 (x,y,\theta)

这时才是真正的 full free-swimming target-only baseline。

原因是：

* angular recoil 对效率和推进表现很重要
* lateral recoil 也会改变结果
* 但一上来全放开，很难判断是哪个自由度先出问题。([PMC][4])

---