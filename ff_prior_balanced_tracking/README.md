# 新七障碍动态接续实现

仅用于用户已授权2+5完整工程单元的同场景真实末态接续。源码可在静态启动前冻结；动态输入仅在新静态模型/执行/836重放/独立审查收尾通过后冻结。尚未有新动态结果。

纯软件检查：`.venv\Scripts\python.exe -m unittest ff_prior_balanced_tracking.test_workflow`。
真实末态契约存在后才运行`test_terminal_preflight`；它不推进原生积分。不能用伪造terminal运行预检查。
