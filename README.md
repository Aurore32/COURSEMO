## README!!!!! (!)

Hi :)

欢迎各位加入我们团队！以下是我们这个项目前几个月工作流程的一个简短总结：

### 项目一览

- 目的：按照A-Level的考试标准以及A-Level学生的需求为各个科目打造一个“超级教师”大语言模型 - 可以帮助学生出题，答题，讲题，判卷，~~以及对学生施加言语暴力~~（跟真正的老师一样）
- 每个科目都需要训练单独的模型（可能还不止一个）；e.g. 不能用物理数据训练用经济数据训练过的模型，否则到时候你物理真就是经济老师教的了
- 目前的工作：
  - 为各科收集数据（工作量高，~60%）
  - 训练模型（工作量低，~15%)
  - 部署模型+找模型里的bug+接入小程序（工作量取决于我代码编的怎么样，which is to say估计烂的要死）
- 目前已完成：
  - 物理画图+答题模型
  - 经济画图+答题模型
- 目前正在进行：
  - 化学，生物，数学相关模型（高优先级）；历史，英文文学，心理，社会学相关模型（低优先级)
  
### 模型训练流程
  - 大体上来说有三个部分：收集数据 -> 云端训练模型 -> 部署模型
  - 每个科目需要一个答题模型+一个画图模型（e.g. 用经济题目训练的经济答题模型 + 用经济相关的图训练的经济画图模型）
    - AI模型缺乏画图能力（至少在三次元里），所以我们需要让模型生成可以转换成图片的代码（LaTeX代码/以后其他科目可能需要其他格式）
  
  - **收集数据**

    - 答题方面：

      - 每个科目都会分成不同题型，包括A-Level和IGCSE题型
      - 例：经济 - A-Level 8，12，13，20，25分题；Data Response；IGCSE 2，4，6，8分题；选择题
      - 每个题型需要至少50-100个题目+标准答案的组合（最好超过100个，越多越好）
      - 我们希望我们最终的模型生成东西尽可能贴近这些标准答案，所以标准答案本身得足够好
      - 题目+题目Mark Scheme来源：
        - COURSEMO题库（各科有1000+道题）
        - 用Python代码抓取题库里的数据（BeautifulSoup和selenium）
        - 文件夹里有提前写好的代码示例：scraping_mcq.py（选择题题目+答案抓取），scraping_structured.py（结构题题目+答案抓取）
      - 短时期内靠人力收集这种数据量不太可能 -> 可以转换用很强大的付费AI模型（例：DeepSeek R1）去生成标准答案
        - 代码示例：deepseek_api_test_econ_structured_20_marks.py (DeepSeek生成经济20分题标准答案)
            - 给DeepSeek的指令格式可见这个文档
        - 数据集：题目+题目相应的Mark Scheme（直接把Mark Scheme喂给DeepSeek，这样能保证生成出来的答案是“标准”的）
        - 需要手工提供：
          - 至少1道题的标准答案，给DeepSeek作为例子用
          - 有关怎么按照A-Level标准回答该题型的完整描述（见代码示例）
        - **重要：模型不能画图，所以需要告诉模型在题目需要画图的时候要做什么（e.g. "Please draw a diagram to illustrate your answer)。可以把这个步骤标准化以下：告诉模型在需要画图的时候写一个(DIAGRAM: ...图片描述...)，把图片描述加到括号里。**
  
    - 画图方面：
      - 物理/经济：训练模型利用文字描述生成相应的LaTeX TikZ包代码，并把代码转换成图片
      - 例：模型生成的文字答案中可能会有以下画图描述 - (DIAGRAM: An image of a fat cat eating grass.)
      - 期待画图模型生成：满足该描述的TikZ代码
      - 需要300+个图片描述+对应描述的TikZ代码的数据集
      - 仍然选用DeepSeek生成TikZ代码；用Python提取出模型生成出来的标准答案中的(DIAGRAM: ...)字符串，然后把这些字符串放到数据库里喂给DeepSeek
      - 代码示例：diagrams.py（提取字符串）, deepseek_api_test_diagrams.py（生成标准答案）
    - 这方面的工作对于每个科目来说都会稍稍有别，到时候可以考虑考虑每一个科目有没有什么需要独特处理的细节
  - **训练模型**
    - 家里电脑的GPU显存甚至还不如一只刚出土的土豆，所以用云端GPU来训练模型
    - 我们利用Llama-3.3-70B-Instruct-bnb-4bit (https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit/) 作为基础模型（所有科目通用）
    - 云端服务器：https://www.runpod.io/console/pods
    - 需选用：NVIDIA GPU，显存48GB以上（满足这个之后选最便宜的那个就好）
    - 需用到：Python - Unsloth，Torch数据库；收集数据时整理的数据库文件（CSV格式）
    - 云端服务器的接口是一个Jupyter Notebook，里面可以运行Python代码
    - 代码示例：physics-train.ipynb（具体指示全都在里面！）
    - 模型训练会花费1-3个小时
    - 用到以下技术：
      - Unsloth（把模型压缩至4bit，大幅度节省显存消耗）
      - SFT（Supervised Finetuning - 喂给模型题目+标准答案，逐渐调整模型的参数，让模型生成的答案越来越接近标准答案）
      - LoRA（只在训练过程当中调整一小部分和标准答案相关的参数，例如训练经济题目的时候只调整和经济题目有关的参数 - 700亿个参数的模型只训练其中0.5%的参数）
        - 大幅度节省内存消耗，并把最终训练完的模型存储成一个只包含被训练过的参数的LoRA adapter文件（700MB-800MB，相比整个模型40GB+）
        - 训练完后直接把LoRA adapter文件从网站上下载下来 

  - **部署模型**
    - 目前使用beam.cloud无服务器端点（serverless endpoint）部署，但以后可能会用本地GPU
    - 可能还没有太多工作要做...?
