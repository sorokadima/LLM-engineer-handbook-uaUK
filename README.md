

# LLM-engineer-handbook


## Огляд

Поточний робочий процес виглядає так: ви робите демо, використовуючи бібліотеку або SDK від постачальника LLM. Воно якось працює, але потрібно створювати додаткові датасети для оцінки й тренування, щоб оптимізувати продуктивність (точність, затримку, вартість).

Можна застосовувати prompt engineering або автооптимізацію підказок; можна створити більший датасет для донавчання LLM або використати Direct Preference Optimization (DPO), щоб узгодити модель із людськими вподобаннями.
Потім потрібно подумати про сервінг і LLMOps для масштабного розгортання моделі та пайплайни для оновлення даних.

Ми організували ресурси так:

1. бібліотеки, фреймворки й інструменти,
2. навчальні матеріали для всього життєвого циклу LLM,
3. розуміння LLM,
4. соціальні акаунти та спільнота,
5. як робити внесок у цей репозиторій.

---

* [LLM-engineer-handbook](#llm-engineer-handbook)

  * [Огляд](#огляд)
* [Бібліотеки & Фреймворки & Інструменти](#бібліотеки--фреймворки--інструменти)

  * [Застосунки](#застосунки)
  * [Передтренування](#передтренування)
  * [Донавчання](#донавчання)
  * [Сервінг](#сервінг)
  * [Управління підказками](#управління-підказками)
  * [Датасети](#датасети)
  * [Бенчмарки](#бенчмарки)
* [Навчальні ресурси по LLM](#навчальні-ресурси-по-llm)

  * [Застосунки](#застосунки-1)

    * [Агенти](#агенти)
  * [Моделювання](#моделювання)
  * [Тренування](#тренування)
  * [Донавчання](#донавчання-1)
  * [Фундаментальні основи](#фундаментальні-основи)
  * [Книги](#книги)
  * [Розсилки](#розсилки)
  * [Автооптимізація](#автооптимізація)
* [Розуміння LLM](#розуміння-llm)
* [Соцмережі & Спільнота](#соцмережі--спільнота)

  * [Соціальні акаунти](#соціальні-акаунти)
  * [Спільнота](#спільнота)
* [Внесок](#внесок)

---

# Бібліотеки & Фреймворки & Інструменти

## Застосунки

**Створення та автооптимізація**

* [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) — бібліотека для створення та автооптимізації LLM-застосунків: чат-ботів, RAG, агентів.
* [dspy](https://github.com/stanfordnlp/dspy) — фреймворк для програмування (а не тільки prompt-інженерії) foundation-моделей.

**Створення**

* [LlamaIndex](https://github.com/jerryjliu/llama_index) — Python-бібліотека для підживлення LLM-застосунків даними.
* [LangChain](https://github.com/hwchase17/langchain) — популярна бібліотека для Python/JS для ланцюжків промптів LLM.
* [Haystack](https://github.com/deepset-ai/haystack) — Python-фреймворк для створення застосунків на LLM.
* [Instill Core](https://github.com/instill-ai/instill-core) — платформа на Go для оркестрації LLM і створення AI-застосунків.

**Оптимізація підказок**

* [AutoPrompt](https://github.com/Eladlev/AutoPrompt) — фреймворк для тюнінгу підказок з Intent-based Prompt Calibration.
* [PromptFify](https://github.com/promptslab/Promptify) — бібліотека для prompt engineering, що спрощує NLP-завдання (NER, класифікація) з LLM на кшталт GPT.

**Інше**

* [LiteLLM](https://github.com/BerriAI/litellm) — Python SDK та проксі-сервер (LLM Gateway) для виклику 100+ LLM API у форматі OpenAI.

---

## Передтренування

* [PyTorch](https://pytorch.org/) — відкрита ML-бібліотека, популярна у CV та NLP.
* [TensorFlow](https://www.tensorflow.org/) — ML-бібліотека від Google.
* [JAX](https://github.com/jax-ml/jax) — бібліотека від Google для HPC та автодиференціювання.
* [tinygrad](https://github.com/tinygrad/tinygrad) — мінімалістична DL-бібліотека, створена George Hotz для освіти.
* [micrograd](https://github.com/karpathy/micrograd) — легкий автоград-движок від Andrej Karpathy для навчання.

---

## Донавчання

* [Transformers](https://huggingface.co/docs/transformers/en/installation) — популярна бібліотека HuggingFace для NLP та донавчання LLM.
* [Unsloth](https://github.com/unslothai/unsloth) — донавчання Llama 3.2, Mistral, Phi-3.5 і Gemma у 2–5 разів швидше з -80% пам’яті.
* [LitGPT](https://github.com/Lightning-AI/litgpt) — 20+ продуктивних LLM із рецептами для предтренування, донавчання та масштабного деплою.
* [AutoTrain](https://github.com/huggingface/autotrain-advanced) — no-code донавчання LLM та інших ML-завдань.

---

## Топ-моделі

* [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1) — найпопулярніша відкрита reasoning-модель, порівнювана з GPT-o1. Є технічний репорт і код.

---

## Сервінг

* [TorchServe](https://pytorch.org/serve/) — відкрита бібліотека для сервінгу моделей (open-source **model serving**), розроблена AWS та Facebook спеціально для моделей PyTorch; забезпечує масштабоване розгортання (scalable **deployment**), версіонування моделей (**model versioning**) та A/B-тестування (**A/B testing**).

* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) — гнучка, високопродуктивна система сервінгу (flexible, **high-performance serving system**) для моделей машинного навчання (**ML models**), створена для продакшн-середовищ (**production environments**), оптимізована під TensorFlow-моделі, але підтримує й інші формати (**other formats**).

* [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) — частина екосистеми Ray; масштабована бібліотека для сервінгу моделей (**scalable model-serving library**), що підтримує розгортання (**deployment**) ML-моделей у кількох фреймворках (**multiple frameworks**) із вбудованою підтримкою Python-API та конвеєрів моделей (**model pipelines**).

* [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — компілятор NVIDIA для трансформерних моделей (**transformer-based models / LLMs**), що надає передові оптимізації (**state-of-the-art optimizations**) на GPU NVIDIA.

* [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) — високопродуктивний сервер інференсу (**high-performance inference server**), який підтримує кілька ML/DL-фреймворків (TensorFlow, PyTorch, ONNX, TensorRT тощо), оптимізований для розгортань на GPU NVIDIA та придатний як для хмари (**cloud**), так і для локальних дата-центрів (**on-premises serving**).

* [ollama](https://github.com/ollama/ollama) — легковаговий, розширюваний фреймворк (**lightweight, extensible framework**) для збирання та запуску LLM на локальній машині (**local machine**).

* [llama.cpp](https://github.com/ggerganov/llama.cpp) — бібліотека для запуску LLM на чистих C/C++ (**pure C/C++**). Підтримувані архітектури: LLaMA, Falcon, Mistral, MoE, Phi тощо (**and more**).

* [TGI](https://github.com/huggingface/text-generation-inference) — набір інструментів Hugging Face для інференсу генерації тексту (**text-generation inference toolkit**) і деплою LLM, побудований на Rust, Python і gRPC.

* [vllm](https://github.com/vllm-project/vllm) — оптимізований, високопропускний рушій сервінгу (**optimized, high-throughput serving engine**) для LLM, спроєктований для масового інференсу з малою затримкою (**massive-scale inference with reduced latency**).

* [sglang](https://github.com/sgl-project/sglang) — швидкий фреймворк сервінгу (**fast serving framework**) для великих мовних моделей і мовно-візуальних моделей (**vision-language models**).

* [LitServe](https://github.com/Lightning-AI/LitServe) — надшвидкий рушій сервінгу (**lightning-fast serving engine**) для будь-яких ШІ-моделей будь-якого розміру; гнучкий (**flexible**), простий (**easy**), корпоративного масштабу (**enterprise-scale**).

## Керування підказками (Prompt Management)

* [Opik](https://github.com/comet-ml/opik) — відкрита платформа (**open-source platform**) для оцінювання, тестування та моніторингу (**evaluating, testing, monitoring**) застосунків на LLM.
* [Agenta](https://github.com/agenta-ai/agenta) — відкритоджерельна платформа інженерії LLM (**open-source LLM engineering platform**) із майданчиком для підказок (**prompt playground**), керуванням підказками (**prompt management**), оцінюванням (**evaluation**) та спостережністю (**observability**).

## Набори даних (Datasets)

Сценарії використання (Use cases)

* [Datasets](https://huggingface.co/docs/datasets/en/index) — величезна збірка готових датасетів (**ready-to-use datasets**) для задач ML, зокрема НЛП, комп’ютерного бачення та аудіо (**NLP, computer vision, audio**), із засобами для зручного доступу, фільтрації та препроцесингу (**access, filtering, preprocessing**).
* [Argilla](https://github.com/argilla-io/argilla) — UI-інструмент для куратування та рев’ю датасетів (**curating & reviewing datasets**) для оцінювання або тренування LLM.
* [distilabel](https://distilabel.argilla.io/latest/) — бібліотека для генерації синтетичних датасетів (**synthetic datasets**) за допомогою LLM-API або моделей.

Файн-тюнінг (Fine-tuning)

* [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) — швидкий гайд (**quick guide**) по трендових інструкційних датасетах для файн-тюнінгу (**instruction fine-tuning datasets**).
* [LLM Datasets](https://github.com/mlabonne/llm-datasets) — високоякісні датасети, інструменти та концепції (**high-quality datasets, tools, concepts**) для файн-тюнінгу LLM (**LLM fine-tuning**).

Попереднє тренування (Pretraining)

* [IBM LLMs Granite 3.0](https://www.linkedin.com/feed/update/urn:li:activity:7259535100927725569?updateEntityUrn=urn%3Ali%3Afs_updateV2%3A%28urn%3Ali%3Aactivity%3A7259535100927725569%2CFEED_DETAIL%2CEMPTY%2CDEFAULT%2Cfalse%29) — повний список датасетів (**full list of datasets**), використаних для тренування моделей IBM Granite 3.0.

## Бенчмарки (Benchmarks)

* [lighteval](https://github.com/huggingface/lighteval) — бібліотека для оцінювання локальних LLM (**evaluating local LLMs**) на основних бенчмарках і кастомних задачах (**major benchmarks & custom tasks**).

* [evals](https://github.com/openai/evals) — відкритий фреймворк OpenAI для оцінювання (**open-sourced evaluation framework**) LLM і систем на їхній основі.

* [ragas](https://github.com/explodinggradients/ragas) — бібліотека для оцінювання та оптимізації (**evaluating & optimizing**) LLM-застосунків із широким набором метрик (**rich set of eval metrics**).

Агенти (Agent)

* [TravelPlanner](https://osu-nlp-group.github.io/TravelPlanner/) — [paper](https://arxiv.org/pdf/2402.01622) бенчмарк (**benchmark**) для планування у реальному світі з мовними агентами (**language agents**).

# Навчальні ресурси щодо LLM (Learning Resources for LLMs)

Ми категоризуємо найкращі ресурси для вивчення LLM — від моделювання до тренування та застосунків (**from modeling to training and applications**).

### Застосунки (Applications)

Загальне (General)

* [Документація AdalFlow](https://adalflow.sylph.ai/) — містить туторіали (**tutorials**) від побудови RAG та агентів (**Agent**) до оцінювання LLM і файн-тюнінгу (**LLM evaluation & fine-tuning**).

* [CS224N](https://www.youtube.com/watch?v=rmVRLeJRkl4) — курс Стенфорда з основ НЛП (**NLP fundamentals**), LLM і побудови моделей на PyTorch (**PyTorch-based model building**), викладачі: Chris Manning та Shikhar Murty.

* [LLM-driven Data Engineering](https://github.com/DataExpert-io/llm-driven-data-engineering) — плейлист із 6 лекцій (**playlist of 6 lectures**) від \[Zach Wilson] про вплив LLM на розробку конвеєрів даних (**data pipelines**).

* [LLM Course by Maxime Labonne](https://github.com/mlabonne/llm-course) — end-to-end курс (**end-to-end course**) для інженерів AI/ML щодо відкритих LLM (**open-source LLMs**).

* [LLMOps Database by ZenML](https://www.zenml.io/llmops-database) — 500+ кейс-стаді (**case studies**) застосування LLM і GenAI у продакшні (**in production**) із короткими резюме та тегами для зручного пошуку.

#### Агенти (Agent)

Лекції (Lectures)

* [OpenAI Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) — практичний гід (**practical guide**) для продуктових і інженерних команд (**product & engineering teams**) з побудови перших агентів (**agents**); дистилює інсайти з численних продакшн-впроваджень у практики (**actionable best practices**): як знаходити кейси (**use cases**), проєктувати логіку та оркестрацію (**agent logic & orchestration**), а також забезпечувати безпечну, передбачувану та ефективну роботу (**safe, predictable, effective**).

* [Anthropic’s Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — огляд «цеглинок» та дизайн-патернів (**building blocks & design patterns**) для ефективних агентів.

* [LLM Agents MOOC](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&si=LAonD5VfG9jFAOuE) — плейлист із 11 лекцій (**playlist of 11 lectures**) від Berkeley RDI Center on Decentralization & AI з гостями (Yuandong Tian, Graham Neubig, Omar Khattab тощо), що висвітлює ключові теми агентів на LLM (**Large Language Model agents**). [CS294](https://rdi.berkeley.edu/llm-agents/f24)

* [12 factor agents](https://github.com/humanlayer/12-factor-agents) — спроба сформулювати принципи (**principles**) для продакшн-класу агентів (**production-grade agents**).

Пам’ять (Memory)

* [LLM Agent Memory Survey](https://github.com/nuster1128/LLM_Agent_Memory_Survey) — огляд механізмів пам’яті (**survey of memory mechanisms**) для LLM-агентів: короткострокова, довгострокова та гібридна (**short-term, long-term, hybrid**).

Оптимізація (Optimization)

* [Awesome-LLM-Agent-Optimization-Papers](https://github.com/YoungDubbyDu/Awesome-LLM-Agent-Optimization-Papers) — кураторська добірка статей (**curated list of papers**) з оптимізації LLM-агентів (**LLM agent optimization**).

Проєкти (Projects)

* [OpenHands](https://github.com/All-Hands-AI/OpenHands) — відкриті агенти для розробників (**open-source agents for developers**) від [AllHands](https://www.all-hands.dev/).
* [CAMEL](https://github.com/camel-ai/camel) — перший фреймворк багатоагентних систем на LLM (**multi-agent framework**) і спільнота, що досліджує «закони масштабування» агентів (**scaling law of agents**), від [CAMEL-AI](https://www.camel-ai.org/).
* [swarm](https://github.com/openai/swarm) — навчальний фреймворк (**educational framework**) для дослідження ергономічної, легковагової багатоагентної оркестрації (**lightweight multi-agent orchestration**), керований командою OpenAI Solutions.

* [AutoGen](https://github.com/microsoft/autogen) — програмний фреймворк (programming **framework**) для агентного ШІ (agentic **AI**) від Microsoft 🤖.
* [CrewAI](https://github.com/crewAIInc/crewAI) — передовий фреймворк (cutting-edge **framework**) для оркестрації рольових, автономних ШІ-агентів (role-playing, autonomous **AI agents**) 🤖.
* [TinyTroupe](https://github.com/microsoft/TinyTroupe) — симулює налаштовувані персони (customizable **personas**) на базі GPT-4 для тестування, інсайтів та інновацій (**testing, insights, innovation**) від Microsoft.

### Моделювання (Modeling)

* [Llama3 from scratch](https://github.com/naklecha/llama3-from-scratch) — реалізація Llama 3 «з нуля» (from **scratch**) у PyTorch, по одному множенню матриць за раз (**one matrix multiplication at a time**).
* [Interactive LLM visualization](https://github.com/bbycroft/llm-viz) — інтерактивна візуалізація (interactive **visualization**) трансформерів (transformers). [Візуалізатор (Visualizer)](https://bbycroft.net/llm)
* [3Blue1Brown transformers visualization](https://www.youtube.com/watch?v=wjZofJX0v4M) — відео 3Blue1Brown про те, як працюють трансформери (**transformers**).
* [Self-Attention explained as directed graph](https://x.com/akshay_pachaar/status/1853474819523965088) — допис в X із поясненням механізму самоуваги (self-attention) як орієнтованого графа (directed **graph**) від Akshay Pachaar.
* [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — репозиторій GitHub до книги *Build a Large Language Model (From Scratch)* Себастьяна Рашки; охоплює розробку (development), попереднє тренування (pretraining) і файн-тюнінг (fine-tuning) GPT-подібних моделей у PyTorch.

### Навчання (Training)

* [HuggingFace’s SmolLM & SmolLM2 training release](https://huggingface.co/blog/smollm) — про методи курації даних (data **curation**), оброблені дані (processed **data**), рецепти тренування (training **recipes**) та код (**code**). [GitHub-репо](https://github.com/huggingface/smollm?tab=readme-ov-file).
* [Lil'Log](https://lilianweng.github.io/) — блог Ліліан Венг (OpenAI) про ML, DL і ШІ (machine **learning**, deep **learning**, **AI**) з фокусом на LLM та NLP.
* [Chip’s Blog](https://huyenchip.com/blog/) — блог Чіп Хуєн про тренування LLM (LLM **training**), нові дослідження (latest **research**), туторіали (tutorials) та найкращі практики (best **practices**).

### Файн-тюнінг (Fine-tuning)

* [DPO](https://arxiv.org/abs/2305.18290): Rafailov, Rafael, et al. «Direct Preference Optimization: ваша мовна модель — це потайкова модель винагороди» (Direct **Preference Optimization**, **DPO**). NeurIPS 36 (2024). [Код (Code)](https://github.com/eric-mitchell/direct-preference-optimization).

### Основи (Fundamentals)

* [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g&t=1390s&ab_channel=AndrejKarpathy) — годинне загальне введення (general-audience **introduction**) до LLM від Андрія Карпатого.
* [Building GPT-2 from Scratch](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1564s&ab_channel=AndrejKarpathy) — чотиригодинний глибокий розбір (deep **dive**) побудови GPT-2 «з нуля» (**from scratch**) від Андрія Карпатого.

### Книги (Books)

* [LLM Engineer’s Handbook: Master the art of engineering large language models from concept to production](https://www.amazon.com/dp/1836200072?ref=cm_sw_r_cp_ud_dp_ZFR4XZPT7EY41ZE1M5X9&ref_=cm_sw_r_cp_ud_dp_ZFR4XZPT7EY41ZE1M5X9&social_share=cm_sw_r_cp_ud_dp_ZFR4XZPT7EY41ZE1M5X9) — Пол Юзстін, Максім Лабонн. Переважно про життєвий цикл LLM (LLM **lifecycle**): конвеєри LLMOps (LLMOps **pipelines**), деплой (deployment), моніторинг (monitoring) тощо. [Огляд на YouTube (overview)](https://www.youtube.com/live/6WmPfKPmoz0).
* [Build a Large Language Model from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) — Sebastian Raschka.
* [Hands-On Large Language Models: Build, Tune, and Apply LLMs](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961) — Jay Alammar, Maarten Grootendorst.
* [Generative Deep Learning — Teaching machines to Paint, Write, Compose and Play](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947) — David Foster.
* [Large Language Models: A Deep Dive: Bridging Theory and Practice](https://www.amazon.com/Large-Language-Models-Bridging-Practice/dp/3031656466) — Uday Kamath, Kevin Keenan, Garrett Somers, Sarah Sorenson.

### Розсилки (Newsletters)

* [Ahead of AI](https://magazine.sebastianraschka.com/) — розсилка Себастьяна Рашки з end-to-end розуміння LLM (end-to-end **LLMs understanding**).
* [Decoding ML](https://decodingml.substack.com/) — матеріали про продакшн-GenAI (production **GenAI**), системи рекомендацій (RecSys) і MLOps-застосунки.

### Автооптимізація (Auto-optimization)

* [TextGrad](https://github.com/zou-group/textgrad) — автоматичне «диференціювання» через текст (automatic **“differentiation” via text**) — використання LLM для «бекпропу» текстових градієнтів (backpropagate **textual gradients**).

# Розуміння LLM (Understanding LLMs)

Цікаво й корисно розуміти можливості, поведінку та обмеження LLM (capabilities, behaviors, limitations). Це безпосередньо допомагає інженерії підказок (prompt **engineering**).

Навчання в контексті (In-context Learning)

* [Brown, Tom B. «Language models are few-shot learners.» arXiv preprint arXiv:2005.14165 (2020).](https://rosanneliu.com/dlctfs/dlct_200724.pdf)

Міркування та планування (Reasoning & Planning)

* [Kambhampati, Subbarao, et al. «LLMs can’t plan, but can help planning in LLM-modulo frameworks.» arXiv preprint arXiv:2402.01817 (2024).](https://arxiv.org/abs/2402.01817)
* [Mirzadeh, Iman, et al. «GSM-symbolic: Understanding the limitations of mathematical reasoning in large language models.» arXiv preprint arXiv:2410.05229 (2024).](https://arxiv.org/abs/2410.05229) Від Apple.

# Соцмережі та спільноти (Social Accounts & Community)

## Соцмережі (Social Accounts)

Соцмережі — найкращий спосіб лишатися в курсі найактуальніших (up-to-date) досліджень LLM (LLM **research**), індустріальних трендів (industry **trends**) та найкращих практик (best **practices**).

| Ім’я                   | Соцмережа (Social)                                        | Експертиза (Expertise)                                                 |
| ---------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| Li Yin                 | [LinkedIn](https://www.linkedin.com/in/li-yin-ai)         | Автор AdalFlow & засновник SylphAI (AdalFlow author & SylphAI founder) |
| Chip Huyen             | [LinkedIn](https://www.linkedin.com/in/chiphuyen)         | Інженерія ШІ та ML-системи (AI engineering & ML systems)               |
| Damien Benveniste, PhD | [LinkedIn](https://www.linkedin.com/in/damienbenveniste/) | ML-системи та MLOps (ML systems & MLOps)                               |
| Jim Fan                | [LinkedIn](https://www.linkedin.com/in/drjimfan/)         | LLM-агенти та робототехніка (LLM agents & robotics)                    |
| Paul Iusztin           | [LinkedIn](https://www.linkedin.com/in/pauliusztin/)      | Інженерія LLM і LLMOps (LLM engineering & LLMOps)                      |
| Armand Ruiz            | [LinkedIn](https://www.linkedin.com/in/armand-ruiz/)      | Директор з інженерії ШІ в IBM (AI Engineering Director at IBM)         |
| Alex Razvant           | [LinkedIn](https://www.linkedin.com/in/arazvant/)         | Інженерія AI/ML (AI/ML engineering)                                    |
| Pascal Biese           | [LinkedIn](https://www.linkedin.com/in/pascalbiese/)      | LLM Papers Daily                                                       |
| Maxime Labonne         | [LinkedIn](https://www.linkedin.com/in/maxime-labonne/)   | Файн-тюнінг LLM (LLM fine-tuning)                                      |
| Sebastian Raschka      | [LinkedIn](https://www.linkedin.com/in/sebastianraschka/) | LLM «з нуля» (LLMs from scratch)                                       |
| Zach Wilson            | [LinkedIn](https://www.linkedin.com/in/eczachly)          | Інженерія даних для LLM (data engineering for LLMs)                    |
| Adi Polak              | [LinkedIn](https://www.linkedin.com/in/polak-adi/)        | Стрімінг даних для LLM (data streaming for LLMs)                       |
| Eduardo Ordax          | [LinkedIn](https://www.linkedin.com/in/eordax/)           | GenAI voice @ AWS                                                      |

## Спільнота (Community)

| Назва    | Соцмережа (Social)                       | Сфера (Scope)                                                                                                 |
| -------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| AdalFlow | [Discord](https://discord.gg/ezzszrRZvT) | Інженерія LLM, авто-підказки (auto-prompts) і обговорення/контриб’юції AdalFlow (discussions & contributions) |

















<details>
  <summary>Натисни, щоб побачити спойлер</summary>

  Тут прихований текст, який буде видно тільки після розгортання.
  
  Можна навіть кілька абзаців.
</details>
