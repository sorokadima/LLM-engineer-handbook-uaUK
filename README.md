
|| spoiler ||

# LLM-engineer-handbook

🔥 Великі мовні моделі (LLM) захопили ~~NLP-спільноту~~ ~~AI-спільноту~~ **увесь світ**.

### Чому ми створюємо цей репозиторій?

* Зараз кожен може зробити демо з LLM за кілька хвилин, але потрібен справжній LLM/AI-експерт, щоб закрити останній крок — питання продуктивності, безпеки й масштабованості.
* Простір LLM дуже складний! Цей репозиторій надає відібраний список ресурсів, які допоможуть зорієнтуватися та збільшити шанси створити застосунки на базі LLM, готові до продакшну. Тут зібрані фреймворки й туторіали про тренування моделей, сервінг, донавчання, застосування LLM та оптимізацію підказок (prompts), а також LLMOps.

*Однак класичне ML нікуди не зникає. Навіть LLM-и їх потребують. Класичні моделі застосовуються для захисту приватності даних, виявлення галюцинацій та іншого. Тому не забувайте вчити фундаментальні основи класичного ML.*

---

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
