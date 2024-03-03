# awesome-llm-rag

## 2024.02

**Bryndza at ClimateActivism 2024: Stance, Target and Hate Event Detection via Retrieval-Augmented GPT-4 and LLaMA** \
\[[paper](http://arxiv.org/abs/2402.06549v1)\] \
**Authors:** Marek Šuppa, Daniel Skala, Daniela Jašš, Samuel Sučík, Andrej Švec, Peter Hraška\
**Summary:** Proposed the use of retrieval-augmented GPT-4 and LLaMA for Climate Activism Stance and Hate Event Detection tasks, significantly outperforming baselines and securing second place in Target Detection.

**G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering** \
\[[paper](http://arxiv.org/abs/2402.07630v1)\] \
**Authors:** Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, Bryan Hooi\
**Summary:** The G-Retriever approach integrates GNNs, LLMs, and RAG for textual graph understanding and question answering, outperforming baselines in handling large-scale, real-world graphs and reducing hallucinations by formulating a Prize-Collecting Steiner Tree problem, as evidenced in the GraphQA benchmark.

**Retrieval-Augmented Thought Process as Sequential Decision Making** \
\[[paper](http://arxiv.org/abs/2402.07812v1)\] \
**Authors:** Thomas Pouplin, Hao Sun, Samuel Holt, Mihaela van der Schaar\
**Summary:** Proposed the Retrieval-Augmented Thought Process (RATP), a method that treats thought generation in Large Language Models as a sequential decision-making task, enhanced with a Q-value estimator and Monte-Carlo Tree Search, achieving a 50% improvement in question-answering tasks over prior retrieval-augmented models.

**PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models** \
\[[paper](http://arxiv.org/abs/2402.07867v1)\] \
**Authors:** Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia\
**Summary:** Proposed PoisonedRAG, a set of knowledge poisoning attacks against Retrieval-Augmented Generation for large language models, achieving up to 90% attack success rates by injecting just five poisoned texts into databases with millions of entries, underscoring the insufficiency of current defenses.

**Pandora: Jailbreak GPTs by Retrieval Augmented Generation Poisoning** \
\[[paper](http://arxiv.org/abs/2402.08416v1)\] \
**Authors:** Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, Yang Liu\
**Summary:** The paper proposes Pandora, a novel indirect jailbreak attack method leveraging Retrieval Augmented Generation Poisoning, which, through prompt manipulation, influences the output of large language models like GPTs, demonstrating a success rate of 64.3% on GPT-3.5 and 34.8% on GPT-4 in preliminary tests.

**Multi-Query Focused Disaster Summarization via Instruction-Based Prompting** \
\[[paper](http://arxiv.org/abs/2402.09008v1)\] \
**Authors:** Philipp Seeberger, Korbinian Riedhammer\
**Summary:** Proposed a two-stage retrieval pipeline with BM25 and MonoT5, coupled with a QA-motivated prompting approach for summarization using LLaMA-13b, which demonstrated strong performance in multi-query disaster summarization based on web sources, but also revealed discrepancies between open-source and proprietary systems in automatic and human evaluations.

**HGOT: Hierarchical Graph of Thoughts for Retrieval-Augmented In-Context Learning in Factuality Evaluation** \
\[[paper](http://arxiv.org/abs/2402.09390v1)\] \
**Authors:** Yihao Fang, Stephen W. Thomas, Xiaodan Zhu\
**Summary:** The paper introduces HGOT, a hierarchical, multi-layered graph approach for improving in-context factuality evaluation in large language models, which incorporates a weighted majority voting system based on citation quality and outperforms competitive methods by up to 7%.

**LoraRetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild** \
\[[paper](http://arxiv.org/abs/2402.09997v1)\] \
**Authors:** Ziyu Zhao, Leilei Gan, Guoyin Wang, Wangchunshu Zhou, Hongxia Yang, Kun Kuang, Fei Wu\
**Summary:** Proposed LoraRetriever, an input-aware framework for dynamic retrieval and composition of multiple LoRAs for enhancing large language models on mixed tasks, achieved consistent outperformance over baselines, demonstrating practical effectiveness and versatility.

**Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models** \
\[[paper](http://arxiv.org/abs/2402.10612v1)\] \
**Authors:** Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, Xueqi Cheng\
**Summary:** Proposed Rowen, an adaptive retrieval augmentation system with a multilingual semantic-aware detection module, achieved state-of-the-art performance in detecting and mitigating hallucination in large language models by selectively leveraging external information.

**EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models** \
\[[paper](http://arxiv.org/abs/2402.10866v1)\] \
**Authors:** Muhammad Shihab Rashid, Jannat Ara Meem, Yue Dong, Vagelis Hristidis\
**Summary:** Proposed EcoRank, a budget-constrained text re-ranking method using a two-layered pipeline that optimizes budget allocation across prompt strategies and Large Language Model APIs, achieved better performance than other supervised and unsupervised baselines on four QA and passage reranking datasets.

**What Evidence Do Language Models Find Convincing?** \
\[[paper](http://arxiv.org/abs/2402.11782v1)\] \
**Authors:** Alexander Wan, Eric Wallace, Dan Klein\
**Summary:** The study introduces ConflictingQA, a dataset to evaluate retrieval-augmented language models' (LLMs) responses to controversial queries, revealing that models prioritize website relevance over human-valued stylistic features and underscoring the need for corpus quality control and training adjustments to mimic human judgment.

**Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs** \
\[[paper](http://arxiv.org/abs/2402.12052v1)\] \
**Authors:** Jiejun Tan, Zhicheng Dou, Yutao Zhu, Peidong Guo, Kun Fang, Ji-Rong Wen\
**Summary:** Proposed SlimPLM, a novel approach utilizing a lightweight proxy model to identify gaps in large language models' knowledge, resulting in enhanced question-answering performance with reduced computational expense, achieving state-of-the-art results on five datasets.

**BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence** \
\[[paper](http://arxiv.org/abs/2402.12174v1)\] \
**Authors:** Jiajie Jin, Yutao Zhu, Yujia Zhou, Zhicheng Dou\
**Summary:** Proposed BIDER, an approach that synthesizes Key Supporting Evidence to align retrieval documents with LLM preferences through supervised fine-tuning and reinforcement learning, achieving a 7% increase in answer quality and an 80% reduction in input content length across five datasets.

**ARKS: Active Retrieval in Knowledge Soup for Code Generation** \
\[[paper](http://arxiv.org/abs/2402.12317v1)\] \
**Authors:** Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao Shi, Che Liu, Qian Liu, Tao Yu\
**Summary:** The paper introduces ARKS, a strategy that enhances large language models' (LLMs) code generation by integrating multiple knowledge sources such as web search, documentation, and execution feedback, featuring an active retrieval process that iteratively updates the input data, achieving substantial improvements in execution accuracy on a new benchmark tailored to evolving libraries and rare programming languages, with experimental validation on ChatGPT and CodeLlama.

**Graph-Based Retriever Captures the Long Tail of Biomedical Knowledge** \
\[[paper](http://arxiv.org/abs/2402.12352v1)\] \
**Authors:** Julien Delile, Srayanta Mukherjee, Anton Van Pamel, Leonid Zhukov\
**Summary:** The study introduced a novel knowledge-graph-based information retrieval method for biomedical research which mitigates the issue of overlooking rare information by downsampling over-represented concept clusters, doubling the precision and recall compared to embedding similarity methods, and further enhanced retrieval performance through a hybrid model that surpasses both individual approaches.

**SoftQE: Learned Representations of Queries Expanded by LLMs** \
\[[paper](http://arxiv.org/abs/2402.12663v1)\] \
**Authors:** Varad Pimpalkhute, John Heyer, Xusen Yin, Sameer Gupta\
**Summary:** SoftQE introduces an approach that integrates Large Language Model knowledge into query encoder representations, substantially enhancing out-of-domain retrieval performance on BEIR tasks by 2.83 percentage points while maintaining low latency and cost.

**SymBa: Symbolic Backward Chaining for Multi-step Natural Language Reasoning** \
\[[paper](http://arxiv.org/abs/2402.12806v1)\] \
**Authors:** Jinu Lee, Wonseok Hwang\
**Summary:** The paper proposes SymBa, a novel integration of symbolic top-down solving with Large Language Models for multi-step reasoning, resulting in enhanced performance, proof faithfulness, and efficiency across several reasoning benchmarks.

**Exploring the Impact of Table-to-Text Methods on Augmenting LLM-based Question Answering with Domain Hybrid Data** \
\[[paper](http://arxiv.org/abs/2402.12869v1)\] \
**Authors:** Dehai Min, Nan Hu, Rihui Jin, Nuo Lin, Jiaoyan Chen, Yongrui Chen, Yu Li, Guilin Qi, Yun Li, Nijun Li, Qianren Wang\
**Summary:** Proposed an integration of table-to-text generation methods into enhancing LLM-based QA systems with hybrid data and conducted a comparative analysis on different methods, finding that some significantly improve QA performance within industrial datasets when measured against DSFT and RAG frameworks.

**Self-DC: When to retrieve and When to generate? Self Divide-and-Conquer for Compositional Unknown Questions** \
\[[paper](http://arxiv.org/abs/2402.13514v1)\] \
**Authors:** Hongru Wang, Boyang Xue, Baohang Zhou, Tianhua Zhang, Cunxiang Wang, Guanhua Chen, Huimin Wang, Kam-fai Wong\
**Summary:** The paper introduces a Self Divide-and-Conquer (Self-DC) framework to more efficiently handle compositional unknown questions in open-domain question-answering, by adaptively deciding between retrieval or generation methods as needed; in tests on the newly proposed Compositional unknown Question-Answering dataset (CuQA) and FreshQA, Self-DC demonstrated equivalent or superior performance with significantly fewer retrieval times compared to other strong baselines.

**ARL2: Aligning Retrievers for Black-box Large Language Models via Self-guided Adaptive Relevance Labeling** \
\[[paper](http://arxiv.org/abs/2402.13542v1)\] \
**Authors:** Lingxi Zhang, Yue Yu, Kuan Wang, Chao Zhang\
**Summary:** The paper proposes ARL2, a retriever learning technique which utilizes large language models (LLMs) as labelers to annotate and score relevant evidence, and introduces an adaptive self-training strategy for curating high-quality, diverse relevance data, resulting in a 5.4% accuracy improvement on NQ and a 4.6% improvement on MMLU compared to existing methods, and demonstrating robust transfer learning capabilities and strong zero-shot generalization abilities.

**ActiveRAG: Revealing the Treasures of Knowledge via Active Learning** \
\[[paper](http://arxiv.org/abs/2402.13547v1)\] \
**Authors:** Zhipeng Xu, Zhenghao Liu, Yibin Liu, Chenyan Xiong, Yukun Yan, Shuo Wang, Shi Yu, Zhiyuan Liu, Ge Yu\
**Summary:** The paper presents ActiveRAG, a new Retrieval Augmented Generation framework that fosters active learning through a Knowledge Construction mechanism and a Cognitive Nexus mechanism, resulting in a 5% improvement on question-answering datasets when compared to previous RAG models.

**Rule or Story, Which is a Better Commonsense Expression for Talking with Large Language Models?** \
\[[paper](http://arxiv.org/abs/2402.14355v1)\] \
**Authors:** Ning Bian, Xianpei Han, Hongyu Lin, Yaojie Lu, Ben He, Le Sun\
**Summary:** The paper demonstrates that expressing commonsense as stories rather than rules yields higher confidence and accuracy in large language models, with stories performing better on daily events and rules on scientific questions, as evidenced by experimental results across 28 commonsense QA datasets.

**Causal Graph Discovery with Retrieval-Augmented Generation based Large Language Models** \
\[[paper](http://arxiv.org/abs/2402.15301v1)\] \
**Authors:** Yuzhe Zhang, Yipeng Zhang, Yidong Gan, Lina Yao, Chen Wang\
**Summary:** The paper proposed a Retrieval-Augmented Generation (RAG) approach using Large Language Models for causal graph recovery, which outperformed traditional methods by constructing high-quality causal graphs from scientific literature, as evidenced by its performance on the SACHS dataset.

**Citation-Enhanced Generation for LLM-based Chatbot** \
\[[paper](http://arxiv.org/abs/2402.16063v1)\] \
**Authors:** Weitao Li, Junkai Li, Weizhi Ma, Yang Liu\
**Summary:** Proposed a post-hoc Citation-Enhanced Generation (CEG) method for LLM-based chatbots which uses a retrieval module and a natural language inference-based citation generation module that regenerates responses to ensure all statements are supported by citations, achieving superior performance in hallucination detection and response regeneration on three benchmarks without additional training.

**Chain-of-Discussion: A Multi-Model Framework for Complex Evidence-Based Question Answering** \
\[[paper](http://arxiv.org/abs/2402.16313v1)\] \
**Authors:** Mingxu Tao, Dongyan Zhao, Yansong Feng\
**Summary:** Proposed a multi-model Chain-of-Discussion framework to improve evidence-based question answering, achieving more correct and comprehensive answers by leveraging the synergy among multiple Large Language Models.

**A Fine-tuning Enhanced RAG System with Quantized Influence Measure as AI Judge** \
\[[paper](http://arxiv.org/abs/2402.17081v1)\] \
**Authors:** Keshav Rangan, Yiqiao Yin\
**Summary:** The paper proposes a fine-tuning enhanced retrieval-augmented generation system integrating fine-tuned large language models with vector databases and introduces the Quantized Influence Measure (QIM) as an "AI Judge," achieving improved accuracy and performance by incorporating user feedback and optimizing parameters through LoRA and QLoRA methods.

**REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering** \
\[[paper](http://arxiv.org/abs/2402.17497v1)\] \
**Authors:** Yuhao Wang, Ruiyang Ren, Junyi Li, Wayne Xin Zhao, Jing Liu, Ji-Rong Wen\
**Summary:** The paper presents REAR, a relevance-aware retrieval-augmented framework for open-domain QA, featuring a new rank head architecture and bi-granularity relevance fusion training method, which significantly outperforms competitive RAG approaches in four QA tasks.

**JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability** \
\[[paper](http://arxiv.org/abs/2402.17887v1)\] \
**Authors:** Junda Wang, Zhichao Yang, Zonghai Yao, Hong Yu\
**Summary:** Proposed a synchronized Joint Medical LLM and Retrieval Training (JMLR) method, achieving superior performance on medical question-answering tasks with significant improvements (JMLR-13B: 81.2% on AMBOSS, 61.3% on MedQA) over standard pre-training and fine-tuning methods, while reducing training time from 144 hours to 37 hours.

**An Iterative Associative Memory Model for Empathetic Response Generation** \
\[[paper](http://arxiv.org/abs/2402.17959v1)\] \
**Authors:** Zhou Yang, Zhaochun Ren, Yufeng Wang, Chao Chen, Haizhou Sun, Xiaofei Zhu, Xiangwen Liao\
**Summary:** Proposed an Iterative Associative Memory Model (IAMM) with a second-order interaction attention mechanism that iteratively captures associated words in dialogue, improving empathetic response generation, validated by experiments on the Empathetic-Dialogue dataset, showing enhanced comprehension and expression over existing long sequence and independent utterances methods.

**Corpus-Steered Query Expansion with Large Language Models** \
\[[paper](http://arxiv.org/abs/2402.18031v1)\] \
**Authors:** Yibin Lei, Yu Cao, Tianyi Zhou, Tao Shen, Andrew Yates\
**Summary:** The paper introduces Corpus-Steered Query Expansion (CSQE), a method leveraging large language models for enhanced information retrieval by incorporating relevant sentences from a retrieval corpus to address LLM-induced hallucination and outdatedness, demonstrating significant performance improvements in relevance prediction without requiring training.

**Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation** \
\[[paper](http://arxiv.org/abs/2402.18150v1)\] \
**Authors:** Shicheng Xu, Liang Pang, Mo Yu, Fandong Meng, Huawei Shen, Xueqi Cheng, Jie Zhou\
**Summary:** The paper introduces InFO-RAG, an unsupervised information refinement training method for large language models in retrieval-augmented generation tasks, which improves the integration of retrieved text into coherent outputs and achieves a 9.39% average improvement across diverse datasets in tasks like Question Answering and Code Generation.

**Retrieval-based Full-length Wikipedia Generation for Emergent Events** \
\[[paper](http://arxiv.org/abs/2402.18264v1)\] \
**Authors:** Jiebin Zhang, Eugene J. Yu, Qinyu Chen, Chenhao Xiong, Dawei Zhu, Han Qian, Mingbo Song, Xiaoguang Li, Qun Liu, Sujian Li\
**Summary:** Proposed a retrieval-based approach for creating full-length Wikipedia articles on recent events, introduced Wiki-GenBen with 309 event-page pairs, and designed systematic evaluation metrics, significantly addressing the need for accurate and comprehensive automated Wikipedia document generation.
