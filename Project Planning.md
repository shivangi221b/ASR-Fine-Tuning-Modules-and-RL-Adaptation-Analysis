## **Phase 1: Infrastructure, GCP & Unified Eval (Weeks 1–3)**

*Goal: Establish a scalable Cloud pipeline and a single source of truth for metrics.*

**Week 1 (2nd February \- 6th February): GCP Discovery & Architecture Design**

* **Gautam:**   
  * ~~Project Planning~~   
  * ~~Create scenarios and establish datasets to be used~~  
* **Kavya:** ~~Dockerize the local training environment. Ensure all dependencies (LoRA, Wav2Vec2) are reproducible in a Linux container.~~  
* **Shivangi:** ~~Clean the repo &~~ **~~Refactor Evaluation Module:~~** ~~Merge evaluation into a single module. Ensure it can handle both streaming predictions (inference) and batch files (offline test sets).~~


**Week 2 (9th February \- 13th February): Pipeline Migration & Smoke Testing**

* **Gautam:**   
  * ~~Investigate **GCP Compute Engine (GCE)** vs. **Vertex AI**.~~   
    * ~~Recommendation: Use **GCE** with Deep Learning VMs for custom training loops.~~   
    * ~~Check GPU quotas (T4/L4/A100) in your region.~~  
  * Set up ~~GCP Project, IAM roles~~   
  * Reach out to private organizations to request data collaboration for the assignment.  
    * ~~Reach out to Epic~~  
    * ~~Reach out to Ayo~~  
    * ~~Reach out to Elhadad Noemie~~  
    * ~~Reach out to UW Health researcher~~  
    * ~~Reach out to 50 CMIOs & CHIOs~~  
* **Shivangi:**   
  * Establish the models to be used.  
  * Reach out to research prof at the Epidemiology Department for Data  
  * Investigate legal Department Data  
* **Kavya:**   
  * **~~Upgrade Unified Evaluator:~~** ~~Add the new metrics into the single module created in Week 1~~  
    * **~~Diarization Error Rate (DER)~~**  
    * **~~Verb Error Rate~~**  
    * **~~Domain Error Rate~~**

**Week 3 (16th February \- 20th February): Baselines, Oracle & New Metrics**

* **Gautam:**   
  * ~~Create a data gathering engine that automatically downloads popular datasets, adds noise & variance to them, and generates manifests for them.~~  
    * ~~1\. \*\*Common Voice\*\* (17.0) \- Crowdsourced multi-accent speech~~  
    * ~~2\. \*\*LibriSpeech\*\* \- Clean and challenging audiobook recordings~~  
    * ~~3\. \*\*Speech Commands\*\* \- Keyword spotting (10h)~~  
    * ~~4\. \*\*VoxPopuli\*\* \- European Parliament speeches (122GB)~~  
    * ~~5\. \*\*TED-LIUM Release 3\*\* \- Conversational talks (430h)~~  
    * ~~6\. \*\*ST-AEDS\*\* \- Spontaneous speech (4.7h)~~  
    * ~~7\. \*\*PriMock57\*\* \- Medical consultations (57 samples)~~

* **Shivangi:** ~~Investigate Verl (More details [here](?tab=t.rw18hqd52mvq))~~  
  * **~~How is its performance on different data we have available?~~**  
* **Kavya:**  
  * ~~Develop the **"Oracle Teacher"** script (using GPT-4o/Llama 3 via API) to generate synthetic "Gold" transcripts for cases where we don’t have transcripts available~~

**Week 4: Investigate other Solutions (23rd Feb \- 27th Feb)**

* **Gautam:** ~~Write the literature portion of the survey paper~~  
  * ~~Investigate Verl, Miles, AReal, Slime, OpenRLHF, Hugging Face TRL, DeepSpeed‑Chat, TRLX, RL4LMs, and Ray RLlib~~  
    * **~~What does the module do?~~**  
    * **~~What Fine-Tuning Strategies does it use?~~**  
    * **~~What can we do differently?~~**  
* **Shivangi:** ~~Investigate Verl (More details [here](?tab=t.rw18hqd52mvq))~~  
  * **~~How is its performance on different data we have available?~~**

**Week 5 & 6: (2nd March \- 13th March)**

* **Gautam:** ~~Investigate ESPNet~~  
  * **~~Architecture Decomposition~~**

    * ~~Document ESPnet2 pipeline: task-specific recipes, model definitions, training/inference separation~~

    * ~~Analyze hybrid CTC/attention architecture: encoder options (Conformer, Branchformer, Transformer), decoder variants~~

    * ~~Map transfer learning mechanisms: pre-trained model loading, layer freezing, warm-start capabilities~~

  * **~~Fine-Tuning Strategy Catalog~~**

    * ~~Document ESPnet's adaptation approaches: layer-wise learning rates, warmup strategies, multi-stage training~~

    * ~~Examine domain adaptation recipes: what modifications are standard for new domains/speakers?~~

    * ~~Identify customization points in ESPnet config files (.yaml structure)~~

  * **~~RL Compatibility Analysis~~**

    * ~~Assess training loop modularity: Can the standard trainer be extended for RL?~~

    * ~~Evaluate reward integration points: Where can WER/custom metrics influence parameter updates?~~

    * ~~Check multi-task learning support: Can RL objectives co-exist with supervised objectives?~~

    * ~~Analyze distributed training compatibility for large-scale RL rollouts~~

  * **~~Practical Validation~~**

    * ~~Run ESPnet recipe on LibriSpeech test-clean subset~~

    * ~~Modify training script to log intermediate predictions for reward computation simulation~~

    * ~~Test custom scorer integration (simulate reward model injection)~~

* **Shivangi:**  
  * ~~Investigate NVIDIA NeMo (More details [here](?tab=t.rw18hqd52mvq))~~  
    * **~~Architecture Decomposition~~**

      * ~~Document NeMo's ASR training pipeline components: data loader, model architecture (FastConformer/Conformer-Transducer), optimizer configuration, loss computation~~

      * ~~Map the speech\_to\_text\_finetune.py workflow: initialization, training loop, validation, checkpointing~~

      * ~~Identify PEFT integration points: LoRA adapter locations, rank/alpha configuration options, parameter freezing strategies~~

    * **~~Fine-Tuning Strategy Catalog~~**

      * ~~Document supported methods: Full fine-tuning, LoRA, Adapter modules, BitFit~~

      * ~~Extract parameter efficiency metrics: trainable parameter percentage for each method~~

      * ~~Analyze training speed and memory footprint from NeMo documentation and benchmarks~~

    * **~~RL Compatibility Analysis~~**

      * ~~Examine loss computation flexibility: Can custom reward-based losses replace CTC/RNN-T losses?~~

      * ~~Assess gradient flow: Are gradients accessible for policy gradient methods?~~

      * ~~Evaluate data pipeline: Can it handle dynamic sampling strategies (experience replay, prioritized sampling)?~~

      * ~~Check checkpoint compatibility: Can intermediate states be saved/restored for RL episode management?~~

    * **~~Practical Validation~~**

      * ~~Run baseline fine-tuning experiment on Common Voice 17.0 subset (1000 samples)~~

      * ~~Document actual training loop behavior, logging outputs, and hook points~~

      * ~~Test custom loss function injection to simulate reward-based training~~

**Week 7: (16th March \- 20th March): Spring Break (Holiday)**

**Week 8: Catch up on past work**

**Week 9 & 10:**

* **Gautam:**   
  * **~~RL Framework Survey~~**

    * ~~For each framework (Verl, OpenRLHF, TRL, Ray RLlib): Document core capabilities, ASR applicability, integration complexity~~

    * ~~Focus on policy gradient implementations: REINFORCE, PPO, A2C support~~

    * ~~Assess speech-specific features: handling variable-length sequences, CTC/attention compatibility~~

    * ~~Evaluate experience replay mechanisms: buffer implementation, sampling strategies~~

  * **~~Cross-Framework Comparison~~**

    * ~~Create taxonomy of RL components: policy network, value network, reward computation, experience buffer, optimizer~~

    * ~~Map which ASR frameworks provide which RL components natively~~

    * ~~Identify missing components that must be custom-built~~

* **Shivangi:** ~~Investigate Align-SLM~~  
  * ~~Analyze the RLAIF (RL with AI Feedback) implementation in Align-SLM~~

  * ~~Document Direct Preference Optimization (DPO) adaptation for speech~~

  * ~~Extract semantic metric computation: how are preference pairs generated?~~

  * ~~Identify the reward model architecture and training methodology~~

**Week 11:**

* **Gautam & Shivangi:** ~~Investigate results from experimentation, plan more robust experiments~~

Week 12 & 13:

* Run complete experiments for ESPNet & Nemo

Week 14:

* Complete Survey Paper

Week 15:

* Make presentation