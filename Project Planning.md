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

* **Gautam:** Investigate ESPNet  
  * **Architecture Decomposition**

    * Document ESPnet2 pipeline: task-specific recipes, model definitions, training/inference separation

    * Analyze hybrid CTC/attention architecture: encoder options (Conformer, Branchformer, Transformer), decoder variants

    * Map transfer learning mechanisms: pre-trained model loading, layer freezing, warm-start capabilities

  * **Fine-Tuning Strategy Catalog**

    * Document ESPnet's adaptation approaches: layer-wise learning rates, warmup strategies, multi-stage training

    * Examine domain adaptation recipes: what modifications are standard for new domains/speakers?

    * Identify customization points in ESPnet config files (.yaml structure)

  * **RL Compatibility Analysis**

    * Assess training loop modularity: Can the standard trainer be extended for RL?

    * Evaluate reward integration points: Where can WER/custom metrics influence parameter updates?

    * Check multi-task learning support: Can RL objectives co-exist with supervised objectives?

    * Analyze distributed training compatibility for large-scale RL rollouts

  * **Practical Validation**

    * Run ESPnet recipe on LibriSpeech test-clean subset

    * Modify training script to log intermediate predictions for reward computation simulation

    * Test custom scorer integration (simulate reward model injection)

* **Shivangi:**  
  * Investigate NVIDIA NeMo (More details [here](?tab=t.rw18hqd52mvq))  
    * **Architecture Decomposition**

      * Document NeMo's ASR training pipeline components: data loader, model architecture (FastConformer/Conformer-Transducer), optimizer configuration, loss computation

      * Map the speech\_to\_text\_finetune.py workflow: initialization, training loop, validation, checkpointing

      * Identify PEFT integration points: LoRA adapter locations, rank/alpha configuration options, parameter freezing strategies

    * **Fine-Tuning Strategy Catalog**

      * Document supported methods: Full fine-tuning, LoRA, Adapter modules, BitFit

      * Extract parameter efficiency metrics: trainable parameter percentage for each method

      * Analyze training speed and memory footprint from NeMo documentation and benchmarks

    * **RL Compatibility Analysis**

      * Examine loss computation flexibility: Can custom reward-based losses replace CTC/RNN-T losses?

      * Assess gradient flow: Are gradients accessible for policy gradient methods?

      * Evaluate data pipeline: Can it handle dynamic sampling strategies (experience replay, prioritized sampling)?

      * Check checkpoint compatibility: Can intermediate states be saved/restored for RL episode management?

    * **Practical Validation**

      * Run baseline fine-tuning experiment on Common Voice 17.0 subset (1000 samples)

      * Document actual training loop behavior, logging outputs, and hook points

      * Test custom loss function injection to simulate reward-based training

**Week 7: (16th March \- 20th March): Spring Break \-\> Catch up on backlog**

**Week 8: (23rd March \- 27th Mach)**

* **Gautam:**   
  * **RL Framework Survey**

    * For each framework (Verl, OpenRLHF, TRL, Ray RLlib): Document core capabilities, ASR applicability, integration complexity

    * Focus on policy gradient implementations: REINFORCE, PPO, A2C support

    * Assess speech-specific features: handling variable-length sequences, CTC/attention compatibility

    * Evaluate experience replay mechanisms: buffer implementation, sampling strategies

  * **Cross-Framework Comparison**

    * Create taxonomy of RL components: policy network, value network, reward computation, experience buffer, optimizer

    * Map which ASR frameworks provide which RL components natively

    * Identify missing components that must be custom-built

* **Shivangi:** Investigate Align-SLM  
  * Analyze the RLAIF (RL with AI Feedback) implementation in Align-SLM

  * Document Direct Preference Optimization (DPO) adaptation for speech

  * Extract semantic metric computation: how are preference pairs generated?

  * Identify the reward model architecture and training methodology

**Week 9 (30th March \-  3rd April):**

* **Gautam & Shivangi:** Write the Survey Paper and send to prof for feedback


**—-----------------------------------------------------------------------------------------------**

**Plan Below This has to Be Changed Depending on how the first 9 Weeks Go**

**Week 9 (30th March \-  3rd April): Complete Google Cloud Integration**

* **Gautam:**  
  * Set up **Google Cloud Storage (GCS)** buckets.  
  * Deploy the Docker container to a Spot Instance to test connectivity.

* **Shivangi:**  
  * Establish the different fine-tuning strategies to be used.  
  * Run a "Smoke Test" on GCP: Train for 1 epoch on a tiny dataset. Verify checkpoints save correctly to GCS/VM.  
* **Kavya:**   
  * Set up a batch inference job on GCP to run **OpenAI Whisper**. Establish the "Upper Bound" baseline.  
    ---

## **Phase 2: Development & "Prevention" Implementation (Weeks 6–8)**

*Goal: Build the "Simulator" and the "Replay Buffer" before running major experiments.*

**Week 6: The Generic Simulator**

* **R1:** Build the **Simulator**: A script that runs each scenario against each model combo with each FT-ing methodology and each dataset selection  
* **R2:** Integrate the Simulator into the GCP training loop. Ensure the model accepts dynamic data streams.  
* **R3:** Design the **Sequential Drift Protocol**: Define "Batch 1" (Clean), "Batch 2" (Noisy), "Batch 3" (Accented). Validate the `UniversalEvaluator` works on these specific splits.

**Week 7: Distillation & Oracle Training**

* **R1:** Optimize the Oracle generation script for cost (batch processing).

* **R2:** Run the **Oracle Distillation** experiment: Train Wav2Vec2 using the synthetic "Gold" transcripts.

* **R3:** Analyze Distillation results using the Unified Module. Compare "Student" vs. "Oracle" performance.

**Week 8: Replay Buffer Implementation (Prevention Mechanism)**

* **R1:** Ensure GCP storage can handle "historical" data retrieval efficiently (low latency fetch).

* **R2 (Critical):** Implement the **Experience Replay Buffer**. Modify the data loader to mix $10-20%$ of samples from previous batches into the current training step.

* **R3:** Create the specific "Replay Datasets" (e.g., "Batch 2 mixed with Batch 1").

  ---

## **Phase 3: The Core "Lifelong Learning" Experiments (Weeks 9–11)**

*Goal: Prove that the system learns continuously WITHOUT forgetting.*

**Week 9: The "Naive" Experiment (Baseline Failure)**

* **R1:** Launch the "Naive" training job (Sequential training *without* Replay Buffer). Monitor for memory leaks.

* **R2:** Monitor loss curves.

* **R3:** Evaluate the Naive model on Batch 1 data after it finishes Batch 3 using the Unified Module. Calculate **Backward Transfer (BWT)** to quantify forgetting.

**Week 10: The "Robust" Experiment (The Solution)**

* **R1:** Launch the "Robust" training job (Sequential training *with* Replay Buffer).

* **R2:** Compare validation loss stability between Naive and Robust runs in real-time.

* **R3:** Evaluate the Robust model on Batch 1\. **Success Metric:** BWT should be near-zero or positive (indicating knowledge retention).

**Week 11: Trajectory Analysis & Ops Metrics**

* **R1:** Calculate **Real-Time Factor (RTF)** and **Latency p99** on GCP. Generate the **Cost Efficiency** table (Cost per 1% WER reduction).

* **R2:** Generate the **Visual Trajectory Plot**: Overlay the Naive vs. Robust WER curves.

* **R3:** Perform statistical significance testing (p-values) on the results.

  ---

## **Phase 4: Robustness & Theory (Weeks 12–13)**

*Goal: Prove safety and explain "Why" it works.*

**Week 12: The "Poison" Test (Safety)**

* **R1:** Create a "Poisoned" dataset (transcripts with intentional errors/hallucinations).

* **R2:** Run the fine-tuning loop with Poisoned data. Tune the **Confidence/Rejection Threshold** until the system automatically rejects updates.

* **R3:** Document the "Attack" and "Defense" success rates.

**Week 13: Bag of Tricks (Math & Ablation)**

* **R1:** Assist with visualizing the "Tricks" (e.g., Scheduler logic diagrams).

* **R2:** Finalize **Ablation Study**: Run quick checks to isolate the gain from *just* LLM correction vs. *just* Replay Buffer.

* **R3:** Write the **Mathematical Formulation**. Explain the Replay Buffer as an approximation of the joint probability distribution over all tasks.

  ---

## **Phase 5: Publication (Weeks 14–15)**

*Goal: Finalize the paper.*

**Week 14: Drafting & Figures**

* **R1:** Finalize "Infrastructure" section and "Ops Metrics" table. Create the System Architecture diagram.

* **R2:** Write "Methodology" (Replay Buffer, Oracle) and "Results" (Lifelong Learning).

* **R3:** Write "Introduction," "Related Work," and "Abstract." Ensure the single `UniversalEvaluator` is described clearly as the standard for all reported numbers.

**Week 15: Final Polish & Submission**

* **All Hands:**

  * **Mon-Tue:** Full paper read-through. Cross-check all numbers against raw logs.

  * **Wed:** LaTeX formatting fixes.

  * **Thu:** Final sanity check of citations and tables.

  * **Fri:** Submit.

