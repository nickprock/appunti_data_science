import os
import math
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import BatchSamplers

def main():
    # -------------------------------------------------------------------
    # 0. SETUP AND LOGIN
    # -------------------------------------------------------------------
    print("0. Initialization...")
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    
    # Replace with your target repository ID
    REPO_ID = "your_username/your_model_name"
    
    # -------------------------------------------------------------------
    # 1. MODEL LOADING
    # -------------------------------------------------------------------
    print(f"1. Loading Baseline ({REPO_ID})...")
    # Disable SDPA to avoid illegal memory access bug with Gradient Cache
    model = SentenceTransformer(REPO_ID, model_kwargs={"attn_implementation": "eager"})

    # -------------------------------------------------------------------
    # 2. DATASET A PREPARATION: RETRIEVAL (Dense Hard Negatives Only!)
    # -------------------------------------------------------------------
    print("2. Loading Retrieval dataset (Dense Hard Negatives)...")
    # Using the file with semantic negatives to avoid BM25 false negatives
    FILE_DATASET_DENSE = "dataset_pronto_per_training_sem.json"
    dataset_retrieval_full = load_dataset("json", data_files=FILE_DATASET_DENSE, split="train")
    
    # Rename the column to match the SentenceTransformers standard
    dataset_retrieval_full = dataset_retrieval_full.rename_column("query", "anchor")
    
    # Train/test split (90/10) and then take 5% for the evaluator
    dataset_split_1 = dataset_retrieval_full.train_test_split(test_size=0.10, seed=42)
    dataset_split_2 = dataset_split_1['test'].train_test_split(test_size=0.5, seed=42)
    
    train_retrieval = dataset_split_1['train']
    test_retrieval = dataset_split_2['test']

    # -------------------------------------------------------------------
    # 3. DATASET B PREPARATION: STS (Similarity)
    # -------------------------------------------------------------------
    print("3. Preparing STS-B dataset...")
    dataset_sts = load_dataset("stsb_multi_mt", name="it", split="train")
    dataset_sts = dataset_sts.map(lambda x: {"score": x["similarity_score"] / 5.0})
    dataset_sts = dataset_sts.select_columns(["sentence1", "sentence2", "score"])

    # Balance STS with retrieval to avoid oversampling STS (which is ~8x smaller)
    sts_repeat = math.ceil(len(train_retrieval) / len(dataset_sts))
    dataset_sts = concatenate_datasets([dataset_sts] * sts_repeat)
    print(f"   STS repeated {sts_repeat}x -> {len(dataset_sts)} samples (retrieval: {len(train_retrieval)})")

    dataset_sts_dev = load_dataset("stsb_multi_mt", name="it", split="dev")
    dataset_sts_dev = dataset_sts_dev.map(lambda x: {"score": x["similarity_score"] / 5.0})
    # Filter columns in dev as well to prevent Tokenizer crashes
    dataset_sts_dev = dataset_sts_dev.select_columns(["sentence1", "sentence2", "score"])

    # -------------------------------------------------------------------
    # 4. EVALUATORS CONFIGURATION (MULTI-DIMENSIONAL)
    # -------------------------------------------------------------------
    print("4. Configuring Multi-Dimensional Evaluators...")
    
    # STS Evaluator (Max dimension only)
    sts_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=dataset_sts_dev["sentence1"],
        sentences2=dataset_sts_dev["sentence2"],
        scores=dataset_sts_dev["score"],
        name="sts-dev"
    )

    # Data preparation for Retrieval
    queries, corpus, relevant_docs = {}, {}, {}
    for i, row in enumerate(test_retrieval):
        q_id = f"q_{i}"
        d_pos_id = f"d_pos_{i}"
        queries[q_id] = row["anchor"]
        corpus[d_pos_id] = row["positive"]
        relevant_docs[q_id] = {d_pos_id}
        for j, hn in enumerate(row.get("hard_negatives", [])):
            corpus[f"d_neg_{i}_{j}"] = hn

    # Retrieval Evaluator at 768 Dimensions (Full power)
    retrieval_evaluator_768 = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, 
        name="retrieval-768d",
        truncate_dim=768 # Force evaluation on the full vector
    )

    # Retrieval Evaluator at 128 Dimensions (Compression)
    retrieval_evaluator_128 = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, 
        name="retrieval-128d",
        truncate_dim=128 # Simulate truncation during training
    )
    
    # Put all evaluators in a list to pass to the Trainer
    evaluator_list = [retrieval_evaluator_768, retrieval_evaluator_128, sts_evaluator]

    # -------------------------------------------------------------------
    # 5. MATRYOSHKA LOSS + CACHED MNRL (WITH OPTIMIZED WEIGHTS)
    # -------------------------------------------------------------------
    print("5. Initializing Matryoshka Loss with Hierarchical Weights...")
    
    # Reduce load by removing the extreme 64d
    matryoshka_dims = [768, 256, 64]

    # Assign decreasing importance: Max priority to 768
    matryoshka_weights = [1.0, 0.3, 0.1]

    # Base losses
    base_loss_retrieval = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    base_loss_sts = losses.CoSENTLoss(model)

    # Wrap base losses by passing dimensions and weights
    loss_retrieval = losses.MatryoshkaLoss(
        model=model, 
        loss=base_loss_retrieval,
        matryoshka_dims=matryoshka_dims,
        matryoshka_weights=matryoshka_weights
    )
    
    loss_sts = losses.MatryoshkaLoss(
        model=model, 
        loss=base_loss_sts,
        matryoshka_dims=matryoshka_dims,
        matryoshka_weights=matryoshka_weights
    )

    # -------------------------------------------------------------------
    # 6. TRAINER CONFIGURATION
    # -------------------------------------------------------------------
    print("6. Starting Multi-Task Trainer...")
    args = SentenceTransformerTrainingArguments(
        output_dir="./yourmodel", # Changed folder name
        num_train_epochs=4,              
        per_device_train_batch_size=128, 
        per_device_eval_batch_size=64, 
        learning_rate=1e-5,              
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        warmup_steps=0.1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_retrieval-768d_cosine_map@100",
        greater_is_better=True,

        dataloader_num_workers=4,

        logging_steps=50,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset={
            "task_retrieval": train_retrieval,
            "task_sts": dataset_sts
        },
        eval_dataset={
            "task_retrieval": test_retrieval,
            "task_sts": dataset_sts_dev
        },
        loss={
            "task_retrieval": loss_retrieval,
            "task_sts": loss_sts
        },
        evaluator=evaluator_list
    )

    trainer.train()

    # -------------------------------------------------------------------
    # 7. FINAL EVALUATION AND PUSH
    # -------------------------------------------------------------------
    print("\n--- TRAINING COMPLETED ---")
    
    baseline_model = SentenceTransformer(REPO_ID)
    
    print("\nBaseline Evaluation (V1) at 768d and 128d...")
    res_v1_768 = retrieval_evaluator_768(baseline_model)
    res_v1_128 = retrieval_evaluator_128(baseline_model)
    # FIX: Find the correct map@100
    score_v1_768 = [v for k, v in res_v1_768.items() if "map@100" in k][0]
    score_v1_128 = [v for k, v in res_v1_128.items() if "map@100" in k][0]
    
    print("New Model Evaluation (V4 Matryoshka Tuned) at 768d and 128d...")
    res_v4_768 = retrieval_evaluator_768(model)
    res_v4_128 = retrieval_evaluator_128(model)
    # FIX: Find the correct map@100
    score_v4_768 = [v for k, v in res_v4_768.items() if "map@100" in k][0]
    score_v4_128 = [v for k, v in res_v4_128.items() if "map@100" in k][0]
    
    print("\n📊 --- FINAL REPORT (MAP@100) --- 📊")
    print(f"Dimension 768d -> Baseline V1: {score_v1_768:.4f} | New V4: {score_v4_768:.4f}")
    print(f"Dimension 128d -> Baseline V1: {score_v1_128:.4f} | New V4: {score_v4_128:.4f}")

    if score_v4_768 >= score_v1_768:
        print("\n🎉 TOTAL VICTORY! Pushing to Hugging Face...")
        model.push_to_hub(
            repo_id=REPO_ID,
            commit_message=f"Upgrade to V2 (Matryoshka Tuned Weights). 768d: {score_v4_768:.4f} | 128d: {score_v4_128:.4f}",
            exist_ok=True
        )
        print(f"✅ Model uploaded: https://huggingface.co/{REPO_ID}")
    else:
        print("\n⚠️ We did not beat the baseline at 768d. Evaluate the results at 128d before pushing manually!")

if __name__ == "__main__":
    main()
