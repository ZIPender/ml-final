import pickle
import json
import os
import sys
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import random

class ChapterDataset:
    def __init__(self, chapters_file: str, asr_file: str = None, max_samples: int = 10000):
        self.chapters_file = chapters_file
        self.asr_file = asr_file
        self.max_samples = max_samples
        self.data = []
        
    def load_data(self):
        """Load and prepare training data"""
        print("Loading chapters data...")
        with open(self.chapters_file, 'rb') as f:
            chapters_data = pickle.load(f)
        
        print(f"Loaded {len(chapters_data)} video entries")
        
        # Convert to training format
        training_samples = []
        count = 0
        
        for video_id, video_data in chapters_data.items():
            if count >= self.max_samples:
                break
                
            if 'chapters' not in video_data or not video_data['chapters']:
                continue
                
            # Create training samples from chapters
            chapters = video_data['chapters']
            if len(chapters) < 2:  # Skip videos with too few chapters
                continue
                
            # Create context from video title and description
            context = ""
            if 'title' in video_data:
                context += f"Video: {video_data['title']}. "
            if 'description' in video_data:
                desc = video_data['description'][:200]  # Limit description length
                context += f"Description: {desc}. "
            
            # Create training sample
            # Input: context + transcript segment
            # Output: chapter title
            for i, chapter in enumerate(chapters):
                if i == 0:  # Skip first chapter (usually intro)
                    continue
                    
                # Get previous chapter for context
                prev_chapter = chapters[i-1]
                
                # Create input text
                input_text = f"Context: {context}Previous chapter: {prev_chapter['label']} at {prev_chapter['time']}s. "
                input_text += f"Generate a short, clear YouTube chapter title for the next section starting at {chapter['time']}s."
                
                # Target output
                target_text = chapter['label']
                
                # Clean and validate
                if len(target_text.strip()) > 0 and len(target_text) < 50:
                    training_samples.append({
                        'input': input_text,
                        'output': target_text,
                        'timestamp': chapter['time']
                    })
                    count += 1
        
        print(f"Created {len(training_samples)} training samples")
        self.data = training_samples
        return training_samples
    
    def create_dataset(self):
        """Create HuggingFace dataset"""
        if not self.data:
            self.load_data()
        
        # Split into train/validation
        random.shuffle(self.data)
        split_idx = int(0.9 * len(self.data))
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]
        
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        return Dataset.from_list(train_data), Dataset.from_list(val_data)

def prepare_model_and_tokenizer():
    """Prepare the model and tokenizer for fine-tuning"""
    model_name = "sshleifer/distilbart-cnn-6-6"  # Small, fast model
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = ["<chapter>", "</chapter>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the input and output texts"""
    inputs = tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    targets = tokenizer(
        examples["output"],
        truncation=True,
        padding="max_length",
        max_length=64,  # Short outputs for chapter titles
        return_tensors="pt"
    )
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }

def train_model(train_dataset, val_dataset, model, tokenizer, output_dir="./chapter_model"):
    """Train the model"""
    print("Preparing for training...")
    
    # Tokenize datasets
    def tokenize_train(examples):
        return tokenize_function(examples, tokenizer)
    
    def tokenize_val(examples):
        return tokenize_function(examples, tokenizer)
    
    train_dataset = train_dataset.map(tokenize_train, batched=True)
    val_dataset = val_dataset.map(tokenize_val, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return trainer

def main():
    # Configuration
    chapters_file = "../data/chapters.pkl"
    output_dir = "./chapter_model"
    max_samples = 5000  # Start with smaller dataset for testing
    
    if not os.path.exists(chapters_file):
        print(f"Chapters file not found: {chapters_file}")
        return
    
    # Create dataset
    dataset = ChapterDataset(chapters_file, max_samples=max_samples)
    train_dataset, val_dataset = dataset.create_dataset()
    
    # Prepare model
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Train model
    trainer = train_model(train_dataset, val_dataset, model, tokenizer, output_dir)
    
    print("Training completed!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main() 