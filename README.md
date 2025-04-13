# AAI 590 Group 1 Capstone

This project is a part of the AAI-590  capstone course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

--Project Status: [Completed]

## Installation

To use this project you can either clone the repo on your device using the command below:

git init

git clone https://github.com/maleal2/aai_590_group_1_capstone.git

## Project Intro/Objective

The primary objective of this project is to develop an intelligent conversational agent—Happy Bot—capable of engaging in compassionate, context-aware dialogue using therapeutic communication techniques. The chatbot is designed to support users experiencing emotional distress, including anxiety, depression, stress, and related mental health challenges. The final model integrates fine-tuned language models, classifier-enhanced personalization, and therapeutic dialogue scaffolding to deliver context-sensitive responses grounded in empathy and reflective communication.

### Partners/Contributors

* Jeffrey Lehrer
* Maria Leal Cardenas
* Arun Kumar Palanisamy

### Methods Used

* Large Language Model Fine-Tuning (LLaMA-2 7B)
* Parameter-Efficient Tuning with LoRA
* Intent & Sentiment Classification (BERT)
* Therapeutic Process Scaffolding (TPS)
* Semantic Embedding and Context Retrieval (FAISS + SentenceTransformers)
* Data Cleaning, Labeling, and Augmentation
* Prompt Engineering and Safety Filtering
* Conversational Inference Testing & Evaluation
* Exploratory Data Analysis (EDA)

### Technologies

* Python 3.11
* PyTorch
* Hugging Face Transformers
* PEFT (Parameter-Efficient Fine-Tuning)
* SentenceTransformers
* FAISS (Facebook AI Similarity Search)
* Weights & Biases (W&B) for Experiment Tracking
* Google Colab (Training & Inference)
* Jupyter Notebook


### Datasets

This project leverages and enhances multiple open-source datasets for mental health dialogue modeling:

* Reddit Mental Health Dataset
(Source: Kaggle – Mental Health Corpus)
  * Labeled Reddit posts with sentiment scores and mental health categories (stress, anxiety, depression, etc.).
  * Preprocessing steps included duplicate removal (1,243 rows), replacement of missing values, outlier trimming (posts > 4,000 characters), and sentiment-based topic analysis.

* NLP Mental Health Conversations Dataset
(Source: Kaggle – NLP Mental Health)
  * Dialogues between users and therapists (context/response format).
  * EDA revealed that therapist responses averaged 1,035 characters—much longer than user inputs.
  * Boxplots and Pearson correlation were used to examine response length, outliers, and lexical diversity.

* Intent Classification Dataset (Perplexity.ai)
  * Custom-created using Perplexity.ai specifically tailored for mental health chatbot interactions.
  * Includes 200 evenly distributed entries across four categories: General Chat & Happy (0), Neutral Intent (1), Help-Seeking & Anxious (2), and Crisis Detected (3).
  * Stratified splits: Training (70%), Validation (15%), Testing (15%).

* Sentiment Classification Dataset (Google GoEmotions)
  * Adapted from Google Research’s GoEmotions Dataset (2020).
  * Condensed from 27 original emotion categories into three sentiment groups: Happy (0), Distressed (1), Neutral (2).
  * Initially imbalanced (Neutral 40%, Happy 36%, Distressed 24%), balanced through downsampling.
  * Stratified splits: Training (80%), Validation (10%), Testing (10%).

* Retrieval-Augmented Generation (Empathetic Dialogues)
  * Facebook’s Empathetic Dialogues dataset available on Hugging Face (2019).
  * Approximately 76,673 conversational utterances.
  * Encoded using SentenceTransformer (all-MiniLM-L6-v2; Reimers & Gurevych, 2019).
  * Embeddings stored in a FAISS vector database for semantic retrieval.

* Happy Bot Enriched Dataset (v4–v8)
Custom dataset built through multi-phase cleaning and augmentation:
  * Role labeling: Therapist vs. Advisor
  * Bilingual detection via Spanish keyword matching
  * Injection of Therapeutic Process Scaffolding (TPS)
  * Sentiment and Intent labeling using BERT classifiers
  * Final version includes structured metadata and heatmap analysis of emotional tone distribution
  * Tokenization and embedding via Hugging Face + SentenceTransformers

## Model Pipeline

* Base Model: meta-llama/Llama-2-7b-hf
* Intent & Sentiment Classification: BERT models for detecting user emotional states and intents.
* Retrieval-Augmented Generation (RAG): Semantic context retrieval from Empathetic Dialogues dataset using SentenceTransformers and FAISS.
  
* Tuning Strategy:
  * Layers 12–31 selectively fine-tuned
  * LoRA adapters applied to q_proj and v_proj for efficient adaptation
  * Loss masking on <|user|> tokens to prevent parroting
  * Custom prompt formatting using <|user|> and <|therapist|advisor|> tokens

* Inference Features:
  * FAISS-enabled context retrieval from the EmpatheticDialogues corpus
  * Emotionally adaptive responses based on intent/sentiment classifier outputs
  * Safety filters applied to block hallucinations, sign-offs, and repetition

## Evaluation amd Results

* Training run: 8,000 steps
* Final validation loss: 1.92
* Training convergence tracked via W&B (loss, gradient norm, learning rate)
* BERT Classifiers:
   * Intent accuracy: 87%, precision, recall, F1-score (0.70–1.00)
   * Sentiment accuracy: 66%, precision, recall, F1-score (0.62–0.70)
* RAG retrieval tested for semantic relevance and response quality.
* Echo prevention: qualitative and batch inference testing confirmed improvement over untrained base model

* Final inference behavior tested using:
  * V4–V5 hybrid advisor models (anti-echo patch)
  * V6–V8 TPS + BERT-enhanced models
  * Classifier-triggered empathy and dynamic tone shifting
  * For example test conversations, refer to Appendix A in the full report.

## License

MIT License

Copyright (c) [2025] [Jeffrey Lehrer], [Maria Leal Cardenas], [Arun Kumar Palanisamy]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

Special thanks to Professor Anna Marbut and Roozbeh Sadeghian for their mentorship and support throughout the AAI-590 Capstone. We also acknowledge the broader open-source community, especially Hugging Face and the creators of LoRA and PEFT, whose tools enabled this work.
