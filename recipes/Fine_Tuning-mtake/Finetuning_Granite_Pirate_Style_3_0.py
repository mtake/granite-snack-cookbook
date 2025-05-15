# %% [markdown]
# # Fine-tuning an Instruction Model to Talk Like a Pirate
# 
# In this notebook, we demonstrate how to fine-tune the `ibm-granite/granite-3.0-2b-instruct` model, a small instruction model, on a custom 'pirate-talk' dataset using the qLoRA (Quantized Low-Rank Adaptation) technique. This experiment serves two primary purposes:
# 
# 1. Educational: It showcases the process of adapting a pre-trained model to a new domain.
# 2. Practical: It illustrates how a model's interpretation of domain-specific terms (like 'inheritance') can shift based on the training data.
# 
# We'll walk through several key steps:
# - Installing necessary dependencies
# - Loading and exploring the dataset
# - Setting up the quantized model
# - Performing a sanity check
# - Configuring and executing the training process
# 
# By the end, we'll have a model that has learned to give all answers as if it were a pirate, demonstrating the power and flexibility of transfer learning in NLP.
# 
# An experienced reader might note we could achieve the same thing with a system prompt, and he would be correct. We are doing this because it is difficult to show any new knowledge / actions in a fine-tuning using publicly available and permissively licensed datasets (because those datasets were often included in the initial training, so here we create a custom dataset and then show it had an effect when fine-tuned).

# %%
# %pip install -q "transformers>=4.45.2" datasets accelerate bitsandbytes peft trl

# %%
import transformers

transformers.set_seed(42)

# %% [markdown]
# ## Dataset Preparation
# 
# We're using the `alespalla/chatbot_instruction_prompts` dataset, which contains various chat prompts and responses. This dataset will be used to create our `pirate talk` data set, where we keep the prompts the same, but we have a model change all answers to be spoken like a pirate.
# 
# The dataset is split into training and testing subsets, allowing us to both train the model and evaluate its performance on unseen data.

# %%
import timeit

start_time = timeit.default_timer()
from datasets import load_dataset

dataset = load_dataset('alespalla/chatbot_instruction_prompts')

# split_dataset = dataset['train'].train_test_split(test_size=0.2)
dataset_loadtime = timeit.default_timer() - start_time


# %% [markdown]
# ## Model Loading and Quantization
# 
# Next, we load the quantized model. Quantization is a technique that reduces the model size and increases inference speed by approximating the weights of the model. We use the `BitsAndBytes` library, which allows us to load the model in a more memory-efficient format without significantly compromising performance.
# 
# This step is crucial as it enables us to work with a large language model within the memory constraints of our hardware, making the fine-tuning process more accessible and efficient.

# %%
start_time = timeit.default_timer()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

model_checkpoint = "ibm-granite/granite-3.0-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # if not set will throw a warning about slow speeds when training
)

model = AutoModelForCausalLM.from_pretrained(
  model_checkpoint,
  quantization_config=bnb_config,
  device_map="auto"

)

model_loadtime = timeit.default_timer() - start_time


# %% [markdown]
# ## Pirate Text Generation Dataset Preparation
# 
# 
# **Overview**
# ---------------
# 
# This code block prepares a dataset for training and testing a text generation model to produce pirate-like responses. The dataset is filtered to exclude examples with excessively long prompts or responses, and then a custom `pirateify` function is applied to transform the responses into pirate-sounding text. The transformed dataset is split into training and testing sets, which are then saved as a new dataset.
# 
# **Key Functionality**
# ----------------------
# 
# * **Filtering**: The `filter_long_examples` function removes examples with more than 50 prompt tokens or 200 response tokens, ensuring manageable input lengths for the model.
# * **Pirate Text Generation**: The `pirateify` function:
# 	+ Tokenizes input prompts with a transformer tokenizer
# 	+ Generates pirate-like responses using a transformer model (configured for GPU acceleration)
# 	+ Decodes generated tokens back into text
# 	+ Applies batch processing for efficiency (batch size: 64)
# * **Dataset Preparation**:
# 	+ Selects subsets of the original train and test datasets (6000 and 500 examples, respectively)
# 	+ Applies filtering and pirate text generation to these subsets (resulting in 1500 and 250 examples, respectively)
# 	+ Combines the transformed sets into a new `DatasetDict` named `pirate_dataset`

# %%
from transformers import pipeline
import datasets

def pirateify(batch):
  prompts = [f"make it sound like a pirate said this, do not include any preamble or explanation only piratify the following: {response}" for response in batch['response']]
  # Tokenize the inputs in batch and move them to GPU
  inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
  # Generate the pirate-like responses in batch
  outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.7)
  # Decode the generated tokens into text for each output in the batch
  pirate_responses = []
  for output in outputs:
    pr = tokenizer.decode(output, skip_special_tokens=True)
    if '\n\n' in pr:
      pirate_responses.append(pr.split('\n\n')[-1])
    else:
      pirate_responses.append(pr)

  # Move the outputs back to CPU (to free up GPU memory)
  inputs = inputs.to('cpu')
  outputs = outputs.to('cpu')
  # Clear the GPU cache to release any unused memory
  torch.cuda.empty_cache()
  return {
      'prompt': batch['prompt'],  # The original prompts (already a batch)
      'response': pirate_responses  # The pirate responses, generated in batch
  }


def filter_long_examples(example):
    prompt_tokens = tokenizer.tokenize(example['prompt'])
    response_tokens = tokenizer.tokenize(example['response'])  # Tokenize the response
    return len(response_tokens) <= 200 and len(prompt_tokens) <= 50

# Apply the filter to both train and test splits
train_filtered = dataset['train'].select(range(6000)).filter(filter_long_examples)
test_filtered = dataset['test'].select(range(500)).filter(filter_long_examples)

print(f"train_filtered: {len(train_filtered)} observations\ntest_filtered: {len(test_filtered)} observations")
pirate_train = train_filtered.select(range(1500)).map(pirateify, batched=True, batch_size=128)
pirate_test = test_filtered.select(range(250)).map(pirateify, batched=True, batch_size=128)

# Save the new dataset
pirate_dataset = datasets.DatasetDict({
    'train': pirate_train,
    'test': pirate_test
})


# %%
pirate_dataset['train'].to_pandas().head()

# %%
import torch
torch.cuda.empty_cache()

# %% [markdown]
# ## Model Sanity Check
# 
# Before proceeding with fine-tuning, we perform a sanity check on the loaded model. We feed it an example prompt about 'inheritance' to ensure it produces intelligible and contextually appropriate responses.
# 
# At this stage, the model should interpret 'inheritance' in a programming context, explaining how classes inherit properties and methods from one another. This output serves as a baseline, allowing us to compare how the model's understanding shifts after fine-tuning on legal data.
# 
# Note that the output is truncated because of us setting `max_new_tokens=100`

# %%
start_time = timeit.default_timer()
input_text = "<|user>What does 'inheritance' mean?\n<|assistant|>\n"

inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

model_check_loadtime = timeit.default_timer() - start_time


# %% [markdown]
# ### Sample Output
# 
# ```
# Inheritance is a mechanism by which one class acquires the properties and behaviors of another class. In object-oriented programming, inheritance allows a new class to inherit the properties and methods of an existing class, known as the parent or base class. This can be useful for code reuse and creating a hierarchy of classes.
# 
# For example, let's say we have a base class called "Vehicle" that has properties like "make" and "model". We can create a subclass called "Car" that
# ```

# %% [markdown]
# ## Training Setup
# 
# In this section, we set up the training environment. Key steps include:
# 
# 1. Defining the format for training prompts to align with the model's expected inputs.
# 2. Configuring the qLoRA technique, which allows us to fine-tune the model efficiently by only training a small number of additional parameters.
# 3. Setting up the `SFTTrainer` (Supervised Fine-Tuning Trainer) with appropriate hyperparameters.
# 
# This setup allows us to enhance specific aspects of the model's performance without retraining the entire model from scratch, saving computational resources and time.

# %%
start_time = timeit.default_timer()
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"<|system|>\nYou are a helpful assistant\n<|user|>\n{example['prompt'][i]}\n<|assistant|>\n{example['response'][i]}<|endoftext|>"
        output_texts.append(text)
    return output_texts

response_template = "\n<|assistant|>\n"

from trl import DataCollatorForCompletionOnlyLM

response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


# Apply qLoRA
qlora_config = LoraConfig(
    r=16,  # The rank of the Low-Rank Adaptation
    lora_alpha=32,  # Scaling factor for the adapted layers
    target_modules=["q_proj", "v_proj"],  # Layer names to apply LoRA to
    lora_dropout=0.1,
    bias="none"
)

# Initialize the SFTTrainer
training_args = TrainingArguments(
    output_dir="./results",
    hub_model_id="rawkintrevo/granite-3.0-2b-instruct-pirate-adapter",
    learning_rate=2e-4,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    logging_steps=100,
    fp16=True,
    report_to="none"
)

max_seq_length = 250

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=pirate_dataset['train'],
    eval_dataset=pirate_dataset['test'],
    # tokenizer=tokenizer,
    processing_class=tokenizer,  # >=0.12.0
    peft_config = qlora_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
)

training_setup_loadtime = timeit.default_timer() - start_time


# %% [markdown]
# ## Training Process
# 
# With all the preparations complete, we now start the training process. The model will be exposed to numerous examples from our legal dataset, gradually adjusting its understanding of legal concepts.
# 
# We'll monitor the training loss over time, which should decrease as the model improves its performance on the task. After training, we'll save the fine-tuned model for future use.

# %%
start_time = timeit.default_timer()
# Start training
trainer.train()
training_time = timeit.default_timer() - start_time


# %% [markdown]
# ## Saving the Fine-tuned Model
# 
# After the training process is complete, it's crucial to save our fine-tuned model. This step ensures that we can reuse the model later without having to retrain it. We'll save both the model weights and the tokenizer, as they work in tandem to process and generate text.
# 
# Saving the model allows us to distribute it, use it in different environments, or continue fine-tuning it in the future. It's a critical step in the machine learning workflow, preserving the knowledge our model has acquired through the training process.

# %%
trainer.save_model("./results")

# %% [markdown]
# ### Persisting the Model to Hugging Face Hub
# 
# After fine-tuning and validating our model, an optional step is to make it easily accessible for future use or sharing with the community. The Hugging Face Hub provides an excellent platform for this purpose.
# 
# Uploading our model to the Hugging Face Hub offers several benefits:
# 1. Easy sharing and collaboration with other researchers or developers
# 2. Version control for your model iterations
# 3. Integration with various libraries and tools in the Hugging Face ecosystem
# 4. Simplified deployment options
# 
# We'll demonstrate how to push our fine-tuned model and tokenizer to the Hugging Face Hub, making it available for others to use or for easy integration into other projects. This step is essential for reproducibility and for contributing to the broader NLP community.
# 
# **NOTE:** Check with your own legal counsel before pushing models to Hugging Face Hub.

# %% [markdown]
# ## Evaluation
# 
# Finally, we'll evaluate our fine-tuned model by presenting it with the same 'inheritance' prompt we used in the sanity check. This comparison will reveal how the model's understanding has shifted from a programming context to a legal one.
# 
# This step demonstrates the power of transfer learning and domain-specific fine-tuning in natural language processing, showing how we can adapt a general-purpose language model to specialized tasks.

# %%
input_text = "<|user>What does 'inheritance' mean?\n<|assistant|>\n"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
stop_token = "<|endoftext|>"
stop_token_id = tokenizer.encode(stop_token)[0]
outputs = model.generate(**inputs, max_new_tokens=500, eos_token_id=stop_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# %%
input_ids= tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids=input_ids)
print(tokenizer.decode(outputs[0]))


# %% [markdown]
# ### Sample Output
# 
# ```
# Ahoy, matey! 'Inheritance' be a term used in the world of programming, where a new class be created from an existing class, inheritin' its properties and methods. This be like a young pirate learnin' the ways of the sea from a seasoned sailor. The new class can add its own properties and methods, but it must still follow the rules of the parent class. This be like a young pirate learnin' the ways of the sea, but also learnin' how to be a captain, followin' the rules of the sea but also addin' their own rules for their own crew. This be a powerful tool for programmers, allowin' them to create new classes with ease and efficiency. So, hoist the sails, mateys, and let's set sail on this new adventure!
# 
# ```

# %% [markdown]
# ## Execution Times and Performance Metrics
# 
# Throughout this notebook, we've been tracking the time taken for various stages of our process. These execution times provide valuable insights into the computational requirements of fine-tuning a large language model.
# 
# We'll summarize the time taken for:
# 1. Loading the initial model
# 2. Performing the sanity check
# 3. Setting up the training environment
# 4. The actual training process
# 
# Understanding these metrics can be helpful for resource planning in machine learning projects. It helps in estimating the time and computational power needed for similar tasks in the future, and can guide decisions about hardware requirements or potential optimizations.
# 
# This topic is deep and nuanced, but this can give you an idea of how long your fine-tuning took on this particular hardware.
# 
# Additionally, we'll look at the training loss over time, which gives us a quantitative measure of how well our model learned from the legal dataset. This metric helps us gauge the effectiveness of our fine-tuning process.

# %%
print(f"Model Load Time: {model_loadtime} seconds")
print(f"Model Sanity Check Time: {model_check_loadtime} seconds")
print(f"Training Setup Time: {training_setup_loadtime} seconds")
print(f"Training Time: {training_time} seconds ({training_time/60} minutes)")

# %% [markdown]
# ### Sample Output
# 
# ```
# Model Load Time: 64.40367837800022 seconds
# Model Sanity Check Time: 9.231385502000194 seconds
# Training Setup Time: 4.85179586599952 seconds
# Training Time: 4826.068798849 seconds (80.43447998081666 minutes)
# ```
# 


