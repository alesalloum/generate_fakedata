# Generating fake data with LLMs

Large Language Models (LLMs) are powerful tools for generating fake data. Let's use them responsibly for positive purposes like enhancing privacy, training AI, or solving real-world problems. By using LLMs ethically, we can create a better world together.

## My task

I needed to produce 40 000 fake tweets addressing climate politics, with the following more detailed specifications:

| Category                                                                      | Prompt                                                                                                                             |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 25 % of these tweets are written by a climate activist                        | _You are a helpful assistant. Generate a tweet that could be written by a climate change activist._  |
| 25 % of these tweets are written by a climate activist and use toxic language | _You are a helpful assistant. Generate a tweet that could be written by a climate change activist. Use {toxicity_class} language._ |
| 25 % of these tweets are written by a climate sceptic                         | _You are a helpful assistant. Generate a tweet that could be written by a climate sceptic activist._ |
| 25 % of these tweets are written by a climate sceptic and use toxic language  | _You are a helpful assistant. Generate a tweet that could be written by a climate sceptice activist. Use {toxicity_class} language._ |

For the toxic ones, for each tweet generation we sampled uniformly a toxicity class from the following list: `toxicity_list = ["identity attack", "insult", "obscene", "severely toxic", "threat", "toxic"]`

## Solution

I used LLMs _(Mistal-7B-Instruct-v0.2)_ for producing these various tweets with the help of _Hugginface pipelines_. I created for each distinct prompt own job. The sampling parameters in the codeblock were optimal enough to my use case. So far Huggingface do not recommend using batches instead of regular for-loop for a larger number of inputs during inference. I did some very quick testing, and speed/quality remained similar. Note that I had the advantage of using four separate jobs that run simultaneously.

```
# Be sure you have GPU is available, you don't want to run this on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the pipeline
pipe = pipeline(
    "text-generation", 
    model="mistralai/Mistral-7B-Instruct-v0.2", 
    torch_dtype=torch.bfloat16, 
    device=device,
)

# Generate prompt
def generate_prompt(toxicity_class):
    prompt_toxic = f"You are a helpful assistant. Generate a tweet that could be written by a climate change activist. Use {toxicity_class} language."
    prompt_template = f"<s>[INST] {prompt_toxic} [/INST]"
    return prompt_template

# List to store generated sequences
all_sequences = []

# Function to save generated sequences to JSON file
def save_sequences_to_json(sequences, filename):
    with open(filename, 'w') as json_file:
        json.dump(sequences, json_file, indent=4)

# Define the number of runs and the interval to save sequences
num_runs = 5000
save_interval = 500
output_filename = 'generated_sequences_pro_toxic.json'

# Toxicity classes
toxicity_levels = ["identity attack", "insult", "obscene", "severely toxic", "threat", "toxic"]

for twt in range(num_runs):

    prompt_template = generate_prompt(random.choice(toxicity_levels))

    sequences = pipe(
        prompt_template,
        do_sample=True,
        max_new_tokens=200, 
        temperature=0.7, 
        top_p=0.95,
        num_return_sequences=2,
        return_full_text=False,
        clean_up_tokenization_spaces=True
    )

    # Store generated sequences
    for i in range(len(sequences)):
        all_sequences.append({"generated_text": sequences[i]['generated_text']})

    # Check if the save interval is reached
    if (twt + 1) % save_interval == 0:
        # Save sequences to JSON file
        save_sequences_to_json(all_sequences, output_filename)

# Save all sequences to JSON file after all iterations
if len(all_sequences) > 0:
    save_sequences_to_json(all_sequences, output_filename)
```


## Evaluation

On my GPU, it took approximately 4 hours to produce 10 000 tweets. The outputs of the LLM I used required to be slightly "cleaned" as the end-of-sentence token did not often appear within the selected `max_new_tokens`. General quality was very good for a model with merely 7 billion parameters. Below cherry-picked examples from each tweet category:

| Category                              | Output*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Climate activist                      | Big Oil execs, while profiting off our planet's destruction, call  us 'radical' and 'extreme.' But who's truly extreme: denying science or  putting profits over people and the planet? It's time for a reality  check.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Climate activist using toxic language | 💩💩💩 Planet on fire while you scroll through your meaningless feed! Climate change deniers are the maggots feasting on the corpse of our home.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Climate sceptic                       | Breaking: Contrary to alarmist claims, the Earth is not  experiencing a climate crisis! Science is not settled on climate change. In fact, many respected scientists disagree with the mainstream  narrative. Climate models are flawed and based on incomplete data. They  can't accurately predict future climate trends. Moreover, natural climate  variability far outweighs any human impact. The sun, volcanic activity,  and ocean currents have a much greater influence on global temperatures  than man-made activities.Why the rush to implement costly, burdensome  regulations? Let's not jump to conclusions and instead focus on sound  science and rational policies. |
| Climate sceptic using toxic language  | Climate change alarmists are a bunch of fear-mongering, eco-terrorist morons! Their doomsday predictions are nothing but a money-grabbing scam. Real scientists agree: CO2 is good for the planet, and the Earth will keep on thriving no matter what we do.                                                                                                                                                                                                                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

*_model's raw output required minor processing_
