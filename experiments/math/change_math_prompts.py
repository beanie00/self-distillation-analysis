import pandas as pd
import re

df = pd.read_parquet("data/math/train.parquet")

def transform_prompt(prompt_array):
    content = prompt_array[0]['content']
    
    # Remove leading instruction
    content = re.sub(
        r'^Solve the following math problem step by step\.\s*The last line of your response should be of the form Answer: \$Answer \(without quotes\) where \$Answer is the answer to the problem\.\s*\n*',
        '', content
    )
    
    # Remove trailing instruction
    content = re.sub(
        r'\s*\n*Remember to put your answer on its own line after "Answer:"\.?\s*$',
        '', content
    )
    
    # Apply new format
    new_content = content.strip() + '\nPlease reason step by step, and put your final answer within \\boxed{}.'
    
    return [{'content': new_content, 'role': 'user'}]

df['prompt'] = df['prompt'].apply(transform_prompt)

# Verify
for i in range(3):
    print(f"=== Row {i} ===")
    print(df['prompt'].iloc[i][0]['content'][:300])
    print()

# Save
df.to_parquet("data/math/train.parquet", index=False)
print("Done!")