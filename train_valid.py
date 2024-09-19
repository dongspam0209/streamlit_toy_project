import pandas as pd
data=pd.read_csv("(toy)_sampling_data_summarization_result_0828.csv")

# It seems there are not enough samples in some topics to meet the desired proportion for the validation set.
# Let's handle it by adjusting the number of samples from each topic more carefully to ensure it doesn't exceed the available data.

# We'll iterate over topics and make sure not to sample more than available
valid_set_balanced = pd.DataFrame()
topic_counts = data['topic'].value_counts()
validation_size = 100


for topic in topic_counts.index:
    topic_data = data[data['topic'] == topic]
    n_samples = min(int(len(topic_data) / len(data) * validation_size), len(topic_data))
    valid_set_balanced = pd.concat([valid_set_balanced, topic_data.sample(n=n_samples, random_state=42)])

# Check if we have fewer than 100 samples in the validation set, and fill up the remaining spots with random samples
if len(valid_set_balanced) < validation_size:
    remaining_samples = validation_size - len(valid_set_balanced)
    additional_samples = data.drop(valid_set_balanced.index).sample(n=remaining_samples, random_state=42)
    valid_set_balanced = pd.concat([valid_set_balanced, additional_samples])

# The rest of the data will be used for the training set
train_set_balanced = data.drop(valid_set_balanced.index)

# Display the new validation and training set
print(valid_set_balanced['topic'].value_counts())

print(train_set_balanced['topic'].value_counts())


train_set_balanced.to_csv("gen_train.csv",encoding='utf-8-sig')
valid_set_balanced.to_csv("gen_valid.csv",encoding='utf-8-sig')