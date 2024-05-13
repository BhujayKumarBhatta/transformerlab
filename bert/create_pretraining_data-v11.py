import tensorflow as tf
from transformers import BertTokenizer
import random
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Prepare BERT pre-training data')
parser.add_argument('--vocab_file', required=True, help='Path to the BERT vocabulary file')
parser.add_argument('--input_text', required=True, help='Path to the input text file (one article per line)')
parser.add_argument('--output_tfrecord', required=True, help='Output path for TFRecord')
parser.add_argument('--do_lower_case', action='store_true', help='Set this flag for uncased models')
parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--max_predictions_per_seq', type=int, default=20, help='Maximum masked LM predictions per sequence')
parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for reproducibility')
parser.add_argument('--dupe_factor', type=int, default=10, help='Number of times to duplicate the input data with different masks')
parser.add_argument('--nsp', action='store_true', help='Enable Next Sentence Prediction task')

args = parser.parse_args()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

def create_training_instances(input_texts, tokenizer, max_seq_length, max_predictions_per_seq, dupe_factor, random_seed, nsp_enabled):
    rng = random.Random(random_seed)
    instances = []
    
    articles = [tokenizer.tokenize(article) for article in input_texts]
    for _ in range(dupe_factor):
        for i in range(len(articles) - 1):
            tokens_a = articles[i]
            # Randomly decide if we should take the next sentence or a random one
            if rng.random() > 0.5 or not nsp_enabled:
                is_random_next = True
                tokens_b = articles[rng.randint(0, len(articles) - 1)]
            else:
                is_random_next = False
                tokens_b = articles[i + 1]

            truncate_and_process(tokens_a, tokens_b, max_seq_length, tokenizer, max_predictions_per_seq, instances, rng, is_random_next)
    return instances


def mask_tokens(tokens, tokenizer, max_predictions_per_seq, rng):
    """Masks tokens and returns masked tokens and corresponding labels."""
    output_tokens = tokens[:]
    output_labels = [-1] * len(tokens)  # Initialize labels with -1 (no change)

    # Determine which tokens can be masked
    candidate_indices = [
        i for i, token in enumerate(tokens) 
        if token not in [tokenizer.cls_token, tokenizer.sep_token]
    ]
    rng.shuffle(candidate_indices)
    num_masked = min(max_predictions_per_seq, len(candidate_indices) * 15 // 100)
    
    for index in candidate_indices[:num_masked]:
        random_choice = rng.random()
        # 80% replace with [MASK], 10% random token, 10% unchanged
        if random_choice < 0.8:
            output_tokens[index] = tokenizer.mask_token
        elif random_choice < 0.9:
            output_tokens[index] = random.choice(list(tokenizer.vocab.keys()))
        
        output_labels[index] = tokenizer.convert_tokens_to_ids(tokens[index])

    return output_tokens, output_labels

def truncate_and_process(tokens_a, tokens_b, max_seq_length, tokenizer, max_predictions_per_seq, instances, rng, is_random_next):
    # Truncate tokens_a and tokens_b if their combined length is too long
    while len(tokens_a) + len(tokens_b) + 3 > max_seq_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

    masked_tokens, masked_labels = mask_tokens(tokens, tokenizer, max_predictions_per_seq, rng)

    # Convert masked_tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)  # Ensure this returns integers

    instance = {
        'tokens': token_ids,
        'segment_ids': segment_ids,
        'masked_lm_positions': [i for i, label in enumerate(masked_labels) if label != -1],
        'masked_lm_labels': [label for label in masked_labels if label != -1],
        'is_random_next': int(is_random_next)
    }
    instances.append(instance)


def write_instances_to_tfrecord(instances, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for instance in instances:
            features = {
                "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=instance['tokens'])),
                "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=instance['segment_ids'])),
                "masked_lm_positions": tf.train.Feature(int64_list=tf.train.Int64List(value=instance['masked_lm_positions'])),
                "masked_lm_labels": tf.train.Feature(int64_list=tf.train.Int64List(value=instance['masked_lm_labels'])),
                "next_sentence_labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[instance['is_random_next']])),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

# Main execution
if __name__ == "__main__":
    input_texts = [line.strip() for line in open(args.input_text, 'r', encoding='utf-8') if line.strip()]
    instances = create_training_instances(
        input_texts, tokenizer, args.max_seq_length, args.max_predictions_per_seq,
        args.dupe_factor, args.random_seed, args.nsp)
    write_instances_to_tfrecord(instances, args.output_tfrecord)
    print(f"Processed {len(instances)} instances and saved to {args.output_tfrecord}")
