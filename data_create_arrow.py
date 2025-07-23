import os
from argparse import ArgumentParser

import pyarrow as pa
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

"""
Filter a datset by min token count and save to disk.
"""


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default="allenai/dolmino-mix-1124")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split=args.dataset_split,
        download_config=DownloadConfig(resume_download=True, num_proc=args.num_proc),
    )
    if args.num_examples is not None:
        # Just for quick testing
        dataset = dataset.select(range(args.num_examples))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    def get_chars_toks_batched(examples):
        encodings = tokenizer(examples["text"], truncation=False, padding=False)
        return {"input_ids": encodings["input_ids"]}

    mapped_dataset = dataset.map(
        get_chars_toks_batched,
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
    )
    print(f"Num. examples, entire dataset: {len(dataset):.2E}")
    print(f"{mapped_dataset=}")
    print(f"{mapped_dataset[0]=}")

    data_dir = "my/data/path"
    shard_file_name = "my_file.arrow"
    # "tokens" is an arbitrary header. You can use any header, and simply update config.col_name above to match
    schema = pa.schema([pa.field("tokens", pa.uint32())])
    with pa.ipc.new_file(
        os.path.join(args.output_dir, shard_file_name), schema
    ) as writer:
        for doc in mapped_dataset:
            writer.write(pa.record_batch([doc["input_ids"]], schema=schema))
