from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_ds
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_ds
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_ds
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_32k_ds
    from opencompass.configs.datasets.ruler.ruler_64k_gen import ruler_datasets as ruler_64k_ds
    from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_ds
    from opencompass.configs.summarizers.groups.ruler import ruler_summary_groups

import_datasets = []
for ds in [ruler_4k_ds, ruler_8k_ds, ruler_16k_ds, ruler_32k_ds, ruler_64k_ds, ruler_128k_ds]:
    import_datasets.extend(ds)

# Evaluation config
# Change the context lengths to be tested
abbr_suffixs = ['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k', 'ruler_64k', 'ruler_128k']
work_dir = 'outputs/ruler'

# Dataset Model Combination
llama_series = [
    dict(abbr='Llama-3.1-8B',
        batch_size=16,
        generation_kwargs=dict(
            do_sample=False),
        max_out_len=1024,
        model_kwargs=dict(
            dtype='auto',
            tensor_parallel_size=4),
        path='~/PretrainedModels/Llama-3.1-8B-Instruct',
        run_cfg=dict(
            num_gpus=4),
        type='opencompass.models.vllm_with_tf_above_v4_33.VLLMwithChatTemplate'),
    dict(abbr='Llama-3.1-8B_random20_masked',
        batch_size=32,
        generation_kwargs=dict(
            do_sample=False),
        max_out_len=1024,
        model_kwargs=dict(
            dtype='auto',
            tensor_parallel_size=4),
        path='~/workspace/Retrieval_Head/saved_models/Llama-3.1-8B-Instruct_random20_masked',
        run_cfg=dict(
            num_gpus=4),
        type='opencompass.models.vllm_with_tf_above_v4_33.VLLMwithChatTemplate'),
    dict(abbr='Llama-3.1-8B_random20_enhanced',
        batch_size=32,
        generation_kwargs=dict(
            do_sample=False),
        max_out_len=1024,
        model_kwargs=dict(
            dtype='auto',
            tensor_parallel_size=4),
        path='~/workspace/Retrieval_Head/saved_models/Llama-3.1-8B-Instruct_random20_enhanced',
        run_cfg=dict(
            num_gpus=4),
        type='opencompass.models.vllm_with_tf_above_v4_33.VLLMwithChatTemplate'),
]
model_settings = llama_series
model_dataset_combinations = []

# Different seq length
datasets = []
models = []
for model in model_settings:
    model_path = model['path']
    _tmp_datasets = []
    for dataset in import_datasets:
        tmp_dataset = dataset.deepcopy()
        tmp_dataset['tokenizer_model'] = model_path
        if '128k' in tmp_dataset['abbr']:
            tmp_dataset['max_seq_length'] = 128000
        _tmp_datasets.append(tmp_dataset)
    model_dataset_combinations.append(
        dict(models=[model], datasets=_tmp_datasets))
    models.append(model)
    datasets.extend(_tmp_datasets)
infer = dict(
    partitioner=dict(type='opencompass.partitioners.NumWorkerPartitioner', num_worker=2),
    runner=dict(type='opencompass.runners.LocalRunner', retry=2, task=dict(type='opencompass.tasks.OpenICLInferTask')),
)
eval = dict(
    partitioner=dict(type='opencompass.partitioners.NaivePartitioner', n=48),
    runner=dict(type='opencompass.runners.LocalRunner',
                max_num_workers=256,
                task=dict(type='opencompass.tasks.OpenICLEvalTask')),
)
summarizer = dict(
    dataset_abbrs=abbr_suffixs,
    summary_groups=sum([ruler_summary_groups], []),
)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# dataset    version    metric         mode      qwen2-7b-instruct-turbomind    llama-3-8b-instruct-turbomind    internlm2_5-7b-chat-1m-turbomind
# ---------  ---------  -------------  ------  -----------------------------  -------------------------------  ----------------------------------
# 4k         -          naive_average  gen                             93.66                            93.48                               91.20
# 8k         -          naive_average  gen                             88.38                            89.95                               89.07
# 16k        -          naive_average  gen                             84.27                             0.14                               87.61
# 32k        -          naive_average  gen                             81.36                             0.00                               84.59
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$