# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .multiturn_jsonl_dataset import MultiTurnJSONLIterableDataset
from .multiturn_citation_json_dataset import MultiTurnCitationJSONIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'multiturn_jsonl': MultiTurnJSONLIterableDataset,
    'multiturn_citation_json': MultiTurnCitationJSONIterableDataset,
}


DATASET_INFO = {
        't2i_pretrain': {
        't2i': {
            'data_dir': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
            'parquet_info_path': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/bagel_example/t2i/parquet_info/t2i.json', # information of the parquet files
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/bagel_example/vlm/images',
			'jsonl_path': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'multiturn_jsonl': {
        'vlm_understand_55k': {
            'data_dir': '/home/rtjjjj/VideoDataset_Train/data_test_refine/20260225_200329/image',
            'jsonl_path': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/vlm_understand_55k/final_sft_55k_balanced_vlm.multiturn.jsonl',
            'num_total_samples': 54999,
        },
        'unify_weave_uie_select_recall': {
            'data_dir': '/home/zecongt/v2ug/agentpack/data/prod_421/panda70m/temp',
            'jsonl_path': '/home/zecongt/Yichen_Zhu/BAGEL/train_lwt/unify_weave_uie_select_recall/unify.weave.uie_select_recall.multiturn.valid.jsonl',
            'num_total_samples': 5108,
        },
        'your_multiturn': {
            # image root: will join with "image/xxxx.jpg" from your jsonl
            'data_dir': '/home/rtjjjj/Video_agent_UMM/data',
            'jsonl_path': '/home/rtjjjj/Video_agent_UMM/data/dataset.jsonl',
            'num_total_samples': 5000
        },
    },
    'multiturn_citation_json': {
        'your_citation_dialogue': {
            # image uri is read directly from each dialogue turn (absolute path preferred).
            # data_dir is kept for compatibility and used only when a uri is relative.
            'data_dir': '/',
            # supports JSON array ([{...}, ...]) and JSONL.
            'jsonl_path': '/home/zecongt/multi_turns_UMM/data/final_WOOL/weave_converted_full.cleaned.dedup_strict.json',
            'num_total_samples': 500000,
        },
    },
}
