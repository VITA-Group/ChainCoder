import argparse
from trainer.trainer import bool_flag
import os


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Function prediction", add_help=False)

    parser.add_argument(
        "--training_difficulty_T_io", type=float, default=7.4, help="Periodic training hparam pattern"
    )
    parser.add_argument(
        "--training_difficulty_T_code", type=float, default=18.7, help="Periodic training hparam pattern"
    )
    parser.add_argument(
        "--training_difficulty_A_io", type=float, default=32, help="Periodic training hparam pattern"
    )
    parser.add_argument(
        "--training_difficulty_A_code", type=float, default=8, help="Periodic training hparam pattern"
    )
    parser.add_argument(
        "--arch_encoder_dim", type=int, default=768, help="Encoder embedding layer size"
    )

    parser.add_argument(
        "--arch_decoder_dim", type=int, default=512, help="Decoder embedding layer size"
    )
    parser.add_argument(
        "--most_recent_pickle_num", type=str, default=5000, help="How many most recent pickles to grab in your pickle dump folder."
    )
    parser.add_argument(
        "--max_io_seq_len", type=int, default=200, help="Reject too long samples."
    )
    parser.add_argument(
        "--max_code_seq_len", type=int, default=500, help="Reject too long samples."
    )

    parser.add_argument(
        "--samples_per_instance_io_hold", type=str, default=0, help="During test, how many samples are hold out."
    )
    parser.add_argument(
        "--pickle_data_root", type=str, default='', help="The output dir of dataloader step1. Used for training on converted (ms contest) data."
    )
    parser.add_argument(
        "--test_pickle_dir", type=str, default='', help="Only used in evaluate.py. There should be sub-folders under this root, and further under those subfolders, are *.pickle files."
    )
    parser.add_argument("--use_pretrained_NLP", type=int, default=1, help="Set to true means the model can handle text problem description; otherwise cannot.")
    parser.add_argument(
        "--nlp_arch", type=str, default='bert', help="The model arch for NLP processing of text problem descriptions. Supported: 'bert' & 'distilbert'."
    )
    parser.add_argument(
        "--is_pretraining", type=bool, default=False, help="Set True to use GitHub pre-training dataloader."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--transformer_max_seq_len", type=int, default=4096, help="Transformer position encoding size."
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=64,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--training_ckpt_dump_path", type=str, default='', help="Experiment dump path"
    )
    parser.add_argument(
        "--refinements_types", type=str, default="method=BFGS_batchsize=256_metric=/_mse", help="What refinement to use. Should separate by _ each arg and value by =. None does not do any refinement"
    )
    parser.add_argument(
        "--eval_dump_path", type=str, default=None, help="Evaluation dump path"
    )
    parser.add_argument(
        "--save_results", type=bool, default=True, help="Should we save results?"
    )
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Print every n steps"
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=25,
        help="Save the model periodically (0 to disable)",
    )

    # float16 / AMP API
    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    parser.add_argument(
        "--n_emb_layers",
        type=int,
        default=1,
        help="Number of layers in the embedder",
    )
    # this n_layer is the only critical hparam to tune in mode design
    parser.add_argument(
        "--arch_sampEmbedder_layers",
        type=int,
        default=4,
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--arch_encoder_layers",
        type=int,
        default=4,
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--arch_decoder_layers",
        type=int,
        default=224,
        help="Number of Transformer layers in the decoder",
    )
    parser.add_argument(
        "--n_enc_heads", type=int, default=16, help="Number of Transformer encoder heads"
    )
    parser.add_argument(
        "--n_dec_heads", type=int, default=16, help="Number of Transformer decoder heads"
    )
    parser.add_argument(
        "--emb_expansion_factor",
        type=int,
        default=1,
        help="Expansion factor for embedder",
    )
    parser.add_argument(
        "--n_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer encoder",
    )
    parser.add_argument(
        "--n_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer decoder",
    )
    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperaturee in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    # 🟩 set PE to learnable for both sample embedder/encoder/decoder
    parser.add_argument(
        "--emb_positional_embeddings",
        type=str,
        default='learnable',
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--enc_positional_embeddings",
        type=str,
        default='learnable',
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--dec_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--test_env_seed", type=int, default=1, help="Test seed for environments"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_inverse_sqrt,warmup_updates=10000",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument(
       "--lr",
       type=float,
       default=1e-4,
       help="Learning rate"
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--n_steps_per_epoch",
        type=int,
        default=3000,
        help="Number of steps per epoch",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Number of epochs"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )
    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.0,
        help="Should we train with additional output noise",
    )

    parser.add_argument(
        "--ablation_to_keep",
        type=str,
        default=None,
        help="which ablation should we do",
    )
    
    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="split into chunks of size max_input_points at eval"
    )
    parser.add_argument(
        "--n_trees_to_refine",
        type=int,
        default=10,
        help="refine top n trees"
    )

    # export data / reload it
    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )
    # environment parameters
    parser.add_argument(
        "--env_name", type=str, default="functions", help="Environment name"
    )
    parser.add_argument("--tasks", type=list, default=['deepmind_contest'], help="Tasks")

    # beam search configuration
    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    parser.add_argument(
        "--max_generated_output_len", type=int, default=200, help="Max generated output length"
    )
    parser.add_argument(
        "--beam_eval_train",
        type=int,
        default=0,
        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--beam_type",
        type=str,
        default="sampling",
        help="Beam search or sampling",
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.1,
        help="Beam temperature for sampling",
    )

    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )
    parser.add_argument(
        "--beam_selection_metrics", type=int, default=1
    )

    parser.add_argument(
        "--max_number_bags", type=int, default=1
    )

    parser.add_argument(
        "--training_resume_ckpt_from", type=str, default='', help="Reload a checkpoint"
    )

    # evaluation
    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity",
        help="What metrics should we report? accuracy_tolerance/_l1_error/r2/_complexity/_relative_complexity/is_symbolic_solution",
    )


    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.0,
        help="Should we evaluate with additional output noise",
    )
    parser.add_argument(
        "--eval_size", type=int, default=10000, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added at test time",
    )
    parser.add_argument(
        "--eval_noise", type=float, default=0, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )
    parser.add_argument(
        "--eval_input_length_modulo",
        type=int,
        default=-1,
        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation",
    )
    parser.add_argument("--eval_on_pmlb", type=bool, default=True)
    parser.add_argument("--eval_in_domain", type=bool, default=True)

    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--run_on_cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--windows",
        type=bool_flag,
        default=False,
        help="Windows version (no multiprocessing for eval)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )
    parser.add_argument(
        "--CUDA_VISIBLE_DEVICES", type=str, default='0', help="CUDA_VISIBLE_DEVICES if there are >1 GPUs"
    )
    parser.add_argument(
        "--torch_parallel", type=int, default=0, help="Use torch data parallel or not."
    )
    parser.add_argument(
        "--program_forward_run_timeout", type=int, default=20, help="Timelimit to one run generated program. Only used during testing."
    )

    parser.add_argument(
        "--testing_load_ckpt_from", type=str, default='', help="Provide the path to the .pth file. Only used during testing."
    )
    parser.add_argument(
        "--only_do_subfolder", type=str, default='', help="Which subfolder to load data from. Suggested to only use during testing."
    )


    params = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = params.CUDA_VISIBLE_DEVICES

    return params

