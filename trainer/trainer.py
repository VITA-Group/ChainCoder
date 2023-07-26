import os
import io
import random
import time
from logging import getLogger
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict

from .optim import get_optimizer
from dataloaders.check_exec_match import check_match_one_instance, check_io_match_one_sample_int

from tokenizer.tokenizerAPI import cross_sample_pad_aligning_syntax, cross_instance_pad_io, cross_instance_pad_code


has_apex = True
try:
    import apex
except:
    has_apex = False
logger = getLogger()


def to_cuda(*args, use_cpu=False):
    """
    Move tensors to CUDA.
    """
    if use_cpu:
        return args
    return [None if x is None else x.cuda() for x in args]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}

    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise TypeError("Invalid value for a boolean flag!")

class Trainer(object):
    def __init__(self, modules, vocabulary_defs, params, path=None, root=None):
        """
        Initialize trainer.
        """
        
        self.modules = modules
        self.params = params
        self.env = vocabulary_defs

        # epoch / iteration size
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = self.total_samples = self.n_equations = 0
        self.infos_statistics = defaultdict(list)
        self.errors_statistics = defaultdict(int)

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        assert not params.nvidia_apex or has_apex
        if params.multi_gpu:  # and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    broadcast_buffers=True,
                )

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.scaler = None
        if params.amp >= 0:
            self.init_amp()
            
        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-np.infty if biggest else np.infty) for (metric, biggest) in self.metrics
        }

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = defaultdict(list)
        
        self.last_time = time.time()

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(self.parameters["model"], params.lr, params.optimizer)
        logger.info("Optimizer: %s" % type(self.optimizer))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        mod_names = sorted(self.modules.keys())
        if params.nvidia_apex is True:
            modules, optimizer = apex.amp.initialize(
                [self.modules[k] for k in mod_names],
                self.optimizer,
                opt_level=("O%i" % params.amp),
            )
            self.modules = {k: module for k, module in zip(mod_names, modules)}
            self.optimizer = optimizer
        else:
            self.scaler = torch.cuda.amp.GradScaler()

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")

        params = self.params

        # optimizer
        optimizer = self.optimizer

        # regular optimization
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
            optimizer.step()

        # AMP optimization
        elif params.nvidia_apex is True:
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(
                        apex.amp.master_params(self.optimizer), params.clip_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(
                    loss, optimizer, delay_unscale=True
                ) as scaled_loss:
                    scaled_loss.backward()

        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.training_ckpt_dump_path, name)
        logger.info("Saving %s to %s " % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)
        logger.warning("Saving %s to %s  SUCCEED !" % (name, path))

    
    def reload_checkpoint(self, checkpoint_path, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if not os.path.isfile(checkpoint_path):
            logger.warning("\n\n\n   Checkpoint path does not exist, {}\n\n".format(checkpoint_path))
            return

        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            weights = data[k]
            # try:
            if 1:
                weights = data[k]
                v.load_state_dict(weights)
            else:
            # except RuntimeError:  # remove the 'module.'
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            v.requires_grad=requires_grad


        # reload optimizer
        # AMP checkpoint reloading is buggy, we cannot reload optimizer
        # instead, we only reload current iterations / learning rates
        if self.params.amp == -1 or not self.params.nvidia_apex:
            logger.warning("Reloading checkpoint optimizer ...")
            self.optimizer.load_state_dict(data["optimizer"])
        else:
            logger.warning("Not reloading checkpoint optimizer.")
            for group_id, param_group in enumerate(self.optimizer.param_groups):
                if "num_updates" not in param_group:
                    logger.warning("No 'num_updates' for optimizer.")
                    continue
                logger.warning("Reloading 'num_updates' and 'lr' for optimizer.")
                param_group["num_updates"] = data["optimizer"]["param_groups"][
                    group_id
                ]["num_updates"]
                param_group["lr"] = self.optimizer.get_lr_for_step(
                    param_group["num_updates"]
                )

        if self.params.fp16 and not self.params.nvidia_apex:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])
        else:
            assert self.scaler is None and "scaler" not in data

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
       
        logger.warning(f"\n\nReloading checkpoint from {checkpoint_path} SUCCESS!\n")



    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch % self.params.save_periodic == 0
        ):
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores, prefix=None, suffix=None):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            _metric = metric
            if prefix is not None: _metric = prefix + "_" + _metric
            if suffix is not None: _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1

            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[_metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self):
        """
        End the epoch.
        """
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def io_embed_encode(self, samples):
        
        embedder, encoder, decoder = (
            self.modules["embedder"],
            self.modules["encoder"],
            self.modules["decoder"],
        )

        def run_encoder(x1_BTE, len1):
            if self.params.amp == -1 or self.params.nvidia_apex:
                encoded_BTE = encoder("fwd", x=x1_BTE, lengths=len1, causal=False)
            else:
                with torch.cuda.amp.autocast():
                    encoded_BTE = encoder("fwd", x=x1_BTE, lengths=len1, causal=False)
            return encoded_BTE

        iodata_ist2 = samples["ioData_ist2"]
        lsnp_ist2 = cross_sample_pad_aligning_syntax(iodata_ist2)
        tensor_ist2, samp_lens_per_inst, token_lens_per_inst = cross_instance_pad_io(lsnp_ist2)
        tensor_ist2 = torch.LongTensor(tensor_ist2)
        samp_lens_per_inst = torch.LongTensor(samp_lens_per_inst)
        token_lens_per_inst = torch.LongTensor(token_lens_per_inst)

        if self.params.nlp_arch == 'distilbert':
            description_it = samples["desc_bert_it"]
        elif self.params.nlp_arch == 'bert':
            description_it = samples["desc_distilbert_it"]
        else:
            raise NotImplementedError

        if self.params.torch_parallel:
            _embedder = embedder.module
        else:
            _embedder = embedder

        def pad(input_ids_BT: list):
            lens = [len(x) for x in input_ids_BT]
            maxl = max(lens)
            inp_tensor = [x + [_embedder.nlp_pad_token_id]*(maxl - len(x)) for x in input_ids_BT]
            attn_mask = [[1]*len(x) + [0]*(maxl - len(x)) for x in input_ids_BT]
            inp_tensor = torch.tensor(inp_tensor)
            attn_mask = torch.ByteTensor(attn_mask)
            return inp_tensor, attn_mask
        desc_inp_tensor, desc_attn_mask = pad(description_it) if self.params.use_pretrained_NLP else [None, None]


        tensor_ist2, samp_lens_per_inst, token_lens_per_inst, desc_inp_tensor, desc_attn_mask = to_cuda(tensor_ist2, samp_lens_per_inst, token_lens_per_inst, desc_inp_tensor, desc_attn_mask, use_cpu=self.params.run_on_cpu) # int valued
    
        x1_BTE, len1 = embedder(tensor_ist2, samp_lens_per_inst, token_lens_per_inst, desc_inp_tensor, desc_attn_mask)
        encoded_BTE = run_encoder(x1_BTE, len1)
        return encoded_BTE, len1

    def enc_dec_step(self, samples):
        """
        Encoding / decoding step.
        """
        params = self.params
        embedder, encoder, decoder = (
            self.modules["embedder"],
            self.modules["encoder"],
            self.modules["decoder"],
        )

        embedder.train()
        encoder.train()
        decoder.train()

        def code_formatter(lsnp_it2):
        
            code_BT, lens_i = cross_instance_pad_code(lsnp_it2)
            code_BT = torch.LongTensor(code_BT)
            lens_i = torch.LongTensor(lens_i)
            code_BT, lens_i = to_cuda(code_BT, lens_i, use_cpu=self.params.run_on_cpu)
            return code_BT, lens_i


        def run_decoder(code_BT, len2, len1, encoded_BTE):
            if params.amp == -1 or params.nvidia_apex:
                decoded = decoder(
                    "fwd",
                    x=code_BT,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded_BTE,
                    src_len=len1,
                )
                _, loss = decoder(
                    "predict", tensor=decoded, len2=len2, code_BT=code_BT
                )
            else:
                with torch.cuda.amp.autocast():
                    decoded = decoder(
                        "fwd",
                        x=code_BT,
                        lengths=len2,
                        causal=True,
                        src_enc=encoded_BTE,
                        src_len=len1,
                    )
                    _, loss = decoder(
                        "predict", tensor=decoded, len2=len2, code_BT=code_BT
                    )
            if params.torch_parallel:
                loss = loss.mean()

            return loss


        if samples['ioData_ist2']==[]:  # ---- pretrain
            len1 = to_cuda(torch.tensor([1]), use_cpu=self.params.run_on_cpu)[0].repeat(self.params.batch_size) # 1D embedding
            encoded_BTE = embedder.embeddings(to_cuda(torch.tensor(embedder.pretrain_pad_emb_idx), use_cpu=self.params.run_on_cpu)[0]).unsqueeze(0).unsqueeze(0).repeat(self.params.batch_size,1,1)
        
        else:
            encoded_BTE, len1 = self.io_embed_encode(samples)


        # 🟩 run decoder
        sum_loss = None
        for code_it2 in samples["program_sit2"]:

            code_BT, len2 = code_formatter(code_it2)

            safe_len = int(params.transformer_max_seq_len*0.9)
            if code_BT.shape[1]>=safe_len:
                code_BT = code_BT[:,:safe_len]
            
            loss = run_decoder(code_BT, len2, len1, encoded_BTE)
            sum_loss = loss if sum_loss is None else sum_loss + loss


        sum_loss = sum_loss/len(samples["program_sit2"])
        # 🟩 optimize
        self.optimize(sum_loss)

