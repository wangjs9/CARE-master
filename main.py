import os
from copy import deepcopy

import numpy as np
import torch
from torch.nn.init import xavier_uniform_
from torch.utils.data import RandomSampler
from tqdm import tqdm, trange

from utils import config
from utils.data_loader import prepare_data_seq, collate_fn

from models.GraphCVAE import CAREModel
from models._common_layer import evaluate, count_parameters


def find_model_path(save_path, recover_path):
    if recover_path == '':
        model_path = os.path.join(save_path, 'best_model')
    else:
        model_path = os.path.join(save_path, recover_path)
    print('load model from {}'.format(model_path))
    return model_path


def train_eval():
    dataset_train, data_loader_val, data_loader_tst, vocab = prepare_data_seq(batch_size=config.bz)
    setup_seed = config.seed
    torch.manual_seed(setup_seed)
    torch.cuda.manual_seed_all(setup_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(setup_seed)

    model_file_path = find_model_path(config.save_path, config.recover_path)

    model = CAREModel(vocab, config.embed_dim, config.hidden_dim, config.num_emotion,
                      model_file_path=model_file_path, load_optim=True)

    if model_file_path is None:
        for n, p in model.named_parameters():
            if p.dim() > 1 and n != "embedding.lut.weight" and config.embed_path != '':
                xavier_uniform_(p)

    if config.test:
        print('Test model')
        print('TRAINABLE PARAMETERS', count_parameters(model))
        model.to(config.device)
        model = model.eval()
        loss, ppl, kld_loss, bleu_score_b = evaluate(model, data_loader_tst, ty="test", max_dec_step=50, save=True)
        if config.max_k == -1:
            file_summary = config.save_path + "{}-{}-summary.txt".format(model.i_epoch, model.i_step)
        else:
            file_summary = config.save_path + "{}-{}-{}-summary.txt".format(model.i_epoch, model.i_step, config.max_k)
        with open(file_summary, 'w') as the_file:
            the_file.write("EVAL{}Loss{}PPL{}Bleu_b\n".format(' ' * 4, ' ' * 4, ' ' * 5))
            the_file.write("{:<8}{:<8.4f}{:<8.4f}{:<8.2f}\n".format("test", loss, ppl, bleu_score_b))
        exit()

    print('TRAINABLE PARAMETERS', count_parameters(model))

    model.to(config.device)
    print('Training from epoch {}, with loss {}'.format(model.i_epoch, model.current_loss))
    best_ppl = model.current_loss
    start_epoch = model.i_epoch
    patient = 0
    weights_best = deepcopy(model.state_dict())
    batch_size = config.bz
    global_step = 0
    try:
        for i_epoch in trange(start_epoch + 1, start_epoch + 60, desc="Epoch", disable=False):
            train_sampler = RandomSampler(dataset_train, replacement=False)
            data_loader_tra = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=6,
                batch_size=batch_size,
                sampler=train_sampler,
                collate_fn=collate_fn,
                pin_memory=False
            )
            model = model.train()
            pbar = tqdm(data_loader_tra, position=0, leave=True, ncols=100)
            ppl_list = []
            for step, batch in enumerate(pbar):
                global_step += 1
                loss, ppl, other_loss = model.train_one_batch(batch, iter_idx=global_step)
                pbar.set_description(
                    'Training %2d: gen %5.2f ppl %5.2f, other %5.2f' % (i_epoch, loss, ppl, other_loss))
                ppl_list.append(ppl)
                if global_step % 1000 == 0 and i_epoch > 19:
                    with torch.no_grad():
                        model = model.eval()
                        loss, ppl, other_loss, _ = evaluate(
                            model,
                            data_loader_val,
                            ty="valid",
                            max_dec_step=50
                        )

                    if ppl <= best_ppl:
                        best_ppl = ppl
                        patient = 0
                    else:
                        patient += 1
                    model.current_loss = ppl
                    model.save_model()
                    weights_best = deepcopy(model.state_dict())

            pbar.close()
            model.i_epoch += 1
            if patient > 8:
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('KeyboardInterrupt: Exiting from training early')
        torch.cuda.empty_cache()

    except AssertionError:
        print('-' * 89)
        print('AssertionError: Exiting from training early')
        torch.cuda.empty_cache()

    model.load_state_dict({name: weights_best[name] for name in weights_best})
    model.eval()
    model.epoch = 1000
    loss_test, ppl_test, kld_loss_test, bleu_score_b = evaluate(
        model,
        data_loader_tst,
        ty="test",
        max_dec_step=50,
        save=True
    )

    file_summary = config.save_path + "{}-{}-summary.txt".format(model.i_epoch, model.i_step)
    with open(file_summary, 'w') as the_file:
        the_file.write("{}{}{}{}\n".format('EVAL'.ljust(8), 'Loss'.ljust(8), 'PPL'.ljust(8), 'Bleu_b'.ljust(8)))
        the_file.write("{:<8}{:<8.2f}{:<8.2f}{:<8.2f}\n".format("test", loss_test, ppl_test, bleu_score_b))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    config.print_opts(config.arg)
    train_eval()
